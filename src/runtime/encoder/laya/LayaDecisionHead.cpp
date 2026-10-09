// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/encoder/laya/LayaDecisionHead.hpp"

#include "compute/ComputeOps.hpp"
#include "core/gpu/AllocCategory.hpp"
#include "core/safetensors/SafetensorsDtype.hpp"
#include "core/safetensors/SafetensorsModel.hpp"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace mimirmind::runtime::encoder::laya {

namespace {

float f16ToF32(std::uint16_t h) {
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
    const std::uint32_t exp  = (h >> 10) & 0x1Fu;
    const std::uint32_t mant = h & 0x3FFu;
    std::uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            std::uint32_t e = 127 - 15 + 1;
            std::uint32_t m = mant;
            while ((m & 0x400u) == 0) { m <<= 1; --e; }
            m &= 0x3FFu;
            bits = sign | (e << 23) | (m << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

float bf16ToF32(std::uint16_t b) {
    const std::uint32_t bits = static_cast<std::uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

std::uint16_t f32ToBf16(float f) {
    std::uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const std::uint32_t rounding = 0x7FFFu + ((x >> 16) & 1u);
    return static_cast<std::uint16_t>((x + rounding) >> 16);
}

} // namespace

void LayaDecisionHead::load(std::string_view dir, compute::ComputeOps& ops,
                            std::size_t hidden, std::size_t nHeadLayers,
                            core::gguf::GgmlType matmulType) {
    core::gpu::ScopedAllocCategory _wc{core::gpu::AllocCategory::Weights};
    _matmulType = matmulType;
    _config.hidden      = hidden;
    _config.headDim     = 64;
    _config.heads       = hidden / 64;
    _config.ffn         = 4 * hidden;
    _config.nHeadLayers = nHeadLayers;
    _config.nAct        = 2;

    core::safetensors::SafetensorsModel sm;
    sm.open(dir);

    auto readAsF32 = [&](const std::string& name) -> std::vector<float> {
        const auto* t = sm.find(name);
        if (t == nullptr) {
            throw std::runtime_error("LayaDecisionHead: missing tensor '" + name + "'");
        }
        const std::span<const std::uint8_t> bytes = sm.tensorBytes(name);
        using core::safetensors::SafetensorsDtype;
        std::vector<float> out;
        switch (t->dtype) {
        case SafetensorsDtype::F32: {
            const std::size_t n = bytes.size() / sizeof(float);
            out.resize(n);
            std::memcpy(out.data(), bytes.data(), n * sizeof(float));
            break;
        }
        case SafetensorsDtype::F16: {
            const std::size_t n = bytes.size() / sizeof(std::uint16_t);
            const auto* src = reinterpret_cast<const std::uint16_t*>(bytes.data());
            out.resize(n);
            for (std::size_t i = 0; i < n; ++i) { out[i] = f16ToF32(src[i]); }
            break;
        }
        case SafetensorsDtype::BF16: {
            const std::size_t n = bytes.size() / sizeof(std::uint16_t);
            const auto* src = reinterpret_cast<const std::uint16_t*>(bytes.data());
            out.resize(n);
            for (std::size_t i = 0; i < n; ++i) { out[i] = bf16ToF32(src[i]); }
            break;
        }
        default:
            throw std::runtime_error(
                "LayaDecisionHead: tensor '" + name + "' has unsupported dtype '" +
                std::string(core::safetensors::dtypeName(t->dtype)) + "'");
        }
        return out;
    };

    auto upload = [&](const std::string& name) -> const float* {
        const std::vector<float> host = readAsF32(name);
        compute::ComputeBuffer buf = ops.allocate(host.size() * sizeof(float));
        _weightBytes += host.size() * sizeof(float);
        ops.uploadHostBytes(buf.get(), host.data(), host.size() * sizeof(float));
        const float* p = static_cast<const float*>(buf.get());
        _owned.push_back(std::move(buf));
        return p;
    };

    auto uploadW = [&](const std::string& name) -> const float* {
        const std::vector<float> host = readAsF32(name);
        const std::size_t n = host.size();
        if (_matmulType == core::gguf::GgmlType::BF16) {
            std::vector<std::uint16_t> bf(n);
            for (std::size_t i = 0; i < n; ++i) { bf[i] = f32ToBf16(host[i]); }
            compute::ComputeBuffer buf = ops.allocate(n * sizeof(std::uint16_t));
            _weightBytes += n * sizeof(std::uint16_t);
            ops.uploadHostBytes(buf.get(), bf.data(), n * sizeof(std::uint16_t));
            const auto* p = reinterpret_cast<const float*>(buf.get());
            _owned.push_back(std::move(buf));
            return p;
        }
        compute::ComputeBuffer buf = ops.allocate(n * sizeof(float));
        _weightBytes += n * sizeof(float);
        ops.uploadHostBytes(buf.get(), host.data(), n * sizeof(float));
        const float* p = static_cast<const float*>(buf.get());
        _owned.push_back(std::move(buf));
        return p;
    };

    _typeEmb     = upload("type_emb.weight");
    _temperature = upload("temperature");

    _layers.resize(nHeadLayers);
    for (std::size_t i = 0; i < nHeadLayers; ++i) {
        const std::string pre = "head.layers." + std::to_string(i) + ".";
        LayaHeadLayerWeights& L = _layers[i];
        L.inProjW  = uploadW(pre + "self_attn.in_proj_weight");
        L.inProjB  = upload(pre + "self_attn.in_proj_bias");
        L.outProjW = uploadW(pre + "self_attn.out_proj.weight");
        L.outProjB = upload(pre + "self_attn.out_proj.bias");
        L.norm1W   = upload(pre + "norm1.weight");
        L.norm1B   = upload(pre + "norm1.bias");
        L.norm2W   = upload(pre + "norm2.weight");
        L.norm2B   = upload(pre + "norm2.bias");
        L.lin1W    = uploadW(pre + "linear1.weight");
        L.lin1B    = upload(pre + "linear1.bias");
        L.lin2W    = uploadW(pre + "linear2.weight");
        L.lin2B    = upload(pre + "linear2.bias");
    }

    _scorerLnW = upload("scorer.0.weight");
    _scorerLnB = upload("scorer.0.bias");
    _scorerW1  = uploadW("scorer.1.weight");
    _scorerB1  = upload("scorer.1.bias");
    _scorerW3  = uploadW("scorer.3.weight");
    _scorerB3  = upload("scorer.3.bias");

    _actW0 = uploadW("act_head.0.weight");
    _actB0 = upload("act_head.0.bias");
    _actW2 = uploadW("act_head.2.weight");
    _actB2 = upload("act_head.2.bias");
}

} // namespace mimirmind::runtime::encoder::laya
