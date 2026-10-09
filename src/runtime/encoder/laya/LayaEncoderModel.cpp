// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/encoder/laya/LayaEncoderModel.hpp"

#include "compute/ComputeOps.hpp"
#include "core/gpu/AllocCategory.hpp"
#include "core/safetensors/SafetensorsDtype.hpp"
#include "core/safetensors/SafetensorsModel.hpp"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace mimirmind::runtime::encoder::laya {

namespace {

std::string readTextFile(const std::filesystem::path& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) {
        throw std::runtime_error("LayaEncoderModel: cannot read " + p.string());
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// IEEE binary16 -> binary32.
float f16ToF32(std::uint16_t h) {
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
    const std::uint32_t exp  = (h >> 10) & 0x1Fu;
    const std::uint32_t mant = h & 0x3FFu;
    std::uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;                       // +/- zero
        } else {
            // subnormal: normalize
            std::uint32_t e = 127 - 15 + 1;
            std::uint32_t m = mant;
            while ((m & 0x400u) == 0) {
                m <<= 1;
                --e;
            }
            m &= 0x3FFu;
            bits = sign | (e << 23) | (m << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | (mant << 13);  // Inf/NaN
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// bfloat16 (raw 16-bit) -> binary32 (zero-extend the mantissa).
float bf16ToF32(std::uint16_t b) {
    const std::uint32_t bits = static_cast<std::uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// binary32 -> bfloat16 (round-to-nearest-even), raw 16-bit pattern.
std::uint16_t f32ToBf16(float f) {
    std::uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const std::uint32_t rounding = 0x7FFFu + ((x >> 16) & 1u);
    return static_cast<std::uint16_t>((x + rounding) >> 16);
}

} // namespace

LayaEncoderConfig parseLayaEncoderConfig(std::string_view configJson) {
    const nlohmann::json j =
        nlohmann::json::parse(configJson.begin(), configJson.end(), nullptr,
                              /*allow_exceptions=*/true, /*ignore_comments=*/true);

    const std::string modelType = j.value("model_type", std::string{});
    if (modelType != "modernbert") {
        throw std::runtime_error(
            "LayaEncoderModel: config model_type='" + modelType +
            "' is not 'modernbert'");
    }
    // The native forward is bias-free; refuse a checkpoint that needs biases.
    if (j.value("attention_bias", false) || j.value("mlp_bias", false) ||
        j.value("norm_bias", false)) {
        throw std::runtime_error(
            "LayaEncoderModel: attention_bias/mlp_bias/norm_bias set — the "
            "native ModernBERT forward is bias-free and has no bias tensors");
    }

    LayaEncoderConfig c{};
    c.hidden     = j.at("hidden_size").get<std::size_t>();
    c.numLayers  = j.at("num_hidden_layers").get<std::size_t>();
    c.heads      = j.at("num_attention_heads").get<std::size_t>();
    c.headDim    = c.hidden / c.heads;
    c.ffn        = j.at("intermediate_size").get<std::size_t>();
    c.vocab      = j.at("vocab_size").get<std::size_t>();
    c.maxPos     = j.at("max_position_embeddings").get<std::size_t>();
    c.localAttention         = j.value("local_attention", std::size_t{128});
    c.globalAttnEveryNLayers = j.value("global_attn_every_n_layers", std::size_t{3});
    // RoPE base per layer type. ModernBERT defaults are global 160000 / local
    // 10000, but the checkpoint carries the real values. New HF format nests
    // them under `rope_parameters.{full,sliding}_attention.rope_theta` (e.g. the
    // Laya multilingual checkpoint sets BOTH to 160000); the old format used
    // flat `global_rope_theta` / `local_rope_theta`.
    c.ropeThetaGlobal = 160000.0F;
    c.ropeThetaLocal  = 10000.0F;
    if (j.contains("rope_parameters")) {
        const nlohmann::json& rp = j.at("rope_parameters");
        if (rp.contains("full_attention")) {
            c.ropeThetaGlobal = rp.at("full_attention").value("rope_theta", c.ropeThetaGlobal);
        }
        if (rp.contains("sliding_attention")) {
            c.ropeThetaLocal = rp.at("sliding_attention").value("rope_theta", c.ropeThetaLocal);
        }
        if (rp.contains("rope_theta")) {   // flat form: one base for all layers
            const float t = rp.value("rope_theta", c.ropeThetaGlobal);
            c.ropeThetaGlobal = t;
            c.ropeThetaLocal  = t;
        }
    }
    c.ropeThetaGlobal = j.value("global_rope_theta", c.ropeThetaGlobal);
    c.ropeThetaLocal  = j.value("local_rope_theta", c.ropeThetaLocal);
    c.normEps    = j.value("norm_eps", 1e-5F);
    c.padTokenId = j.value("pad_token_id", std::int32_t{0});
    c.clsTokenId = j.value("cls_token_id", std::int32_t{0});
    c.meanPooling = (j.value("classifier_pooling", std::string{"mean"}) == "mean");
    return c;
}

void LayaEncoderModel::load(std::string_view dir, compute::ComputeOps& ops,
                            core::gguf::GgmlType matmulType,
                            std::string_view tensorPrefix) {
    // Tag these weights (they load outside the guarded InferenceEngine path).
    core::gpu::ScopedAllocCategory _wc{core::gpu::AllocCategory::Weights};
    _matmulType = matmulType;
    const std::filesystem::path root{dir};
    // The Laya checkpoints keep the ModernBERT config in an `encoder/`
    // subfolder while the weights (prefix `encoder.`) live in the parent's
    // model.safetensors — so fall back to <dir>/encoder/config.json.
    std::filesystem::path cfgPath = root / "config.json";
    if (!std::filesystem::exists(cfgPath)) {
        cfgPath = root / "encoder" / "config.json";
    }
    _config = parseLayaEncoderConfig(readTextFile(cfgPath));

    core::safetensors::SafetensorsModel sm;
    sm.open(dir);
    const std::string prefix{tensorPrefix};

    // Decode a stored tensor (F16/BF16/F32) to host F32.
    auto readAsF32 = [&](const std::string& name) -> std::vector<float> {
        const auto* t = sm.find(name);
        if (t == nullptr) {
            throw std::runtime_error("LayaEncoderModel: missing tensor '" + name + "'");
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
            for (std::size_t i = 0; i < n; ++i) {
                out[i] = f16ToF32(src[i]);
            }
            break;
        }
        case SafetensorsDtype::BF16: {
            const std::size_t n = bytes.size() / sizeof(std::uint16_t);
            const auto* src = reinterpret_cast<const std::uint16_t*>(bytes.data());
            out.resize(n);
            for (std::size_t i = 0; i < n; ++i) {
                out[i] = bf16ToF32(src[i]);
            }
            break;
        }
        default:
            throw std::runtime_error(
                "LayaEncoderModel: tensor '" + name + "' has unsupported dtype '" +
                std::string(core::safetensors::dtypeName(t->dtype)) + "'");
        }
        return out;
    };

    // Upload one tensor as F32 to USM (norms, embeddings). Returns nullptr if
    // the tensor is absent AND `optional` (layer 0 attn_norm is Identity).
    auto upload = [&](const std::string& name, bool optional = false) -> const float* {
        if (optional && sm.find(name) == nullptr) {
            return nullptr;
        }
        const std::vector<float> host = readAsF32(name);
        compute::ComputeBuffer buf = ops.allocate(host.size() * sizeof(float));
        _weightBytes += host.size() * sizeof(float);
        ops.uploadHostBytes(buf.get(), host.data(), host.size() * sizeof(float));
        const float* p = static_cast<const float*>(buf.get());
        _owned.push_back(std::move(buf));
        return p;
    };

    // Upload a dense LINEAR weight as _matmulType (BF16 for tensor-core GEMM at
    // M>1, or F32 verbatim). Returned as an opaque device pointer the matmul
    // reads per GgmlType.
    auto uploadW = [&](const std::string& name) -> const float* {
        const std::vector<float> host = readAsF32(name);
        const std::size_t n = host.size();
        if (_matmulType == core::gguf::GgmlType::BF16) {
            std::vector<std::uint16_t> bf(n);
            for (std::size_t i = 0; i < n; ++i) {
                bf[i] = f32ToBf16(host[i]);
            }
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

    // Embeddings block (tok embeddings + the embedding LayerNorm; no pos/type).
    _tokEmb     = upload(prefix + "embeddings.tok_embeddings.weight");
    _embNormW   = upload(prefix + "embeddings.norm.weight");

    // Encoder layers.
    _layers.resize(_config.numLayers);
    for (std::size_t i = 0; i < _config.numLayers; ++i) {
        const std::string pre = prefix + "layers." + std::to_string(i) + ".";
        LayaLayerWeights& L = _layers[i];
        // Layer 0's attn_norm is Identity in ModernBERT -> tensor absent.
        L.attnNormW = upload(pre + "attn_norm.weight", /*optional=*/true);
        L.qkvW      = uploadW(pre + "attn.Wqkv.weight");
        L.oW        = uploadW(pre + "attn.Wo.weight");
        L.mlpNormW  = upload(pre + "mlp_norm.weight");
        L.wiW       = uploadW(pre + "mlp.Wi.weight");
        L.woW       = uploadW(pre + "mlp.Wo.weight");
    }

    _finalNormW = upload(prefix + "final_norm.weight");
}

} // namespace mimirmind::runtime::encoder::laya
