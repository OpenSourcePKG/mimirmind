// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/dflash/DFlashDraftModel.hpp"

#include "compute/ComputeOps.hpp"
#include "core/gguf/GgufReader.hpp"
#include "core/safetensors/SafetensorsDtype.hpp"
#include "core/safetensors/SafetensorsHeader.hpp"
#include "core/safetensors/SafetensorsModel.hpp"

#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace mimirmind::runtime::dflash {

namespace {

namespace fe = mimirmind::core::safetensors;

std::string layerPrefix(std::size_t i) {
    return "layers." + std::to_string(i) + ".";
}

std::string shapeStr(const std::vector<std::uint64_t>& s) {
    std::string out = "[";
    for (std::size_t i = 0; i < s.size(); ++i) {
        out += std::to_string(s[i]);
        if (i + 1 < s.size()) { out += ", "; }
    }
    out += "]";
    return out;
}

} // namespace

void DFlashDraftModel::load(std::string_view dir, compute::ComputeOps& ops) {
    fe::SafetensorsModel sm;
    sm.open(dir);

    // Require a BF16 tensor by name and return its descriptor (shape/bytes).
    auto require = [&](const std::string& name) -> const fe::SafetensorsTensor* {
        const auto* t = sm.find(name);
        if (t == nullptr) {
            throw std::runtime_error("DFlashDraftModel: missing tensor '" + name + "'");
        }
        if (t->dtype != fe::SafetensorsDtype::BF16) {
            throw std::runtime_error("DFlashDraftModel: tensor '" + name +
                                     "' is not BF16 (draft checkpoint must be BF16)");
        }
        return t;
    };

    // --- Derive architecture dims from tensor shapes (self-describing) -------
    const auto& normShape = require("norm.weight")->shape;
    if (normShape.size() != 1) {
        throw std::runtime_error("DFlashDraftModel: norm.weight must be 1-D, got " +
                                 shapeStr(normShape));
    }
    const std::uint64_t H = normShape[0];

    const auto& qNormShape = require(layerPrefix(0) + "self_attn.q_norm.weight")->shape;
    if (qNormShape.size() != 1) {
        throw std::runtime_error("DFlashDraftModel: q_norm.weight must be 1-D, got " +
                                 shapeStr(qNormShape));
    }
    const std::uint64_t HD = qNormShape[0];

    const std::uint64_t Q  = require(layerPrefix(0) + "self_attn.q_proj.weight")->shape.at(0);
    const std::uint64_t KV = require(layerPrefix(0) + "self_attn.k_proj.weight")->shape.at(0);
    const std::uint64_t I  = require(layerPrefix(0) + "mlp.gate_proj.weight")->shape.at(0);

    const auto& fcShape = require("fc.weight")->shape;
    if (fcShape.size() != 2 || fcShape[0] != H || (fcShape[1] % H) != 0) {
        throw std::runtime_error("DFlashDraftModel: fc.weight expected [hidden, taps*hidden], got " +
                                 shapeStr(fcShape));
    }
    const std::uint64_t taps = fcShape[1] / H;

    if (HD == 0 || (Q % HD) != 0 || (KV % HD) != 0) {
        throw std::runtime_error("DFlashDraftModel: q/k dims not divisible by head_dim");
    }

    // Count layers by probing until a layer's input_layernorm is absent.
    std::size_t numLayers = 0;
    constexpr std::size_t kMaxProbe = 128;
    while (numLayers < kMaxProbe &&
           sm.find(layerPrefix(numLayers) + "input_layernorm.weight") != nullptr) {
        ++numLayers;
    }
    if (numLayers == 0) {
        throw std::runtime_error("DFlashDraftModel: no transformer layers found");
    }

    _config = DFlashDraftConfig{
        .numLayers = numLayers,
        .hidden    = static_cast<std::size_t>(H),
        .headDim   = static_cast<std::size_t>(HD),
        .nQHeads   = static_cast<std::size_t>(Q / HD),
        .nKvHeads  = static_cast<std::size_t>(KV / HD),
        .inter     = static_cast<std::size_t>(I),
        .taps      = static_cast<std::size_t>(taps),
    };

    // --- Upload one BF16 tensor verbatim, validating its expected shape -----
    auto upload = [&](const std::string& name,
                      const std::vector<std::uint64_t>& want) -> const std::uint16_t* {
        const auto* t = require(name);
        if (t->shape != want) {
            throw std::runtime_error("DFlashDraftModel: tensor '" + name +
                                     "' shape " + shapeStr(t->shape) +
                                     " != expected " + shapeStr(want));
        }
        const std::span<const std::uint8_t> bytes = sm.tensorBytes(name);
        compute::ComputeBuffer buf = ops.allocate(bytes.size());
        ops.uploadHostBytes(buf.get(), bytes.data(), bytes.size());
        const auto* p = static_cast<const std::uint16_t*>(buf.get());
        _owned.push_back(std::move(buf));
        _uploadedBytes += bytes.size();
        return p;
    };

    // Upload a 1-D RMSNorm weight, converting BF16 -> F32 on host (the
    // ComputeOps::rmsNormAsync path takes a `const float* weight`). BF16 is the
    // upper 16 bits of the F32 pattern, so the widen is a shift. These tensors
    // are small ([hidden] / [head_dim]).
    auto uploadF32 = [&](const std::string& name,
                         const std::vector<std::uint64_t>& want) -> const float* {
        const auto* t = require(name);
        if (t->shape != want) {
            throw std::runtime_error("DFlashDraftModel: tensor '" + name +
                                     "' shape " + shapeStr(t->shape) +
                                     " != expected " + shapeStr(want));
        }
        const std::span<const std::uint8_t> bytes = sm.tensorBytes(name);
        const std::size_t n = bytes.size() / sizeof(std::uint16_t);
        const auto* bf = reinterpret_cast<const std::uint16_t*>(bytes.data());
        std::vector<float> f(n);
        for (std::size_t i = 0; i < n; ++i) {
            const std::uint32_t bits = static_cast<std::uint32_t>(bf[i]) << 16;
            std::memcpy(&f[i], &bits, sizeof(float));
        }
        compute::ComputeBuffer buf = ops.allocate(n * sizeof(float));
        ops.uploadHostBytes(buf.get(), f.data(), n * sizeof(float));
        const auto* p = static_cast<const float*>(buf.get());
        _owned.push_back(std::move(buf));
        _uploadedBytes += n * sizeof(float);
        return p;
    };

    // --- Transformer layers -------------------------------------------------
    _layers.resize(numLayers);
    for (std::size_t l = 0; l < numLayers; ++l) {
        const std::string p = layerPrefix(l);
        DFlashDraftLayerWeights& w = _layers[l];
        w.qProj      = upload(p + "self_attn.q_proj.weight", {Q, H});
        w.kProj      = upload(p + "self_attn.k_proj.weight", {KV, H});
        w.vProj      = upload(p + "self_attn.v_proj.weight", {KV, H});
        w.oProj      = upload(p + "self_attn.o_proj.weight", {H, Q});
        w.qNorm      = uploadF32(p + "self_attn.q_norm.weight", {HD});
        w.kNorm      = uploadF32(p + "self_attn.k_norm.weight", {HD});
        w.inputLn    = uploadF32(p + "input_layernorm.weight", {H});
        w.postAttnLn = uploadF32(p + "post_attention_layernorm.weight", {H});
        w.gateProj   = upload(p + "mlp.gate_proj.weight", {I, H});
        w.upProj     = upload(p + "mlp.up_proj.weight", {I, H});
        w.downProj   = upload(p + "mlp.down_proj.weight", {H, I});
    }

    // --- DFlash-specific top-level tensors ----------------------------------
    _fc         = upload("fc.weight", {H, taps * H});      // bf16 linear
    _hiddenNorm = uploadF32("hidden_norm.weight", {H});    // f32 RMSNorm weight
    _norm       = uploadF32("norm.weight", {H});           // f32 RMSNorm weight

    // The drafter must NOT ship its own embed/lm_head — those are borrowed.
    for (const char* n : {"embed_tokens.weight", "lm_head.weight",
                          "model.embed_tokens.weight"}) {
        if (sm.find(n) != nullptr) {
            throw std::runtime_error(std::string("DFlashDraftModel: unexpected '") + n +
                                     "' (embed/lm_head are borrowed from the target)");
        }
    }
}

void DFlashDraftModel::borrowTarget(const core::gguf::GgufTensor* embedTokens,
                                    const core::gguf::GgufTensor* lmHead) {
    auto hasHiddenDim = [&](const core::gguf::GgufTensor* t) -> bool {
        for (const auto d : t->dimensions) {
            if (static_cast<std::size_t>(d) == _config.hidden) { return true; }
        }
        return false;
    };
    if (embedTokens != nullptr && _config.hidden != 0 && !hasHiddenDim(embedTokens)) {
        throw std::runtime_error(
            "DFlashDraftModel::borrowTarget: embed_tokens has no dim == drafter hidden");
    }
    if (lmHead != nullptr && _config.hidden != 0 && !hasHiddenDim(lmHead)) {
        throw std::runtime_error(
            "DFlashDraftModel::borrowTarget: lm_head has no dim == drafter hidden");
    }
    _embed  = embedTokens;
    _lmHead = lmHead;
}

} // namespace mimirmind::runtime::dflash
