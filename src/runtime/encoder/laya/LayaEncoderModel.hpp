// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "compute/ComputeBuffer.hpp"
#include "core/gguf/GgufTypes.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace mimirmind::compute {
class ComputeOps;
} // namespace mimirmind::compute

namespace mimirmind::runtime::encoder::laya {

/**
 * Static config of a ModernBERT encoder (the backbone of the "Laya" System-One
 * typed-decision model, `convaiinnovations/laya`), parsed from the HF
 * `config.json` (`model_type: modernbert`).
 *
 * ModernBERT differs from the XLM-RoBERTa encoder in EncoderConfig/EncoderModel:
 *   - pre-LN (norm before attention and MLP), a final norm, and NO biases
 *     (attention_bias = mlp_bias = norm_bias = false);
 *   - rotary positions (RoPE), NO learned absolute/token-type embeddings;
 *   - alternating attention: every `globalAttnEveryNLayers`-th layer is global
 *     (full window), the rest are local sliding-window (`localAttention` total
 *     width, i.e. ±localAttention/2), with a separate RoPE theta per kind;
 *   - fused QKV (`attn.Wqkv`, [3*hidden, hidden]) and GeGLU MLP
 *     (`mlp.Wi`, [2*ffn, hidden] = gate+input), activation gelu;
 *   - layer 0's `attn_norm` is Identity (the embedding norm already ran).
 *
 * This is the backbone only (tensor prefix `encoder.`). The Laya decision head
 * (`act_head` + `temperature`) is a separate, laya-specific piece.
 */
struct LayaEncoderConfig {
    std::size_t  numLayers{0};
    std::size_t  hidden{0};
    std::size_t  heads{0};
    std::size_t  headDim{0};                 // hidden / heads
    std::size_t  ffn{0};                     // intermediate_size
    std::size_t  vocab{0};
    std::size_t  maxPos{0};
    std::size_t  localAttention{128};        // full local-window width (±width/2)
    std::size_t  globalAttnEveryNLayers{3};  // layer i is global iff i % this == 0
    float        ropeThetaGlobal{160000.0F};
    float        ropeThetaLocal{10000.0F};
    float        normEps{1e-5F};
    std::int32_t padTokenId{0};
    std::int32_t clsTokenId{0};
    bool         meanPooling{true};          // classifier_pooling: "mean" (laya) vs "cls"

    /// Whether layer `i` uses global (full-window) attention. Local otherwise.
    [[nodiscard]] bool isGlobalLayer(std::size_t i) const noexcept {
        return globalAttnEveryNLayers == 0 || (i % globalAttnEveryNLayers) == 0;
    }
    /// RoPE base for layer `i` (global layers use the long-context base).
    [[nodiscard]] float ropeTheta(std::size_t i) const noexcept {
        return isGlobalLayer(i) ? ropeThetaGlobal : ropeThetaLocal;
    }
    /// Local sliding-window radius in tokens (each side), 0 for a global layer.
    [[nodiscard]] std::size_t localRadius(std::size_t i) const noexcept {
        return isGlobalLayer(i) ? 0 : (localAttention / 2);
    }
};

/// Parse the fields LayaEncoderConfig needs out of an HF ModernBERT
/// `config.json`. Throws if `model_type` is not `modernbert` or if any bias
/// flag (attention_bias/mlp_bias/norm_bias) is set (the native forward is
/// bias-free; a bias'd checkpoint would need extra tensors/kernels).
[[nodiscard]] LayaEncoderConfig parseLayaEncoderConfig(std::string_view configJson);

/// Per-layer device weight pointers (all in USM). Linear weights are stored as
/// LayaEncoderModel::matmulType() (F32 or BF16) in HF Linear layout [out, in]
/// = ComputeMatmul's `W[N,K]`; norms stay F32. No biases exist in ModernBERT.
struct LayaLayerWeights {
    const float* attnNormW{};  // pre-attention norm; NULL on layer 0 (Identity)
    const float* qkvW{};       // fused Q|K|V, [3*hidden, hidden]
    const float* oW{};         // attention output, [hidden, hidden]
    const float* mlpNormW{};   // pre-MLP norm
    const float* wiW{};        // GeGLU in, fused input|gate, [2*ffn, hidden]
    const float* woW{};        // MLP out, [hidden, ffn]
};

/**
 * Loads a ModernBERT encoder backbone (the Laya checkpoints ship F16; this
 * loader also accepts BF16/F32) from HF safetensors straight into USM — no GGUF
 * conversion. The generic `core::safetensors::SafetensorsModel` reader parses
 * the container; this maps the ModernBERT tensor names to encoder roles,
 * converts the stored dtype, and uploads. Backend-neutral (uploads via the
 * abstract ComputeOps).
 *
 * Tensor names (prefix configurable, default `encoder.`):
 *   <p>embeddings.tok_embeddings.weight, <p>embeddings.norm.weight,
 *   <p>layers.{i}.{attn_norm,mlp_norm}.weight,
 *   <p>layers.{i}.attn.{Wqkv,Wo}.weight, <p>layers.{i}.mlp.{Wi,Wo}.weight,
 *   <p>final_norm.weight.  (layer 0 attn_norm absent = Identity.)
 */
class LayaEncoderModel {
public:
    LayaEncoderModel() = default;

    LayaEncoderModel(const LayaEncoderModel&)            = delete;
    LayaEncoderModel& operator=(const LayaEncoderModel&) = delete;
    LayaEncoderModel(LayaEncoderModel&&) noexcept            = default;
    LayaEncoderModel& operator=(LayaEncoderModel&&) noexcept = default;

    /// `dir` is a folder with config.json + model.safetensors (or a sharded
    /// index), or a direct .safetensors path (config.json must sit next to it).
    /// `matmulType` is the dtype the dense LINEAR weights are stored/matmul'd
    /// as: F32 (exact — parity) or BF16 (tensor-core GEMM at M>1); norms always
    /// stay F32. `tensorPrefix` selects the backbone namespace inside the
    /// container (laya wraps ModernBERT under `encoder.`). Throws on a missing
    /// tensor or an unsupported stored dtype.
    void load(std::string_view dir, compute::ComputeOps& ops,
              core::gguf::GgmlType matmulType = core::gguf::GgmlType::F32,
              std::string_view tensorPrefix = "encoder.");

    [[nodiscard]] const LayaEncoderConfig& config() const noexcept { return _config; }
    [[nodiscard]] core::gguf::GgmlType matmulType() const noexcept { return _matmulType; }
    [[nodiscard]] std::size_t weightBytes() const noexcept { return _weightBytes; }

    // Embeddings block.
    [[nodiscard]] const float* tokEmb() const noexcept { return _tokEmb; }
    [[nodiscard]] const float* embNormW() const noexcept { return _embNormW; }
    [[nodiscard]] const float* finalNormW() const noexcept { return _finalNormW; }

    [[nodiscard]] const LayaLayerWeights& layer(std::size_t i) const {
        return _layers.at(i);
    }

private:
    LayaEncoderConfig    _config{};
    core::gguf::GgmlType _matmulType{core::gguf::GgmlType::F32};

    std::vector<compute::ComputeBuffer> _owned;   // lifetime of every USM buffer
    std::size_t _weightBytes{0};
    std::vector<LayaLayerWeights> _layers;

    const float* _tokEmb{};
    const float* _embNormW{};
    const float* _finalNormW{};
};

} // namespace mimirmind::runtime::encoder::laya
