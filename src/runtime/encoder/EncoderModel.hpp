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

namespace mimirmind::runtime::encoder {

/**
 * Static config of a BERT/RoBERTa/XLM-R bidirectional encoder, parsed from
 * the HF `config.json`. This is the cross-encoder reranker path (EncoderRunner)
 * — not a decoder, so no rope/kv/sampler fields.
 */
struct EncoderConfig {
    std::size_t numLayers{0};
    std::size_t hidden{0};
    std::size_t heads{0};
    std::size_t headDim{0};       // hidden / heads
    std::size_t ffn{0};
    std::size_t vocab{0};
    std::size_t maxPos{0};
    std::size_t typeVocab{1};
    std::size_t numLabels{1};     // 1 for a reranker logit
    std::int32_t padTokenId{1};
    std::size_t posOffset{2};     // learned-abs positions: padTokenId + 1
    float lnEps{1e-5F};
};

/// Parse the fields EncoderConfig needs out of an HF `config.json` string.
[[nodiscard]] EncoderConfig parseEncoderConfig(std::string_view configJson);

/// Per-layer device weight pointers (all F32 in USM, HF Linear layout
/// [out, in] which matches ComputeMatmul's `W[N,K]`).
struct EncoderLayerWeights {
    const float* qW{};   const float* qB{};
    const float* kW{};   const float* kB{};
    const float* vW{};   const float* vB{};
    const float* aoW{};  const float* aoB{};   // attention.output.dense
    const float* aoLnW{}; const float* aoLnB{}; // attention.output.LayerNorm
    const float* fiW{};  const float* fiB{};   // intermediate.dense
    const float* foW{};  const float* foB{};   // output.dense
    const float* outLnW{}; const float* outLnB{}; // output.LayerNorm
};

/**
 * Loads a dense F32 XLM-R sequence-classification checkpoint (e.g.
 * bge-reranker-v2-m3) from HF safetensors straight into USM — no GGUF
 * conversion. The generic `core::safetensors::SafetensorsModel` reader does
 * the parsing; this only maps HF tensor names to encoder roles and uploads
 * the bytes. Backend-neutral: uploads go through the abstract ComputeOps.
 *
 * All weight tensors must be F32 (torch_dtype float32 for bge-reranker-v2-m3).
 */
class EncoderModel {
public:
    EncoderModel() = default;

    EncoderModel(const EncoderModel&)            = delete;
    EncoderModel& operator=(const EncoderModel&) = delete;
    EncoderModel(EncoderModel&&) noexcept            = default;
    EncoderModel& operator=(EncoderModel&&) noexcept = default;

    /// `dir` is a folder with config.json + model.safetensors (or a sharded
    /// index) or a direct .safetensors path (config.json must sit next to it).
    /// `matmulType` is the dtype the dense LINEAR weights are stored/matmul'd
    /// as: F32 (exact, slow — parity) or BF16 (tensor-core GEMM at M>1, the
    /// fast serving path — activations stay F32, ranking-grade precision).
    /// Biases, LayerNorm and embeddings always stay F32. Throws on a missing/
    /// wrong-dtype tensor.
    void load(std::string_view dir, compute::ComputeOps& ops,
              core::gguf::GgmlType matmulType = core::gguf::GgmlType::F32);

    [[nodiscard]] const EncoderConfig& config() const noexcept { return _config; }

    /// The GgmlType the linear weights are stored as (F32 or BF16) — the type
    /// EncoderRunner must pass to ComputeMatmul::matmulAsync for them.
    [[nodiscard]] core::gguf::GgmlType matmulType() const noexcept { return _matmulType; }

    // Embeddings block.
    [[nodiscard]] const float* wordEmb() const noexcept { return _wordEmb; }
    [[nodiscard]] const float* posTable() const noexcept { return _posTable; }
    [[nodiscard]] const float* typeVec() const noexcept { return _typeVec; }
    [[nodiscard]] const float* embLnW() const noexcept { return _embLnW; }
    [[nodiscard]] const float* embLnB() const noexcept { return _embLnB; }

    [[nodiscard]] const EncoderLayerWeights& layer(std::size_t i) const {
        return _layers.at(i);
    }

    // Classifier head (RobertaClassificationHead: dense -> tanh -> out_proj).
    // Present only on sequence-classification checkpoints (rerankers). A pure
    // embedding checkpoint (e.g. bge-m3 dense) ships the encoder + embeddings
    // only — `hasClassifier()` is false and the head pointers stay null; the
    // EncoderRunner::embed() path (CLS-pool + L2-norm) needs no head.
    [[nodiscard]] bool hasClassifier() const noexcept { return _hasClassifier; }
    [[nodiscard]] const float* clsDenseW() const noexcept { return _clsDenseW; }
    [[nodiscard]] const float* clsDenseB() const noexcept { return _clsDenseB; }
    [[nodiscard]] const float* clsOutW() const noexcept { return _clsOutW; }
    [[nodiscard]] const float* clsOutB() const noexcept { return _clsOutB; }

private:
    EncoderConfig       _config{};
    core::gguf::GgmlType _matmulType{core::gguf::GgmlType::F32};

    std::vector<compute::ComputeBuffer> _owned;   // lifetime of every USM buffer
    std::vector<EncoderLayerWeights>    _layers;

    const float* _wordEmb{};
    const float* _posTable{};
    const float* _typeVec{};
    const float* _embLnW{};
    const float* _embLnB{};
    bool         _hasClassifier{false};
    const float* _clsDenseW{};
    const float* _clsDenseB{};
    const float* _clsOutW{};
    const float* _clsOutB{};
};

} // namespace mimirmind::runtime::encoder
