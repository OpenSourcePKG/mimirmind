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

namespace mimirmind::runtime::audio {

/**
 * Static config of a Whisper-family encoder-decoder ASR model, parsed from
 * the HF `config.json` (WhisperConfig / WhisperForConditionalGeneration).
 * Encoder and decoder can in principle differ in depth/heads/ffn, so they
 * are tracked separately (they coincide on the stock checkpoints).
 */
struct WhisperConfig {
    std::size_t numMelBins{80};          // 80 (tiny..medium) or 128 (large-v3)
    std::size_t dModel{0};               // hidden width
    std::size_t encoderLayers{0};
    std::size_t decoderLayers{0};
    std::size_t encoderHeads{0};
    std::size_t decoderHeads{0};
    std::size_t encoderFfn{0};
    std::size_t decoderFfn{0};
    std::size_t vocab{0};
    std::size_t maxSourcePositions{1500}; // encoder positions (30s / 20ms)
    std::size_t maxTargetPositions{448};  // decoder positions

    std::int32_t bosTokenId{50257};
    std::int32_t eosTokenId{50257};
    std::int32_t decoderStartTokenId{50258};
    std::int32_t padTokenId{50257};

    [[nodiscard]] std::size_t encoderHeadDim() const {
        return encoderHeads ? dModel / encoderHeads : 0;
    }
    [[nodiscard]] std::size_t decoderHeadDim() const {
        return decoderHeads ? dModel / decoderHeads : 0;
    }
};

/// True if `configJson` describes a Whisper model (model_type == "whisper" or
/// an architectures entry that starts with "Whisper"). Cheap arch gate used to
/// route an ASR checkpoint to this loader.
[[nodiscard]] bool isWhisperConfig(std::string_view configJson);

/// Parse the fields WhisperConfig needs out of an HF `config.json` string.
[[nodiscard]] WhisperConfig parseWhisperConfig(std::string_view configJson);

/**
 * Per-encoder-layer device weight pointers. Whisper attention omits the K
 * bias (k_proj bias=False) — hence no `kB`. Linear weights are stored/matmul'd
 * as WhisperModel::matmulType(); biases and norms are always F32. HF Linear
 * layout is [out, in], matching ComputeMatmul's W[N,K].
 */
struct WhisperEncoderLayer {
    const float* qW{};   const float* qB{};
    const float* kW{};                          // k_proj: no bias
    const float* vW{};   const float* vB{};
    const float* oW{};   const float* oB{};     // self_attn.out_proj
    const float* attnLnW{}; const float* attnLnB{}; // self_attn_layer_norm
    const float* fc1W{}; const float* fc1B{};
    const float* fc2W{}; const float* fc2B{};
    const float* finalLnW{}; const float* finalLnB{}; // final_layer_norm
};

/**
 * Per-decoder-layer device weight pointers: masked self-attention, then
 * cross-attention over the encoder output (encoder_attn), then FFN. Both
 * attention blocks omit the K bias, as in the encoder.
 */
struct WhisperDecoderLayer {
    // Masked self-attention.
    const float* qW{};   const float* qB{};
    const float* kW{};
    const float* vW{};   const float* vB{};
    const float* oW{};   const float* oB{};
    const float* selfLnW{}; const float* selfLnB{};   // self_attn_layer_norm
    // Cross-attention over encoder states (encoder_attn).
    const float* cqW{};  const float* cqB{};
    const float* ckW{};
    const float* cvW{};  const float* cvB{};
    const float* coW{};  const float* coB{};
    const float* crossLnW{}; const float* crossLnB{}; // encoder_attn_layer_norm
    // Feed-forward.
    const float* fc1W{}; const float* fc1B{};
    const float* fc2W{}; const float* fc2B{};
    const float* finalLnW{}; const float* finalLnB{};
};

/**
 * Loads a Whisper-family ASR checkpoint (HF safetensors, single or sharded)
 * into USM via the abstract ComputeOps. The generic
 * `core::safetensors::SafetensorsModel` reader parses the container; this maps
 * HF tensor names to encoder/decoder roles, converts F16 source to F32, and
 * uploads the bytes. Backend-neutral.
 *
 * `matmulType` is the dtype the dense LINEAR weights (attention/FFN projections)
 * are stored as: F32 (exact) or BF16 (half the bytes, tensor-core GEMM at M>1).
 * Convolutions, embeddings, positional tables, biases and norms always stay F32.
 *
 * Scope: weight residency + shape validation only. The forward pass (conv stem,
 * encoder, cross-attention decoder) is a separate step.
 */
class WhisperModel {
public:
    WhisperModel() = default;

    WhisperModel(const WhisperModel&)            = delete;
    WhisperModel& operator=(const WhisperModel&) = delete;
    WhisperModel(WhisperModel&&) noexcept            = default;
    WhisperModel& operator=(WhisperModel&&) noexcept = default;

    /// `dir` is a folder with config.json + model.safetensors (or a sharded
    /// index), or a direct .safetensors path (config.json must sit next to it).
    /// Throws on a non-Whisper config, a missing tensor, an unsupported source
    /// dtype, or a shape that disagrees with the config.
    void load(std::string_view dir, compute::ComputeOps& ops,
              core::gguf::GgmlType matmulType = core::gguf::GgmlType::F32);

    [[nodiscard]] const WhisperConfig& config() const noexcept { return _config; }
    [[nodiscard]] core::gguf::GgmlType matmulType() const noexcept { return _matmulType; }

    // Convolutional stem (mel -> d_model), kernel size 3.
    [[nodiscard]] const float* conv1W() const noexcept { return _conv1W; }
    [[nodiscard]] const float* conv1B() const noexcept { return _conv1B; }
    [[nodiscard]] const float* conv2W() const noexcept { return _conv2W; }
    [[nodiscard]] const float* conv2B() const noexcept { return _conv2B; }

    // Encoder.
    [[nodiscard]] const float* encPosEmb() const noexcept { return _encPosEmb; }
    [[nodiscard]] const float* encLnW() const noexcept { return _encLnW; }
    [[nodiscard]] const float* encLnB() const noexcept { return _encLnB; }
    [[nodiscard]] const WhisperEncoderLayer& encoderLayer(std::size_t i) const {
        return _encLayers.at(i);
    }

    // Decoder.
    [[nodiscard]] const float* decTokEmb() const noexcept { return _decTokEmb; }
    [[nodiscard]] const float* decPosEmb() const noexcept { return _decPosEmb; }
    [[nodiscard]] const float* decLnW() const noexcept { return _decLnW; }
    [[nodiscard]] const float* decLnB() const noexcept { return _decLnB; }
    [[nodiscard]] const WhisperDecoderLayer& decoderLayer(std::size_t i) const {
        return _decLayers.at(i);
    }

    /// Output projection (logits). Whisper ties it to decTokEmb; a checkpoint
    /// with an explicit `proj_out.weight` returns that pointer, otherwise this
    /// is null and callers must GEMM against decTokEmb() (the tied weight).
    [[nodiscard]] const float* projOut() const noexcept { return _projOut; }
    [[nodiscard]] bool projOutTied() const noexcept { return _projOut == nullptr; }

private:
    WhisperConfig        _config{};
    core::gguf::GgmlType _matmulType{core::gguf::GgmlType::F32};

    std::vector<compute::ComputeBuffer> _owned;   // lifetime of every USM buffer
    std::vector<WhisperEncoderLayer>    _encLayers;
    std::vector<WhisperDecoderLayer>    _decLayers;

    const float* _conv1W{}; const float* _conv1B{};
    const float* _conv2W{}; const float* _conv2B{};
    const float* _encPosEmb{};
    const float* _encLnW{};  const float* _encLnB{};
    const float* _decTokEmb{};
    const float* _decPosEmb{};
    const float* _decLnW{};   const float* _decLnB{};
    const float* _projOut{};
};

} // namespace mimirmind::runtime::audio
