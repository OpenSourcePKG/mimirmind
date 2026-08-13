// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "compute/ComputeBuffer.hpp"

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace mimirmind::compute {
class ComputeOps;
} // namespace mimirmind::compute

namespace mimirmind::core::gguf {
struct GgufTensor;
} // namespace mimirmind::core::gguf

namespace mimirmind::runtime::dflash {

/// Per-layer device weight pointers for one DFlash draft transformer layer.
/// All tensors are BF16 (raw 16-bit patterns), HF Linear layout [out, in]
/// which matches ComputeMatmul's `W[N,K]`. RMSNorm weights are [head_dim] or
/// [hidden]. Pointers are opaque device handles owned by DFlashDraftModel.
struct DFlashDraftLayerWeights {
    const std::uint16_t* qProj{};   // [nQHeads*headDim, hidden]
    const std::uint16_t* kProj{};   // [nKvHeads*headDim, hidden]
    const std::uint16_t* vProj{};   // [nKvHeads*headDim, hidden]
    const std::uint16_t* oProj{};   // [hidden, nQHeads*headDim]
    const std::uint16_t* qNorm{};   // [headDim]
    const std::uint16_t* kNorm{};   // [headDim]
    const std::uint16_t* inputLn{};        // [hidden]
    const std::uint16_t* postAttnLn{};     // [hidden]
    const std::uint16_t* gateProj{};       // [inter, hidden]
    const std::uint16_t* upProj{};         // [inter, hidden]
    const std::uint16_t* downProj{};       // [hidden, inter]
};

/// Architecture dims of a DFlash drafter, derived from the checkpoint's
/// tensor shapes at load (no config.json needed for the upload).
struct DFlashDraftConfig {
    std::size_t numLayers{};   // 6
    std::size_t hidden{};      // 2048 (== target text hidden_size)
    std::size_t headDim{};     // 128
    std::size_t nQHeads{};     // 32
    std::size_t nKvHeads{};    // 8
    std::size_t inter{};       // 6144
    std::size_t taps{};        // 8  (fc in-features / hidden)
};

/**
 * Loads a Qwen3.6-A3B DFlash block-diffusion drafter (z-lab checkpoint) from
 * HF BF16 safetensors straight onto the device. Mirrors EncoderModel: the
 * generic SafetensorsModel reader parses the file; this maps HF tensor names
 * to draft roles and uploads the BF16 bytes verbatim (no F32 round-trip — the
 * checkpoint is already BF16). Backend-neutral: uploads go through the
 * abstract ComputeOps.
 *
 * The drafter has NO embed_tokens / lm_head of its own — those are borrowed
 * from the loaded target via borrowTarget() (Phase 3). Load and borrow are
 * independent: load() only touches the drafter's own 69 tensors.
 *
 * Throws on a missing / wrong-dtype / wrong-shape tensor.
 */
class DFlashDraftModel {
public:
    DFlashDraftModel() = default;

    DFlashDraftModel(const DFlashDraftModel&)            = delete;
    DFlashDraftModel& operator=(const DFlashDraftModel&) = delete;
    DFlashDraftModel(DFlashDraftModel&&) noexcept            = default;
    DFlashDraftModel& operator=(DFlashDraftModel&&) noexcept = default;

    /// `dir` is a folder with the drafter's config.json + safetensors (single
    /// file or sharded index), or a direct .safetensors path. Uploads all 69
    /// BF16 tensors to device and records their handles. Derives the arch dims
    /// from the tensor shapes and validates every tensor's shape/dtype.
    void load(std::string_view dir, compute::ComputeOps& ops);

    /// Borrow the target's `token_embd.weight` (noise embedding) and
    /// `output.weight` (draft-logit lm_head) — the drafter reuses them rather
    /// than shipping its own. Both are optional here (Phase-3 forward needs
    /// them; the upload path does not). Validates hidden-size consistency when
    /// a tensor is provided. Pass nullptr to leave a slot unborrowed.
    void borrowTarget(const core::gguf::GgufTensor* embedTokens,
                      const core::gguf::GgufTensor* lmHead);

    [[nodiscard]] const DFlashDraftConfig& config() const noexcept { return _config; }

    [[nodiscard]] const DFlashDraftLayerWeights& layer(std::size_t i) const {
        return _layers.at(i);
    }
    [[nodiscard]] std::size_t layerCount() const noexcept { return _layers.size(); }

    // DFlash-specific top-level tensors.
    [[nodiscard]] const std::uint16_t* fc() const noexcept { return _fc; }                 // [hidden, taps*hidden]
    [[nodiscard]] const std::uint16_t* hiddenNorm() const noexcept { return _hiddenNorm; } // [hidden], RMSNorm after fc
    [[nodiscard]] const std::uint16_t* norm() const noexcept { return _norm; }             // [hidden], final RMSNorm

    // Borrowed target weights (nullptr until borrowTarget()).
    [[nodiscard]] const core::gguf::GgufTensor* embedTokens() const noexcept { return _embed; }
    [[nodiscard]] const core::gguf::GgufTensor* lmHead() const noexcept { return _lmHead; }

    /// Total device bytes uploaded by load() (for diagnostics / tests).
    [[nodiscard]] std::size_t uploadedBytes() const noexcept { return _uploadedBytes; }

private:
    DFlashDraftConfig _config{};

    std::vector<compute::ComputeBuffer> _owned;   // lifetime of every device buffer
    std::vector<DFlashDraftLayerWeights> _layers;

    const std::uint16_t* _fc{};
    const std::uint16_t* _hiddenNorm{};
    const std::uint16_t* _norm{};

    const core::gguf::GgufTensor* _embed{};
    const core::gguf::GgufTensor* _lmHead{};

    std::size_t _uploadedBytes{};
};

} // namespace mimirmind::runtime::dflash
