// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "compute/ComputeOps.hpp"           // compute::ComputeBuffer
#include "runtime/dflash/DFlashDraftModel.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace mimirmind::runtime {
class InferenceEngine;
} // namespace mimirmind::runtime

namespace mimirmind::runtime::dflash {
class DFlashDraftRunner;
} // namespace mimirmind::runtime::dflash

namespace mimirmind::runtime::engine {

/**
 * DFlash block-diffusion speculative decoder (Phase 3.4, C1 single-stream).
 *
 * Mirrors engine::MtpDecoder: a friend collaborator of InferenceEngine that
 * borrows the target's trunk (forwardVerify / commitVerified), embed_tokens
 * and lm_head, and drives a draft->verify->accept loop. The draft comes from
 * the external DFlash drafter (DFlashDraftModel + DFlashDraftRunner) instead of
 * the in-model nextn head.
 *
 * DFlash-specific mechanics vs MTP:
 *   - The draft conditions on the TARGET hidden states at the 8 tap layers
 *     {1,6,11,16,22,27,32,37}, captured via Qwen35MoeBackend::configureHiddenTap
 *     (Phase 2). Each forwardVerify writes the tap sinks position-major at the
 *     absolute committed positions; feedContext() concatenates the 8 sinks per
 *     position into `_ctxHidden` (the [ctxLen, taps*hidden] the runner wants).
 *   - ONE draft forward proposes the whole K-token block in parallel: the noise
 *     block [token0, mask x K] -> borrowed embed -> DFlashDraftRunner::draftForward
 *     -> borrowed lm_head -> argmax at positions 1..K.
 *   - Verify is standard causal linear-chain longest-prefix (identical to MTP);
 *     the GatedDeltaNet recurrent state is snapshotted/restored on partial
 *     accept exactly as MtpDecoder does.
 *
 * Greedy only (temperature 0). CUDA + qwen35moe target; the engine gates on
 * dflashAvailable(). Constructed lazily by InferenceEngine::generateDflash.
 * Not thread-safe.
 */
class DFlashDecoder {
public:
    // Both defined out-of-line in the .cpp: the constructor's member-cleanup
    // and the destructor need the complete DFlashDraftRunner (unique_ptr member
    // is only forward-declared in this header).
    explicit DFlashDecoder(InferenceEngine& engine);
    ~DFlashDecoder();

    DFlashDecoder(const DFlashDecoder&)            = delete;
    DFlashDecoder& operator=(const DFlashDecoder&) = delete;

    /// See InferenceEngine::generateDflash. `drafterDir` holds the DFlash
    /// checkpoint (loaded + tap-sink/scratch allocated on the first call, then
    /// cached). `draftN` is the number of tokens proposed per round (block size
    /// bs = draftN + 1); 3-4 is the sweet spot (SGLang oracle accept-len ~3.3).
    /// Output is bit-identical to greedy generate() at temperature 0.
    [[nodiscard]] std::vector<std::int32_t>
    generate(std::span<const std::int32_t> promptIds,
             std::size_t                   maxNew,
             std::size_t                   draftN,
             std::int32_t                  eosId,
             std::string_view              drafterDir,
             std::size_t*                  draftedOut,
             std::size_t*                  acceptedOut);

    // ---- Serving-batched reuse (5.9.1) ----------------------------------
    /// Load the drafter + borrow the target embed/lm_head (idempotent). Public
    /// wrapper so ServingSession::generateBatchDflash can prepare the drafter.
    void ensureDflashLoaded(std::string_view drafterDir) { ensureLoaded(drafterDir); }

    /// One-block draft for a single slot: given that slot's context hidden
    /// `ctxHidden` [ctxLen, taps*hidden] (device) and anchor `token0`, propose K
    /// tokens (embed [token0, mask x K] -> draftForward -> borrowed lm_head +
    /// device per-row argmax). Reuses the single-block scratch (call per slot,
    /// sequentially). Requires ensureDflashLoaded() first.
    void draftOneBlock(const float* ctxHidden, std::size_t ctxLen,
                       std::int32_t token0, std::size_t K,
                       std::vector<std::int32_t>& drafts);

    /// Serving prefill: run the trunk over `prompt` with the 8-tap live, build
    /// the shared prompt context in the internal accumulator rows [0, promptLen)
    /// and return the greedy anchor token0. The caller broadcasts
    /// ctxHidden()[0 .. promptLen*ctxRowStride()) into its per-slot contexts.
    /// Uses the single-session KV (separate from the serving paged KV, like
    /// generateBatchMtp's token0 prefill). Requires ensureDflashLoaded() first.
    [[nodiscard]] std::int32_t buildPromptContext(
        std::span<const std::int32_t> prompt, std::size_t& promptLen);

    /// Device base of the internal [maxPos, taps*hidden] context accumulator.
    [[nodiscard]] float* ctxHidden() noexcept;
    [[nodiscard]] std::size_t ctxRowStride() const noexcept { return _taps * _d; }

    [[nodiscard]] std::size_t hiddenDim() const noexcept { return _d; }
    [[nodiscard]] std::size_t tapCount() const noexcept { return _taps; }
    [[nodiscard]] std::span<const std::size_t> tapLayers() const noexcept {
        return std::span<const std::size_t>{kTapLayers, _taps};
    }

private:
    /// Load the drafter, borrow the target embed/lm_head, build the runner, and
    /// allocate the tap sinks + context/scratch buffers. Idempotent.
    void ensureLoaded(std::string_view drafterDir);

    /// Concatenate the 8 tap sinks for absolute positions [p0, p1) into
    /// `_ctxHidden` (row-major [pos, taps*hidden], taps in tap-layer order).
    void feedContext(std::size_t p0, std::size_t p1);

    InferenceEngine& _e;

    bool                                     _loaded{false};
    dflash::DFlashDraftModel                 _model;
    std::unique_ptr<dflash::DFlashDraftRunner> _runner;

    // Drafter config.json mask token (block filler). Read at load; the drafter
    // ships mask_token_id=248077 for the Qwen3.6 checkpoint.
    std::int32_t _maskTok{248077};

    std::size_t _d{0};        // hidden (2048)
    std::size_t _taps{0};     // 8
    std::size_t _vocabLm{0};  // lm_head rows
    std::size_t _vocabEmb{0}; // embed rows

    // 8 tap sinks (position-major [maxPos, hidden]) + their device pointers,
    // handed to the backend via configureHiddenTap.
    std::vector<compute::ComputeBuffer> _tapSink;
    std::vector<float*>                 _tapPtr;

    compute::ComputeBuffer _ctxHidden;  // [maxPos, taps*hidden] concat accumulator
    compute::ComputeBuffer _noise;      // [maxBlock, hidden] embedded block
    compute::ComputeBuffer _draftOut;   // [maxBlock, hidden] draft-forward output
    // Batched draft readout (Hebel 1): ONE lm_head matmul over the K draft
    // positions -> [maxBlock-1, vocab], then on-device per-row argmax, so only
    // K token ids are read back per round instead of K full-vocab D2H copies.
    compute::ComputeBuffer    _draftLogits;    // [maxBlock-1, vocabLm]
    compute::ComputeBuffer    _draftArgmaxDev; // [maxBlock-1] device argmax ids
    std::vector<std::int32_t> _draftArgmaxHost;

    // GDN ReplaySSM (MIMIRMIND_DFLASH_FOLD): replace the partial-accept
    // re-forward with a fold of the accepted prefix. Per-recurrent-block capture
    // rings of the verify window's recurrence inputs + conv input, filled by
    // Qwen35MoeBackend during forwardVerify.
    bool                                _foldMode{false};
    std::vector<std::size_t>            _recurBlocks;   // recurrent block indices
    std::size_t _hV{0}, _sState{0}, _convDim{0}, _convStateElems{0};
    std::vector<compute::ComputeBuffer> _capK, _capV, _capG, _capB, _capConv;
    compute::ComputeBuffer _ssmBak;     // GatedDeltaNet state snapshot (verify rollback)
    compute::ComputeBuffer _convBak;    // rolling conv-tail snapshot

    // Tap layers (block indices) whose residual-stream output feeds the fusion,
    // in concat order — must match the reference [hidden_states[lid+1]] list.
    static constexpr std::size_t kTapLayers[8] = {1, 6, 11, 16, 22, 27, 32, 37};
    // Upper bound on the per-round block size (bs = draftN + 1) for scratch.
    static constexpr std::size_t kMaxBlock = 33;
};

} // namespace mimirmind::runtime::engine
