// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "runtime/InferenceEngine.hpp"   // InferenceEngine + ServingSlotStep
#include "runtime/TokenLogprobs.hpp"     // 8.19.14 optional logprob capture

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace mimirmind::runtime { class KvCache; }   // nextn (MTP) KV cache
namespace mimirmind::model { class ToolCallConstraint; }  // 8.19.13.2

namespace mimirmind::runtime::engine {

struct ServingState;   // persistent continuous-batch substrate (defined in .cpp)
struct L0ServingState; // non-paged L0/Xe-LPG slab substrate (defined in .cpp)

/**
 * Batched / serving-class decode substrate, extracted from InferenceEngine.
 * Bundles the one-shot batched-generation harnesses (generateBatch /
 * generateServingParity) and the persistent continuous-batching state
 * (ensureServingState / stepServing) that ContinuousBatcher drives. Owns the
 * persistent ServingState; everything else — backend, weights, ops, matmul —
 * is reached through the engine, of which this is a friend collaborator.
 *
 * Constructed lazily by InferenceEngine on first serving call. CUDA +
 * qwen35moe only (the methods throw otherwise).
 */
class ServingSession {
public:
    explicit ServingSession(InferenceEngine& engine);  // out-of-line (ServingState
    ~ServingSession();                                 // complete only in the .cpp)

    ServingSession(const ServingSession&)            = delete;
    ServingSession& operator=(const ServingSession&) = delete;

    /// See InferenceEngine::generateBatch.
    [[nodiscard]] std::vector<std::vector<std::int32_t>>
    generateBatch(const std::vector<std::vector<std::int32_t>>& prompts,
                  std::size_t maxNew, std::int32_t eosId);

    /// See InferenceEngine::generateServingParity.
    [[nodiscard]] std::vector<std::vector<std::int32_t>>
    generateServingParity(std::span<const std::int32_t> promptIds,
                          std::size_t nSeq, std::size_t maxNew);

    /// See InferenceEngine::ensureServingState.
    void ensureServingState(std::size_t maxBatch, std::size_t maxContext);

    /// See InferenceEngine::stepServing.
    void stepServing(std::span<const InferenceEngine::ServingSlotStep> steps,
                     std::span<std::int32_t>                           outTokens,
                     std::vector<runtime::TokenLogprobs>* outLp = nullptr);

    /// 8.19.5: set slot's sampling params for all subsequent tokens (first
    /// prefill token + every decode step) until the next call. seed==0 draws a
    /// fresh non-deterministic seed. temperature<=0 with neutral penalties ==
    /// greedy argmax (bit-identical to the pre-sampling behaviour). Call at
    /// admission. `promptTail` seeds the penalty window with the end of the
    /// prompt (M7f: an immediate first-token echo of the prompt's last phrase
    /// is discouraged too); each sampled token is appended afterwards, so the
    /// repetition/frequency/presence penalties see prompt tail + generated.
    void setSlotSampling(std::size_t slot, const compute::SamplingParams& sp,
                         std::span<const std::int32_t> promptTail = {});

    /// 8.19.13.2 — bind a per-slot grammar constraint (or nullptr to clear).
    /// When set and active, the per-slot sampler masks each step's logits so
    /// the model can only spell an offered tool name / schema key inside a
    /// tool-call. Reset at bind time (start of the slot's response).
    void setSlotToolConstraint(std::size_t slot,
                               std::shared_ptr<model::ToolCallConstraint> c);

    /// See InferenceEngine::prefillSlot.
    [[nodiscard]] std::int32_t
    prefillSlot(std::size_t slot, std::span<const std::int32_t> tokens,
                std::size_t startPos, bool produceToken,
                runtime::TokenLogprobs* outLp = nullptr);

    /// 5.21-III MULTI-SLOT: prefill N contiguous slots [firstSlot, firstSlot+N)
    /// in ONE ragged batched forward (sbMixed). chunks[s]/startPositions[s]
    /// describe slot firstSlot+s; the greedy next-token per slot (or -1 when
    /// produceToken is false) is written into outFirstTok[s]. Requires the
    /// slots to be a contiguous run so recurrence state / block-table slices
    /// address as base + firstSlot*stride + seq*stride.
    void runVarlenPrefill(std::size_t firstSlot,
                          std::span<const std::span<const std::int32_t>> chunks,
                          std::span<const std::size_t>                   startPositions,
                          bool                                           produceToken,
                          std::span<std::int32_t>                        outFirstTok,
                          std::vector<runtime::TokenLogprobs>* outLp = nullptr);

    /// See InferenceEngine::servingPrefillChunk.
    [[nodiscard]] std::size_t prefillChunkSize() const noexcept;

    /// See InferenceEngine::servingPrefillMaxRows.
    [[nodiscard]] std::size_t prefillMaxRows() const noexcept;

    /// See InferenceEngine::stepServingVerify.
    [[nodiscard]] std::vector<std::vector<float>>
    stepServingVerify(std::span<const InferenceEngine::VerifySlot> slots,
                      std::span<const std::int32_t>                tokensTimeMajor,
                      std::size_t                                  depth);

    /// See InferenceEngine::mtpDraftParity.
    [[nodiscard]] InferenceEngine::MtpDraftParityResult
    mtpDraftParity(std::span<const std::int32_t> prompt,
                   std::size_t nSeq, std::size_t depth);

    /// See InferenceEngine::generateBatchMtp.
    [[nodiscard]] std::vector<std::vector<std::int32_t>>
    generateBatchMtp(std::span<const std::int32_t> prompt,
                     std::size_t nSeq, std::size_t maxNew,
                     std::size_t depth, std::int32_t eosId);

    /// See InferenceEngine::generateBatchMtpMulti.
    [[nodiscard]] std::vector<std::vector<std::int32_t>>
    generateBatchMtpMulti(const std::vector<std::vector<std::int32_t>>& prompts,
                          std::size_t maxNew, std::size_t depth,
                          std::int32_t eosId);

    /// DFlash serving-batched spec decode (5.9.1): one shared prompt over nSeq
    /// slots, block-diffusion draft per slot (DFlashDecoder), batched verify over
    /// M=nSeq*(depth+1) (the expert-read amortization) + per-slot accept via the
    /// per-timestep SSM export. `drafterDir` is the DFlash checkpoint dir.
    [[nodiscard]] std::vector<std::vector<std::int32_t>>
    generateBatchDflash(std::span<const std::int32_t> prompt,
                        std::size_t nSeq, std::size_t maxNew, std::size_t depth,
                        std::int32_t eosId, std::string_view drafterDir,
                        std::size_t* draftedOut = nullptr,
                        std::size_t* acceptedOut = nullptr);

    [[nodiscard]] std::size_t maxBatch() const noexcept;
    [[nodiscard]] std::size_t maxContext() const noexcept;

    /// 5.28.1.3 — GDN warm-slot prefix reuse via a bounded per-slot ring of
    /// INTERIOR SSM+conv checkpoints. `captureSlotSsmCkpt` copies the slot's live
    /// sub-slab OUT tagged with token position `pos` (called at each prefill-chunk
    /// boundary). `slotCkptBestPos` returns the largest checkpoint pos <= `lcp`
    /// (0 = none usable). `restoreSlotSsmCkptAtPos` copies the checkpoint at `pos`
    /// back into the live sub-slab (no-op if absent — caller must then NOT reuse).
    /// `pruneSlotSsmCkpts` drops checkpoints beyond `keepMaxPos` (a reuse's shared
    /// prefix); `clearSlotSsmCkpts` frees the ring (cold eviction). All no-op on
    /// the L0 slab path. Device→device, queued on the compute stream.
    void        captureSlotSsmCkpt(std::size_t slot, std::size_t pos);
    [[nodiscard]] std::size_t slotCkptBestPos(std::size_t slot,
                                              std::size_t lcp) const;
    void        restoreSlotSsmCkptAtPos(std::size_t slot, std::size_t pos);
    void        pruneSlotSsmCkpts(std::size_t slot, std::size_t keepMaxPos);
    void        clearSlotSsmCkpts(std::size_t slot);

    /// Resident paged-KV pool footprint for the memory-telemetry route (8.16).
    /// `active` is false until the serving state has been allocated (no pool
    /// yet). `residentBytes` = numLayers x 2(K+V) x numBlocks x blockSize x
    /// numKvHeads x headDim x elemBytes — the full device slab (the pool is
    /// allocated up-front, not grown per request).
    struct KvStats {
        bool        active{false};
        std::size_t residentBytes{0};
        std::size_t numLayers{0};
        std::size_t numBlocks{0};
        std::size_t blockSize{0};
        std::size_t elemBytes{0};
    };
    [[nodiscard]] KvStats kvStats() const noexcept;

private:
    /// Lazily (re)allocate the Increment-E1 verify scratch for up to
    /// `maxBatch` slots × `depth + 1` verify tokens. Grows monotonically.
    void ensureVerifyCapacity(std::size_t depth);

    /// Shared verify trunk forward: validates `slots`/`tokensTimeMajor`, runs
    /// the N*(depth+1) full-attention + GatedDeltaNet-verify trunk, and leaves
    /// per-position logits in `st.vLogitsB` on the device (NOT flushed / read
    /// back). Returns M = N*(depth+1) (0 if N==0). The two entry points below
    /// differ only in how they harvest those logits.
    [[nodiscard]] std::size_t
    verifyForward(std::span<const InferenceEngine::VerifySlot> slots,
                  std::span<const std::int32_t>                tokensTimeMajor,
                  std::size_t                                  depth);

    /// MTP verify harvest for the perf path: device argmax over each of the M
    /// rows, reading back only the M token ids (a few bytes) instead of the
    /// full M*vocab logits (tens of MB). Row r = j*N+s is slot s's j-th verify
    /// position. Byte-identical argmax to `stepServingVerify` + host `argmax`.
    [[nodiscard]] std::vector<std::int32_t>
    stepServingVerifyIds(std::span<const InferenceEngine::VerifySlot> slots,
                         std::span<const std::int32_t>                tokensTimeMajor,
                         std::size_t                                  depth);

    /// Increment E2 — lazily allocate the per-slot nextn (MTP) KV caches +
    /// shared draft scratch. Requires a loaded nextn head. Idempotent.
    void ensureMtpServingState();

    /// Increment E2 — draft `K` tokens through `kv` (a 1-layer nextn KV
    /// cache) starting from trunk hidden `hidden0` (device [d]) and token
    /// `prevTok`, committing each step. Appends the drafted ids to `out`.
    /// `runMtpDraftStep` per step (the draft is cheap — blk.<blockCount>).
    void draftKInto(KvCache& kv, const float* hidden0, std::int32_t prevTok,
                    std::size_t K, std::vector<std::int32_t>& out);

    /// Increment E3 — restore slot `slot`'s GatedDeltaNet recurrent + conv
    /// state from verify snapshot `snapIdx` (== accepted-token count), a
    /// per-layer per-slot slice copy. Avoids the single-session re-forward
    /// on a partial accept.
    // MV-d: commit slot's recurrent state to accept position `a` from the
    // fused-kernel export (packed with slot stride `N`), and its conv state
    // from convSnap[a]. Called for EVERY accept (the fused kernel does not
    // advance the live state), unlike the old snapshot path.
    void restoreSlotSsm(std::size_t slot, std::size_t a, std::size_t N);

    /// Increment E5b — seed the paged nextn KV (pool layer mtpPoolLayer) for
    /// `nSeq` slots by replaying an IDENTICAL prompt: broadcast the trunk
    /// hiddens `promptHiddens` ([P,d], device) and step the batched nextn
    /// block P-1 times (no head). Sets nextnLen[s] = P-1.
    void mtpSeedBatched(std::size_t nSeq, const float* promptHiddens,
                        std::span<const std::int32_t> prompt);

    /// Increment E5b — draft `K` tokens for ALL `nSeq` slots in one batched
    /// nextn forward per step (round-start hidden in `mtpHid`, per-slot
    /// `token0`), advancing each slot's nextnLen. One flush+argmax readback
    /// per step (K total) instead of the N×K of the per-slot draftKInto.
    void draftBatchRound(std::size_t nSeq, std::size_t K,
                         const std::vector<std::int32_t>&        token0,
                         std::vector<std::vector<std::int32_t>>& drafts);

    InferenceEngine&                _e;
    std::unique_ptr<ServingState>   _state;   // qwen35moe paged path
    std::unique_ptr<L0ServingState> _l0;      // L0/Xe-LPG non-paged slab path
};

} // namespace mimirmind::runtime::engine
