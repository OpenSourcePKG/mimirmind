// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "runtime/InferenceEngine.hpp"   // InferenceEngine + ServingSlotStep

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace mimirmind::runtime { class KvCache; }   // nextn (MTP) KV cache

namespace mimirmind::runtime::engine {

struct ServingState;   // persistent continuous-batch substrate (defined in .cpp)

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
                     std::span<std::int32_t>                           outTokens);

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

    [[nodiscard]] std::size_t maxBatch() const noexcept;
    [[nodiscard]] std::size_t maxContext() const noexcept;

private:
    /// Lazily (re)allocate the Increment-E1 verify scratch for up to
    /// `maxBatch` slots × `depth + 1` verify tokens. Grows monotonically.
    void ensureVerifyCapacity(std::size_t depth);

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
    void restoreSlotSsm(std::size_t slot, std::size_t snapIdx);

    InferenceEngine&              _e;
    std::unique_ptr<ServingState> _state;
};

} // namespace mimirmind::runtime::engine
