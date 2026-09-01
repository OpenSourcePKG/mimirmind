// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace mimirmind::core::backend { class ComputeContext; }
namespace mimirmind::model         { class LlmConfig; }

namespace mimirmind::runtime::serving {

/**
 * Snapshot of the HW-capacity probe run at InferenceEngine startup.
 * Backend-neutral; the actual bandwidth / VRAM accessors come from
 * `ComputeContext::bandwidthGBps()` etc. once those land in
 * M-Startup.CapacityProbe Sub-Step 2.
 *
 * Populated by `BatchCapacityProbe::estimate(...)`. Consumers read
 * `sustainableBatch` + `servingClassRecommended` to decide whether
 * PagedAttention / Continuous Batching (M-Cuda.Batch) should be
 * enabled at this instance.
 */
struct BatchCapacityEstimate {
    std::size_t bandwidthGBps{0};              // from ComputeBackend / BackendPool
    std::size_t freeVramGB{0};                 // cudaMemGetInfo / zeMemAllocProperties / hipMemGetInfo
    std::size_t weightBytes{0};                // from LlmConfig + GgufReader
    std::size_t kvBytesPerToken{0};            // from LlmConfig (nHeads * kvDim * dtype)
    std::size_t maxContext{0};                 // from LlmConfig
    std::size_t sustainableBatch{1};           // rounded to {1,2,4,8,16,32}
    bool        servingClassRecommended{false};// sustainableBatch >= kDefaultMinServingBatch
    std::string reasoning;                     // one-line human-readable
};

/**
 * Startup-time HW-roofline probe. Skeleton in this commit; real
 * accessors + config gate + InferenceEngine wiring come with later
 * Sub-Steps of the M-Startup.CapacityProbe milestone (Bragi phase).
 *
 * Server-side decision only — no user-per-request toggle
 * (`feedback_no_user_toggles`). Config is `serving.enable_batching`
 * ("auto" | true | false) which drives whether Serving-Class-Features
 * light up when the probe agrees.
 */
class BatchCapacityProbe {
public:
    /// Default recommend-threshold — batching-mode is enabled when
    /// `sustainableBatch >= this value`. Config-overridable via
    /// `serving.min_batch_for_enable` in a later Sub-Step.
    static constexpr std::size_t kDefaultMinServingBatch = 8;

    /// Hard ceiling on the auto-derived batch. The VRAM path can compute
    /// very large batches on a big unified pool; cap so a single probe
    /// never commits the scheduler to an unvalidated extreme. Operators
    /// raise past this deliberately via `MIMIRMIND_SERVING_MAXBATCH`.
    static constexpr std::size_t kMaxServingBatch = 128;

    /// Fraction of free device memory the VRAM path budgets for the KV
    /// pool. The remainder is left for activations, KV growth beyond the
    /// probe's `maxContext` assumption, and co-resident models that may
    /// load after this probe runs. Percent (of 100) to stay integer.
    static constexpr std::size_t kMemoryReservePct = 50;

    /**
     * Rounds a raw batch estimate DOWN to the nearest of
     * {1, 2, 4, 8, 16, 32, 64, 128}. Values >= 128 clamp to 128.
     * Scheduler decisions want predictable step sizes; a raw estimate
     * of 27 becomes 16 (not 32) so we don't over-commit VRAM based on
     * borderline probes.
     *
     * Exposed static for direct unit-testing without needing a
     * ComputeContext.
     */
    [[nodiscard]] static std::size_t roundToSchedulerStep(std::size_t raw) noexcept;

    /**
     * Conservative fallback estimate — used when any probe input is 0
     * (unknown-HW / model-not-loaded / probe-disabled). Returns
     * `sustainableBatch=1, servingClassRecommended=false, reasoning=
     * "probe not yet implemented — assuming single-session"`.
     */
    [[nodiscard]] static BatchCapacityEstimate estimateConservativeFallback() noexcept;

    /**
     * Real HW-driven estimate. All inputs are scalars so this stays in
     * `mimirmind_core_common` without a link-dep on any backend. The
     * caller (`InferenceEngine::finalizeLoad`) fetches them from:
     *   bandwidthGBps    — `ComputeContext::bandwidthGBps()` (Sub-Step 2)
     *   weightBytes      — `GgufReader::totalTensorBytes()`
     *   kvBytesPerToken  — `LlmConfig::kvBytesPerToken(...)` (Sub-Step 3)
     *   maxContext       — `LlmConfig::contextLength` or config override
     *
     * VRAM-aware (v2): when `freeMemoryBytes > 0` the batch is sized
     * from real headroom — how many per-sequence KV pools of
     * `kvBytesPerToken * maxContext` fit in a conservative reserve of
     * the free pool (`kMemoryReserveFrac`), so it accounts for
     * activations, KV growth, and co-resident-model slack. `bandwidthGBps
     * < 80` still forces batch 1 (single-user iGPU / CPU tier, e.g.
     * Xe-LPG) regardless of memory. Result is bounded to
     * `roundToSchedulerStep` and `kMaxServingBatch`.
     *
     * CAVEAT: on a co-resident multi-model host `freeMemoryBytes`
     * depends on load order (not-yet-loaded models still look free), so
     * the estimate is an upper bound. Operators pin an exact cap with
     * `MIMIRMIND_SERVING_MAXBATCH` for such serves; that env override
     * wins over this estimate at the scheduler.
     *
     * Fallback (v1): when `freeMemoryBytes == 0` (non-CUDA backend or a
     * failed query) the old bandwidth-tier proxy is used:
     *   <  80 GB/s → batch 1  ·  < 200 → 4  ·  < 400 → 16  ·  >= 400 → 32
     *
     * All inputs are scalars so this stays in `mimirmind_core_common`
     * without a link-dep on any backend. Any other zero input falls
     * back to 1. `reasoning` logs the inputs + which path was taken.
     *
     * Never throws.
     */
    [[nodiscard]] static BatchCapacityEstimate estimate(
        std::size_t bandwidthGBps,
        std::size_t weightBytes,
        std::size_t kvBytesPerToken,
        std::size_t maxContext,
        std::size_t freeMemoryBytes = 0) noexcept;
};

} // namespace mimirmind::runtime::serving
