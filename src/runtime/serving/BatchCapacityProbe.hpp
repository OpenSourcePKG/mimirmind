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

    /**
     * Rounds a raw batch estimate DOWN to the nearest of
     * {1, 2, 4, 8, 16, 32}. Values >= 32 clamp to 32. Scheduler
     * decisions want predictable step sizes; a raw estimate of 27
     * becomes 16 (not 32) so we don't over-commit VRAM based on
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
     * v1 heuristic: bandwidth-tier proxy for total device memory (a
     * real per-device VRAM probe is a follow-up milestone). Rough tiers:
     *   <  80 GB/s → batch 1  (low-end iGPU / CPU)
     *   <  200     → batch 4  (mid iGPU — HIP Phoenix class)
     *   <  400     → batch 16 (high-end integrated — DGX Spark class)
     *   >= 400     → batch 32 (discrete dGPU class)
     * Bounded to `roundToSchedulerStep`. Any zero input falls back to 1.
     * `reasoning` field logs all inputs so operators can debug the
     * decision at `/v1/system/info`.
     *
     * Never throws.
     */
    [[nodiscard]] static BatchCapacityEstimate estimate(
        std::size_t bandwidthGBps,
        std::size_t weightBytes,
        std::size_t kvBytesPerToken,
        std::size_t maxContext) noexcept;
};

} // namespace mimirmind::runtime::serving
