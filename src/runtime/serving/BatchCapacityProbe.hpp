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
     * Conservative fallback estimate. Skeleton returns
     * `sustainableBatch=1, servingClassRecommended=false, reasoning=
     * "probe not yet implemented — assuming single-session"`.
     * Real HW-driven `estimate(ComputeContext&, LlmConfig&, ...)`
     * lands in Sub-Step 5 (InferenceEngine wiring).
     */
    [[nodiscard]] static BatchCapacityEstimate estimateConservativeFallback() noexcept;
};

} // namespace mimirmind::runtime::serving
