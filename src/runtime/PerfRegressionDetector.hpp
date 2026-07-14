// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace mimirmind::runtime {

/**
 * In-process regression detector. Sits next to the M9.6.3 per-token
 * NDJSON sink and does what an external tail-and-alert script would do,
 * but without a second process and without an extra file to ship.
 *
 * How it works
 *   - InferenceEngine::generate() feeds one Sample per decode token via
 *     onDecodeToken(). The first kWarmupTokens are ignored (cold-start,
 *     cache miss, cap ramp) and the rest fill a ring buffer.
 *   - onRunComplete() at the end of the decode loop computes p50 over
 *     the ring buffer, compares it against the rolling baseline (median
 *     of prior runs in the last kBaselineDays), and — if the run had
 *     enough samples AND enough baseline history — emits an alert when
 *     the p50 exceeds kAlertThreshold × baseline.
 *   - The baseline is a plain JSON file at the path passed to the ctor;
 *     samples older than kBaselineDays are dropped at every write. The
 *     "current run" is always promoted into the baseline (a real
 *     regression becomes the new normal after ~kMinBaselineN more runs).
 *
 * Thread safety
 *   - onDecodeToken()/onRunComplete() run on the decode thread only.
 *   - lastAlert()/currentP50Ms()/baselineP50Ms() are safe to call from
 *     the HTTP worker thread (ApiServer::handleSystemStatus).
 *   - All cross-thread state is protected by an internal mutex.
 *
 * The detector never throws and never allocates on the hot path
 * (onDecodeToken is noexcept + no-alloc). If persisting the baseline
 * fails, it logs and moves on — the detector must not break inference.
 */
class PerfRegressionDetector {
public:
    /// One per decoded token. Fields mirror the NDJSON sink so the same
    /// numbers reach both places.
    struct Sample {
        double        wall_ms;
        std::uint32_t cap_mhz;
        double        pkg_c;
    };

    struct Alert {
        double        current_p50_ms;
        double        baseline_p50_ms;
        double        delta_ratio;
        std::uint64_t internal_version;
        std::int64_t  detected_unix;
    };

    /// Reads the baseline from `baselineJsonPath` (empty/missing/malformed
    /// => fresh start). Bumps the persisted internal_version counter so
    /// this process runs one higher than the last one — that number is
    /// what gets stamped onto every baseline sample + alert. First-ever
    /// start writes version 1.
    explicit PerfRegressionDetector(std::string_view baselineJsonPath);
    ~PerfRegressionDetector();

    PerfRegressionDetector(const PerfRegressionDetector&)            = delete;
    PerfRegressionDetector& operator=(const PerfRegressionDetector&) = delete;
    PerfRegressionDetector(PerfRegressionDetector&&)                 = delete;
    PerfRegressionDetector& operator=(PerfRegressionDetector&&)      = delete;

    /// Hot path — called once per decode token. No I/O, no allocation.
    void onDecodeToken(const Sample& s) noexcept;

    /// End-of-run: compute current p50, compare against baseline, maybe
    /// alert, promote the run into the baseline, persist. Never throws.
    void onRunComplete(std::size_t emittedTokens) noexcept;

    /// Cross-thread accessors — used by ApiServer::handleSystemStatus.
    /// currentP50Ms()/baselineP50Ms() return -1.0 when not yet valid.
    [[nodiscard]] std::optional<Alert> lastAlert() const noexcept;
    [[nodiscard]] double               currentP50Ms() const noexcept;
    [[nodiscard]] double               baselineP50Ms() const noexcept;
    [[nodiscard]] std::size_t          baselineSampleCount() const noexcept;
    [[nodiscard]] std::uint64_t        internalVersion() const noexcept {
        return _internalVersion;
    }

    // --- Tuning constants (not env-configurable) ------------------------
    /// Tokens ignored at the start of a run (cache miss, cap ramp).
    static constexpr std::size_t kWarmupTokens   = 20;
    /// Ring-buffer size for the current-run p50 estimate.
    static constexpr std::size_t kRollingWindow  = 500;
    /// Minimum post-warmup tokens for a run to count.
    static constexpr std::size_t kMinRunSamples  = 100;
    /// Minimum prior runs to have a valid baseline.
    static constexpr std::size_t kMinBaselineN   = 3;
    /// Alert if current_p50 > threshold × baseline_p50.
    static constexpr double      kAlertThreshold = 1.15;
    /// Rolling baseline window in days.
    static constexpr int         kBaselineDays   = 7;

private:
    struct RunSample {
        std::int64_t  unix_sec;
        double        p50_ms;
        std::size_t   n;
        std::uint64_t internal_version;
    };

    void loadBaseline() noexcept;
    void persistBaseline() noexcept;
    void trimBaselineToWindowLocked() noexcept;

    [[nodiscard]] double computeCurrentP50() const;
    [[nodiscard]] double computeBaselineP50Locked() const;

    std::string           _baselinePath;
    /// Monotonically increasing run counter. Read from baseline.json at
    /// ctor, incremented once, persisted before the first token flows.
    /// Assigned to this-process alerts + baseline samples so we can
    /// tell "regression appeared at version N" without needing a build
    /// commit tag or external timestamp.
    std::uint64_t         _internalVersion{0};

    // Decode-thread state — no lock needed.
    std::vector<double>   _window;
    std::size_t           _windowIdx{0};
    std::size_t           _windowFilled{0};
    std::size_t           _tokensSeen{0};

    // Cross-thread state — always accessed under _mutex.
    mutable std::mutex        _mutex;
    std::vector<RunSample>    _baselineSamples;
    std::optional<Alert>      _lastAlert;
    double                    _lastCurrentP50Ms{-1.0};
    double                    _lastBaselineP50Ms{-1.0};
};

} // namespace mimirmind::runtime