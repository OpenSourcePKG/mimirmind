// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/perf/PerfRegressionDetector.hpp"

#include "core/log/Log.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <fstream>
#include <utility>

namespace mimirmind::runtime {

using nlohmann::json;

namespace {

std::int64_t nowUnix() noexcept {
    return static_cast<std::int64_t>(std::time(nullptr));
}

/// Median of the values in `v`. Non-const so we can sort in place.
/// Returns -1.0 for an empty input.
double medianInPlace(std::vector<double>& v) {
    if (v.empty()) {
        return -1.0;
    }
    std::sort(v.begin(), v.end());
    const std::size_t n = v.size();
    if ((n % 2U) == 1U) {
        return v[n / 2U];
    }
    return 0.5 * (v[(n / 2U) - 1U] + v[n / 2U]);
}

} // namespace

PerfRegressionDetector::PerfRegressionDetector(std::string_view baselineJsonPath)
    : _baselinePath(baselineJsonPath),
      _window(kRollingWindow, 0.0) {
    loadBaseline();  // fills _internalVersion from persisted state (0 if fresh)
    ++_internalVersion;
    {
        std::lock_guard<std::mutex> lock(_mutex);
        trimBaselineToWindowLocked();
    }
    // Persist immediately so a process that crashes before finishing a
    // run still burns its version number — otherwise a crash loop would
    // re-use the same version and confound the baseline history.
    persistBaseline();
    MM_LOG_INFO("perf_regression",
                "detector armed (version={}, baseline={} samples, path='{}')",
                _internalVersion,
                _baselineSamples.size(),
                _baselinePath);
}

PerfRegressionDetector::~PerfRegressionDetector() = default;

void PerfRegressionDetector::onDecodeToken(const Sample& s) noexcept {
    ++_tokensSeen;
    if (_tokensSeen <= kWarmupTokens) {
        return;
    }
    _window[_windowIdx] = s.wall_ms;
    _windowIdx = (_windowIdx + 1U) % kRollingWindow;
    if (_windowFilled < kRollingWindow) {
        ++_windowFilled;
    }
}

void PerfRegressionDetector::onRunComplete(std::size_t /*emittedTokens*/) noexcept {
    try {
        const double      currentP50 = computeCurrentP50();
        const std::size_t sampleN    = _windowFilled;

        // Reset the ring for the next run — done regardless of whether
        // the current run counts. A run that never got past warmup just
        // resets state and does nothing.
        _windowIdx    = 0;
        _windowFilled = 0;
        _tokensSeen   = 0;

        std::optional<Alert> emittedAlert;
        double               baselineP50 = -1.0;

        {
            std::lock_guard<std::mutex> lock(_mutex);

            _lastCurrentP50Ms  = currentP50;
            baselineP50        = computeBaselineP50Locked();
            _lastBaselineP50Ms = baselineP50;

            const bool haveCurrent  = currentP50  > 0.0 && sampleN >= kMinRunSamples;
            const bool haveBaseline = baselineP50 > 0.0 &&
                                      _baselineSamples.size() >= kMinBaselineN;

            if (haveCurrent && haveBaseline &&
                currentP50 > baselineP50 * kAlertThreshold) {
                Alert a{};
                a.current_p50_ms   = currentP50;
                a.baseline_p50_ms  = baselineP50;
                a.delta_ratio      = currentP50 / baselineP50;
                a.internal_version = _internalVersion;
                a.detected_unix    = nowUnix();
                _lastAlert   = a;
                emittedAlert = a;
            }

            // Promote the current run into the rolling baseline even if
            // it just alerted — the baseline naturally converges to the
            // "new normal" after enough follow-up runs. That is the right
            // behaviour: a persistent regression stops being an alert
            // once it becomes the baseline, but the *first* time it
            // shows up we still see it.
            if (haveCurrent) {
                RunSample rs{};
                rs.unix_sec         = nowUnix();
                rs.p50_ms           = currentP50;
                rs.n                = sampleN;
                rs.internal_version = _internalVersion;
                _baselineSamples.push_back(std::move(rs));
                trimBaselineToWindowLocked();
                // Alert check above used the pre-push baseline (that is
                // semantically right — a run should not be compared against
                // itself). But /v1/system/status reports "baseline right now",
                // which should include the run we just committed. Recompute.
                _lastBaselineP50Ms = computeBaselineP50Locked();
            }
        }

        if (emittedAlert) {
            MM_LOG_WARN(
                "perf_regression",
                "regression detected: p50={:.2f}ms baseline={:.2f}ms "
                "delta={:.2f}x threshold={:.2f}x version={}",
                emittedAlert->current_p50_ms,
                emittedAlert->baseline_p50_ms,
                emittedAlert->delta_ratio,
                kAlertThreshold,
                emittedAlert->internal_version);
        }

        persistBaseline();
    } catch (...) {
        // The detector must never break the inference loop. Swallow.
    }
}

std::optional<PerfRegressionDetector::Alert>
PerfRegressionDetector::lastAlert() const noexcept {
    std::lock_guard<std::mutex> lock(_mutex);
    return _lastAlert;
}

double PerfRegressionDetector::currentP50Ms() const noexcept {
    std::lock_guard<std::mutex> lock(_mutex);
    return _lastCurrentP50Ms;
}

double PerfRegressionDetector::baselineP50Ms() const noexcept {
    std::lock_guard<std::mutex> lock(_mutex);
    return _lastBaselineP50Ms;
}

std::size_t PerfRegressionDetector::baselineSampleCount() const noexcept {
    std::lock_guard<std::mutex> lock(_mutex);
    return _baselineSamples.size();
}

double PerfRegressionDetector::computeCurrentP50() const {
    if (_windowFilled == 0U) {
        return -1.0;
    }
    std::vector<double> copy(_window.begin(),
                             _window.begin() +
                                 static_cast<std::ptrdiff_t>(_windowFilled));
    return medianInPlace(copy);
}

double PerfRegressionDetector::computeBaselineP50Locked() const {
    if (_baselineSamples.size() < kMinBaselineN) {
        return -1.0;
    }
    std::vector<double> values;
    values.reserve(_baselineSamples.size());
    for (const auto& r : _baselineSamples) {
        values.push_back(r.p50_ms);
    }
    return medianInPlace(values);
}

void PerfRegressionDetector::loadBaseline() noexcept {
    try {
        std::ifstream in(_baselinePath);
        if (!in) {
            return;  // fresh start — file does not exist yet
        }
        json j;
        in >> j;
        if (!j.is_object()) {
            return;
        }
        _internalVersion = j.value("internal_version", std::uint64_t{0});
        if (!j.contains("samples") || !j["samples"].is_array()) {
            return;
        }
        for (const auto& s : j["samples"]) {
            RunSample rs{};
            rs.unix_sec         = s.value("unix_sec",         std::int64_t{0});
            rs.p50_ms           = s.value("p50_ms",           0.0);
            rs.n                = s.value("n",                std::size_t{0});
            rs.internal_version = s.value("internal_version", std::uint64_t{0});
            if (rs.unix_sec > 0 &&
                rs.p50_ms   > 0.0 &&
                rs.n        >= kMinRunSamples) {
                _baselineSamples.push_back(std::move(rs));
            }
        }
    } catch (const std::exception& e) {
        MM_LOG_WARN("perf_regression",
                    "baseline read failed ('{}'): {} — starting fresh",
                    _baselinePath, std::string_view{e.what()});
        _baselineSamples.clear();
        _internalVersion = 0;
    } catch (...) {
        _baselineSamples.clear();
        _internalVersion = 0;
    }
}

void PerfRegressionDetector::persistBaseline() noexcept {
    std::vector<RunSample> snapshot;
    {
        std::lock_guard<std::mutex> lock(_mutex);
        snapshot = _baselineSamples;
    }
    try {
        json samples = json::array();
        for (const auto& r : snapshot) {
            samples.push_back(json{
                {"unix_sec",         r.unix_sec},
                {"p50_ms",           r.p50_ms},
                {"n",                r.n},
                {"internal_version", r.internal_version},
            });
        }
        const json body{
            {"schema_version",   1},
            {"internal_version", _internalVersion},
            {"samples",          samples},
        };

        // Atomic replace: write to .tmp, rename over the real path.
        const std::string tmp = _baselinePath + ".tmp";
        {
            std::ofstream out(tmp, std::ios::trunc);
            if (!out) {
                MM_LOG_WARN("perf_regression",
                            "cannot open '{}' for write — baseline not "
                            "persisted this run",
                            tmp);
                return;
            }
            out << body.dump();
        }
        if (std::rename(tmp.c_str(), _baselinePath.c_str()) != 0) {
            MM_LOG_WARN("perf_regression",
                        "rename '{}' -> '{}' failed — baseline not "
                        "persisted this run",
                        tmp, _baselinePath);
            (void)std::remove(tmp.c_str());
        }
    } catch (const std::exception& e) {
        MM_LOG_WARN("perf_regression",
                    "baseline write failed: {}",
                    std::string_view{e.what()});
    } catch (...) {
    }
}

void PerfRegressionDetector::trimBaselineToWindowLocked() noexcept {
    const std::int64_t cutoff =
        nowUnix() -
        static_cast<std::int64_t>(kBaselineDays) * 24 * 3600;
    auto isStale = [cutoff](const RunSample& r) {
        return r.unix_sec < cutoff;
    };
    _baselineSamples.erase(
        std::remove_if(_baselineSamples.begin(),
                       _baselineSamples.end(),
                       isStale),
        _baselineSamples.end());
}

} // namespace mimirmind::runtime