// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "runtime/thermal/PowerMonitor.hpp"

#include <optional>

namespace mimirmind::runtime {

class FanController;
class ThermalGuard;

/**
 * RAII bundle for the per-request environmental telemetry that brackets a
 * single InferenceEngine::generate() call. Extracted from generate() so the
 * hot path reads as orchestration (roadmap 8.30.11.1).
 *
 * Construction performs, in order:
 *   1. M9.2 thermal admission — `ThermalGuard::checkAdmission()` throws
 *      ThermalLimitExceeded if a hard limit is currently breached (the caller
 *      turns that into HTTP 503). This runs FIRST, before any fan/power side
 *      effect, so a rejected request leaves the hardware untouched.
 *   2. M9.11.b proactive fan boost — ramp the chassis fan up before prefill so
 *      the GPU clock governor has thermal headroom when matmul starts pulling
 *      watts.
 *   3. RAPL snapshot — capture package energy counters so packageJoules() can
 *      report how much energy the request consumed.
 *
 * Destruction releases the fan back to auto. Because it is RAII, an exception
 * anywhere in prefill/decode still releases the fan — leaving it pinned at
 * 100 % across an idle period would be loud and would burn power.
 *
 * Every collaborator is optional (non-owning pointer, may be nullptr) and each
 * step no-ops when its device is absent or unavailable.
 */
class RequestTelemetryScope {
public:
    /// @throws whatever `thermal->checkAdmission()` throws (ThermalLimitExceeded)
    RequestTelemetryScope(ThermalGuard*   thermal,
                          FanController*  fan,
                          PowerMonitor*   power);
    ~RequestTelemetryScope();

    RequestTelemetryScope(const RequestTelemetryScope&)            = delete;
    RequestTelemetryScope& operator=(const RequestTelemetryScope&) = delete;
    RequestTelemetryScope(RequestTelemetryScope&&)                 = delete;
    RequestTelemetryScope& operator=(RequestTelemetryScope&&)      = delete;

    /// Package (socket) energy in Joules consumed since construction, or
    /// std::nullopt when the monitor is unavailable or produced no reading.
    /// The first discovered RAPL domain is the package socket (intel-rapl:0).
    [[nodiscard]] std::optional<double> packageJoules() const;

private:
    FanController*          _fan{nullptr};
    PowerMonitor*          _power{nullptr};
    PowerMonitor::Snapshot _powerStart{};
    bool                   _fanBoosted{false};
};

} // namespace mimirmind::runtime
