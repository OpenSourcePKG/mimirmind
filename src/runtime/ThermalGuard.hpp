// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "runtime/SystemMonitor.hpp"
#include "runtime/ThermalProfile.hpp"

#include <chrono>
#include <stdexcept>
#include <string>

namespace mimirmind::runtime {

/// Thrown by ThermalGuard::checkAdmission() when any "hard" limit in
/// the active profile is breached. ApiServer catches this and turns it
/// into an HTTP 503 with a Retry-After hint.
class ThermalLimitExceeded : public std::runtime_error {
public:
    explicit ThermalLimitExceeded(const std::string& msg)
        : std::runtime_error{msg} {}
};

/// Snapshot of what the guard would do given the latest sensor reading.
/// Exposed verbatim via GET /v1/system/status.
struct ThermalDecision {
    enum class State {
        Ok,         ///< Below all soft limits.
        Throttling, ///< Above a soft limit but below the hard limit.
        Critical,   ///< At or above a hard limit — admit_new = false.
    };
    State                     state{State::Ok};
    std::chrono::milliseconds pause{0};
    bool                      admit_new_request{true};
    std::string               reason{};   // free-form, for humans + status page
};

/**
 * Combines a SystemMonitor (where am I right now) with a ThermalProfile
 * (what this machine can handle) and exposes two questions to the engine:
 *
 *   - May this request be admitted at all? (checkAdmission, throws)
 *   - How long should we sleep between decode tokens? (paceForCurrentReading)
 *
 * The guard never mutates the profile. It does cache the latest reading
 * (via SystemMonitor's own refresh-interval cache) so /v1/system/status
 * is cheap to poll.
 *
 * Decision policy: each enabled metric independently computes a "fraction
 * of the way from soft to hard". The worst fraction wins. Pause is
 * fraction × metric.throttle_max_ms.
 */
class ThermalGuard {
public:
    ThermalGuard(ThermalProfile profile, SystemMonitor& monitor) noexcept
        : _profile{std::move(profile)}, _monitor{monitor} {}

    /// Read sensors, throw ThermalLimitExceeded if any hard limit is hit.
    /// Caller (the engine) calls this exactly once at the start of each
    /// generate(), before prefill.
    void checkAdmission();

    /// Read sensors and return the current pause-per-token recommendation.
    /// Returns 0 when below all soft limits or when monitoring is unconfigured.
    [[nodiscard]] std::chrono::milliseconds paceForCurrentReading();

    /// Full decision snapshot for diagnostics / status endpoint.
    [[nodiscard]] ThermalDecision decide();

    [[nodiscard]] const ThermalProfile& profile() const noexcept { return _profile; }
    [[nodiscard]] SystemReading         lastReading();

private:
    ThermalProfile _profile;
    SystemMonitor& _monitor;
};

} // namespace mimirmind::runtime