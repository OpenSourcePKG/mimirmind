// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/RequestTelemetryScope.hpp"

#include "runtime/thermal/FanController.hpp"
#include "runtime/thermal/ThermalGuard.hpp"

namespace mimirmind::runtime {

RequestTelemetryScope::RequestTelemetryScope(ThermalGuard*  thermal,
                                             FanController* fan,
                                             PowerMonitor*  power)
    : _fan{fan}, _power{power} {
    // 1. Thermal admission first — may throw before any side effect.
    if (thermal != nullptr) {
        thermal->checkAdmission();
    }
    // 2. Proactive fan boost.
    if (_fan != nullptr && _fan->available()) {
        (void)_fan->boost();
        _fanBoosted = true;
    }
    // 3. RAPL start snapshot.
    if (_power != nullptr && _power->available()) {
        _powerStart = _power->snapshot();
    }
}

RequestTelemetryScope::~RequestTelemetryScope() {
    if (_fanBoosted && _fan != nullptr && _fan->available()) {
        _fan->releaseToAuto();
    }
}

std::optional<double> RequestTelemetryScope::packageJoules() const {
    if (_power == nullptr || !_power->available() ||
        _powerStart.raw_energy_uj.empty()) {
        return std::nullopt;
    }
    const auto powerEnd = _power->snapshot();
    const auto joules   = _power->energyBetween(_powerStart, powerEnd);
    if (joules.empty()) {
        return std::nullopt;
    }
    return joules.front();
}

} // namespace mimirmind::runtime
