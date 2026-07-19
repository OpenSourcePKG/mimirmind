// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/thermal/ThermalGuard.hpp"

#include <cstdio>
#include <sstream>
#include <string>

namespace mimirmind::runtime {

namespace {

/// fraction in [0, 1]: 0 = at-or-below soft, 1 = at-or-above hard.
/// Caller has already verified hard > soft.
float fractionFor(float reading, float soft, float hard) noexcept {
    if (reading <= soft) {
        return 0.0F;
    }
    if (reading >= hard) {
        return 1.0F;
    }
    return (reading - soft) / (hard - soft);
}

std::string formatTemp(float c) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%.1f", static_cast<double>(c));
    return std::string{buf};
}

} // namespace

ThermalDecision ThermalGuard::decide() {
    const auto reading = _monitor.read();

    ThermalDecision d{};

    if (!_profile.hasPackageLimits() || !reading.package_temp_c.has_value()) {
        // Profile disables monitoring or sensor unavailable — stay Ok.
        return d;
    }

    const float t    = *reading.package_temp_c;
    const float soft = *_profile.package_temp_soft_c;
    const float hard = *_profile.package_temp_hard_c;
    const float frac = fractionFor(t, soft, hard);

    d.pause = std::chrono::milliseconds{
        static_cast<int>(frac * static_cast<float>(_profile.package_throttle_max_ms))
    };

    if (frac >= 1.0F) {
        d.state             = ThermalDecision::State::Critical;
        d.admit_new_request = false;
        std::ostringstream os;
        os << "package_temp_c=" << formatTemp(t)
           << " at-or-above hard=" << formatTemp(hard);
        d.reason = os.str();
    } else if (frac > 0.0F) {
        d.state             = ThermalDecision::State::Throttling;
        d.admit_new_request = true;
        std::ostringstream os;
        os << "package_temp_c=" << formatTemp(t)
           << " above soft=" << formatTemp(soft);
        d.reason = os.str();
    }
    return d;
}

void ThermalGuard::checkAdmission() {
    const auto d = decide();
    if (!d.admit_new_request) {
        throw ThermalLimitExceeded(d.reason);
    }
}

std::chrono::milliseconds ThermalGuard::paceForCurrentReading() {
    return decide().pause;
}

SystemReading ThermalGuard::lastReading() {
    return _monitor.read();
}

} // namespace mimirmind::runtime