#include "runtime/GpuClockGovernor.hpp"

#include "runtime/SystemMonitor.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <string>

namespace mimirmind::runtime {

namespace {

bool fileExists(const std::string& path) {
    std::ifstream f{path};
    return f.is_open();
}

bool writeFreq(const std::string& path, std::uint32_t mhz) {
    // We open in binary mode to avoid any trailing whitespace
    // surprises; i915 is fussy about input format on some kernel
    // versions. The actual return value is discarded — i915
    // sometimes signals failure even when the kernel did apply
    // the new cap. Caller re-reads to confirm.
    std::ofstream out{path, std::ios::binary};
    if (!out.is_open()) {
        return false;
    }
    out << mhz;
    out.flush();
    return true;
}

} // namespace

GpuClockGovernor::GpuClockGovernor(std::string_view sysfsRoot) {
    probe(sysfsRoot);
}

GpuClockGovernor::~GpuClockGovernor() {
    if (_available) {
        // Best-effort restore. Don't throw, don't log — destructor.
        (void)writeFreq(_cardPath + "/gt_max_freq_mhz", _rp0Mhz);
    }
}

std::uint32_t GpuClockGovernor::readFreqFile(const std::string& path) {
    std::ifstream in{path};
    if (!in.is_open()) {
        return 0;
    }
    std::uint32_t v = 0;
    in >> v;
    return v;
}

void GpuClockGovernor::probe(std::string_view sysfsRoot) {
    // Walk card0..card15 looking for an entry with gt_max_freq_mhz
    // present + readable + RP0 readable. The first one that's
    // writable wins. Skip cards that exist but aren't iGPU (no
    // gt_*_freq_mhz attributes — e.g. virtual DRM devices).
    for (int i = 0; i < 16; ++i) {
        std::string base = std::string{sysfsRoot} + "/card" + std::to_string(i);
        std::string max_path = base + "/gt_max_freq_mhz";
        std::string rp0_path = base + "/gt_RP0_freq_mhz";
        std::string rpn_path = base + "/gt_RPn_freq_mhz";

        if (!fileExists(max_path) || !fileExists(rp0_path) || !fileExists(rpn_path)) {
            continue;
        }
        const std::uint32_t rp0 = readFreqFile(rp0_path);
        const std::uint32_t rpn = readFreqFile(rpn_path);
        const std::uint32_t cur = readFreqFile(max_path);
        if (rp0 == 0 || rpn == 0 || cur == 0) {
            // Sysfs file present but unreadable — likely a DRM
            // sub-node we don't care about.
            continue;
        }

        // Test write access by setting the cap back to itself
        // (genuine no-op). i915 may signal failure but the file
        // will still be open + writable if the mount is rw.
        if (!writeFreq(max_path, cur)) {
            // Open-for-write failed → mount is ro, skip this card.
            continue;
        }
        const std::uint32_t verified = readFreqFile(max_path);
        if (verified == 0) {
            continue;
        }

        _cardPath   = std::move(base);
        _rp0Mhz     = rp0;
        _rpnMhz     = rpn;
        _currentCap = verified;
        _available  = true;
        return;
    }

    _unavailableReason = "no /sys/class/drm/card* found with writable "
                         "gt_max_freq_mhz — check LXC + Docker mount "
                         "config (see doc/setup-ct.md)";
}

std::uint32_t GpuClockGovernor::setMaxFreqMhz(std::uint32_t mhz) {
    if (!_available) {
        return 0;
    }
    mhz = std::clamp(mhz, _rpnMhz, _rp0Mhz);
    (void)writeFreq(_cardPath + "/gt_max_freq_mhz", mhz);
    _currentCap = readFreqFile(_cardPath + "/gt_max_freq_mhz");
    return _currentCap;
}

std::uint32_t GpuClockGovernor::resetToMax() noexcept {
    if (!_available) {
        return 0;
    }
    return setMaxFreqMhz(_rp0Mhz);
}

std::uint32_t GpuClockGovernor::adjustForTemp(float current_temp_c) {
    if (!_available) {
        return 0;
    }
    const float error = current_temp_c - _targetTempC;

    // Deadband — within ±kDeadbandC of target the cap is left alone.
    if (error >= -kDeadbandC && error <= kDeadbandC) {
        return _currentCap;
    }

    // Asymmetric gain: drop fast on overshoot, creep up on undershoot.
    const float gain   = (error > 0.0F) ? kGainDownMhzPerC : kGainUpMhzPerC;
    const float deltaF = -error * gain;
    // Round toward zero so a sub-1-MHz target doesn't churn the cap.
    const std::int32_t delta = static_cast<std::int32_t>(deltaF);
    const std::int64_t target =
        static_cast<std::int64_t>(_currentCap) + delta;
    const std::uint32_t clamped = static_cast<std::uint32_t>(
        std::clamp<std::int64_t>(target, _rpnMhz, _rp0Mhz));
    return setMaxFreqMhz(clamped);
}

std::uint32_t GpuClockGovernor::tick(SystemMonitor& monitor) {
    if (!_available) {
        return 0;
    }
    const auto reading = monitor.read();
    if (!reading.package_temp_c.has_value()) {
        return _currentCap;
    }
    return adjustForTemp(*reading.package_temp_c);
}

} // namespace mimirmind::runtime