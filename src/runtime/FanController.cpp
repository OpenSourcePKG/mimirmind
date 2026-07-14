// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/FanController.hpp"

#include <algorithm>
#include <fstream>
#include <string>
#include <unistd.h> // access

namespace mimirmind::runtime {

namespace {

bool fileExists(const std::string& path) {
    std::ifstream f{path};
    return f.is_open();
}

bool isWritable(const std::string& path) {
    return ::access(path.c_str(), W_OK) == 0;
}

/// Best-effort integer read from a sysfs file. Returns -1 on any error
/// so the caller can decide whether that's fatal (probe rejection) or
/// benign (transient sensor read glitch).
int readIntFile(const std::string& path) {
    std::ifstream in{path};
    if (!in.is_open()) {
        return -1;
    }
    int v = -1;
    in >> v;
    return v;
}

std::string readTrimmed(const std::string& path) {
    std::ifstream in{path};
    if (!in.is_open()) {
        return {};
    }
    std::string s;
    std::getline(in, s);
    while (!s.empty() && (s.back() == '\n' || s.back() == ' ' || s.back() == '\r')) {
        s.pop_back();
    }
    return s;
}

/// Write an integer, best-effort. Returns true when the syscall
/// succeeded — but sysfs often accepts the write and then internally
/// clamps or rejects. Callers should always re-read to confirm.
bool writeInt(const std::string& path, int value) {
    std::ofstream out{path, std::ios::binary};
    if (!out.is_open()) {
        return false;
    }
    out << value;
    out.flush();
    return true;
}

} // namespace

FanController::FanController(std::string_view sysfsRoot) {
    probe(sysfsRoot);
}

FanController::~FanController() {
    if (!_available) {
        return;
    }
    // Best-effort restore. Order matters — write the pwm value first,
    // then the enable mode. Some EC firmware locks pwm writes as soon
    // as pwm_enable leaves manual mode.
    if (!_pwmPath.empty() && _originalPwm > 0) {
        (void)writeInt(_pwmPath, _originalPwm);
    }
    if (!_pwmEnablePath.empty() && _originalEnableMode >= 0) {
        (void)writeInt(_pwmEnablePath, _originalEnableMode);
    }
}

void FanController::probe(std::string_view sysfsRoot) {
    // Walk hwmon0..hwmon15. Kernel sometimes exposes 20+ nodes on
    // modern platforms (each PCH domain + each PCI device with a
    // sensor), but the fan controller is nearly always in the first
    // few. Cap at 16 to bound the probe cost.
    for (int i = 0; i < 16; ++i) {
        const std::string base = std::string{sysfsRoot} + "/hwmon" + std::to_string(i);
        const std::string namePath = base + "/name";

        if (!fileExists(namePath)) {
            continue;
        }
        const std::string chipName = readTrimmed(namePath);
        if (chipName.empty()) {
            continue;
        }

        // Find a fan[N]_input sensor. Kernel supports fan1..fan8 on
        // most chipsets; we pick the lowest-numbered one that reports
        // a non-zero baseline. A silent zero-RPM reading often means
        // the header is unpopulated, but some ECs report 0 briefly at
        // boot — accept 0 if it's the only fan node present.
        std::string fanInputCandidate;
        for (int f = 1; f <= 8; ++f) {
            const std::string p = base + "/fan" + std::to_string(f) + "_input";
            if (fileExists(p)) {
                fanInputCandidate = p;
                if (readIntFile(p) > 0) {
                    // Reporting spinning — prefer this one.
                    break;
                }
            }
        }
        if (fanInputCandidate.empty()) {
            continue;
        }

        // Find a writable pwm[N] + pwm[N]_enable pair.
        std::string pwmCandidate;
        std::string pwmEnableCandidate;
        for (int p = 1; p <= 8; ++p) {
            const std::string pwm       = base + "/pwm" + std::to_string(p);
            const std::string pwmEnable = pwm + "_enable";
            if (fileExists(pwm) && fileExists(pwmEnable)
                && isWritable(pwm) && isWritable(pwmEnable)) {
                pwmCandidate       = pwm;
                pwmEnableCandidate = pwmEnable;
                break;
            }
        }
        if (pwmCandidate.empty()) {
            continue;
        }

        // Capture baseline values. Both must read cleanly — if either
        // returns -1 we don't have a safe restore state.
        const int origPwm    = readIntFile(pwmCandidate);
        const int origEnable = readIntFile(pwmEnableCandidate);
        if (origPwm < 0 || origPwm > 255 || origEnable < 0) {
            continue;
        }

        _chipPath          = base;
        _chipName          = chipName;
        _pwmPath           = pwmCandidate;
        _pwmEnablePath     = pwmEnableCandidate;
        _fanInputPath      = fanInputCandidate;
        _originalPwm       = static_cast<std::uint8_t>(origPwm);
        _originalEnableMode = origEnable;
        _available         = true;
        return;
    }

    _unavailableReason =
        "no /sys/class/hwmon/hwmon* found with fan_input + writable pwm/pwm_enable — "
        "check container caps (SYS_ADMIN or rw bind mount on /sys/class/hwmon), "
        "and BIOS fan mode (must permit software override)";
}

std::uint8_t FanController::currentPwm() const {
    if (!_available) return 0;
    const int v = readIntFile(_pwmPath);
    if (v < 0 || v > 255) return 0;
    return static_cast<std::uint8_t>(v);
}

std::uint32_t FanController::currentFanRpm() const {
    if (!_available) return 0;
    const int v = readIntFile(_fanInputPath);
    return v <= 0 ? 0U : static_cast<std::uint32_t>(v);
}

int FanController::currentEnableMode() const {
    if (!_available) return -1;
    return readIntFile(_pwmEnablePath);
}

void FanController::setBoostPwm(std::uint8_t v) noexcept {
    _boostPwm = std::max<std::uint8_t>(v, kAbsoluteMinPwm);
}

void FanController::setMinSafePwm(std::uint8_t v) noexcept {
    _minSafePwm = std::clamp<std::uint8_t>(v, kAbsoluteMinPwm, 200);
}

std::uint8_t FanController::boost() {
    if (!_available) {
        return 0;
    }
    // Enable manual mode first so the pwm write is not immediately
    // overridden by the auto controller.
    (void)writeInt(_pwmEnablePath, 1);
    (void)writeInt(_pwmPath, _boostPwm);
    _boostActive = true;
    return currentPwm();
}

void FanController::releaseToAuto() {
    if (!_available) {
        return;
    }
    (void)writeInt(_pwmEnablePath, _originalEnableMode);
    _boostActive = false;
}

std::uint8_t FanController::setPwmPercent(std::uint8_t percent) {
    if (!_available) {
        return 0;
    }
    const std::uint8_t clampedPct =
        std::min<std::uint8_t>(percent, 100);
    // (percent * 255 + 50) / 100 with integer rounding.
    const std::uint16_t raw =
        static_cast<std::uint16_t>((clampedPct * 255U + 50U) / 100U);
    return setPwmRaw(static_cast<std::uint8_t>(raw));
}

std::uint8_t FanController::setPwmRaw(std::uint8_t value) {
    if (!_available) {
        return 0;
    }
    const std::uint8_t clamped = std::max<std::uint8_t>(value, _minSafePwm);
    (void)writeInt(_pwmEnablePath, 1);
    (void)writeInt(_pwmPath, clamped);
    _boostActive = (clamped >= _boostPwm);
    return currentPwm();
}

} // namespace mimirmind::runtime