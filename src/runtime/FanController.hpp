#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace mimirmind::runtime {

/**
 * Software-side chassis-fan controller for boost-on-inference.
 *
 * At construction time the controller walks /sys/class/hwmon/hwmon* and
 * picks the first chip that exposes a triple:
 *   - `name` file (chip identifier — e.g. "nct6798", "coretemp", "iwlwifi")
 *   - at least one `fan[N]_input` (RPM sensor)
 *   - at least one `pwm[N]` + `pwm[N]_enable` pair that is writable
 *
 * "Writable" is checked via access(W_OK) so nothing is dirtied during
 * probe. The first suitable chip wins. Systems with multiple fans get
 * the first-numbered one — good enough for a single-fan NUC14, will
 * need per-fan selection later if we ever run on a chassis with
 * asymmetric cooling zones.
 *
 * The controller records the original pwm value and pwm_enable mode
 * observed at startup so the destructor can restore them — the BIOS's
 * fan curve stays intact once mimirmind exits. Do not use with
 * `Ctrl-C` in a debugger session that skips destructors.
 *
 * Boost policy:
 *   - boost()  → pwm_enable = 1 (manual), pwm = boostPwm (default 255)
 *   - releaseToAuto() → pwm_enable = original captured mode
 *   - setPwmPercent(0-100) → manual mode, pwm clamped to
 *     [minSafePwm, 255]. Never writes 0 — a stopped fan on a passively
 *     cooled chassis is a thermal shutdown hazard.
 *
 * Not thread-safe. The engine calls boost()/releaseToAuto() at request
 * boundaries so single-threaded access is fine; if we later want per-
 * token PWM modulation from the decode loop we'll need a mutex.
 *
 * Container access: the docker image must have /sys mounted read-write
 * (Docker default) AND the underlying kernel driver must permit writes
 * to pwm*_enable. Some Intel EC firmware refuses manual mode from
 * userspace; in that case available() returns false with an explanatory
 * reason. See doc/setup-ct.md.
 */
class FanController {
public:
    /// Probes /sys/class/hwmon/* for a fan controller. Never throws;
    /// check available() after construction. Optional sysfsRoot override
    /// for tests against a synthetic /sys tree.
    explicit FanController(std::string_view sysfsRoot = "/sys/class/hwmon");

    /// Restores original pwm + pwm_enable on shutdown so the BIOS
    /// fan curve resumes control after the process exits.
    ~FanController();

    FanController(const FanController&)            = delete;
    FanController& operator=(const FanController&) = delete;

    [[nodiscard]] bool             available()         const noexcept { return _available; }
    [[nodiscard]] std::string_view unavailableReason() const noexcept { return _unavailableReason; }

    [[nodiscard]] std::string_view chipPath()          const noexcept { return _chipPath; }
    [[nodiscard]] std::string_view chipName()          const noexcept { return _chipName; }
    [[nodiscard]] std::string_view pwmPath()           const noexcept { return _pwmPath; }
    [[nodiscard]] std::string_view pwmEnablePath()     const noexcept { return _pwmEnablePath; }
    [[nodiscard]] std::string_view fanInputPath()      const noexcept { return _fanInputPath; }

    /// Original values captured at startup. Used for RAII restore
    /// and reported through /system/info so the operator can see
    /// what the BIOS default was before we touched anything.
    [[nodiscard]] std::uint8_t originalPwm()        const noexcept { return _originalPwm; }
    [[nodiscard]] int          originalEnableMode() const noexcept { return _originalEnableMode; }

    /// Live sysfs reads. Return 0 when unavailable.
    [[nodiscard]] std::uint8_t  currentPwm()       const;
    [[nodiscard]] std::uint32_t currentFanRpm()    const;
    [[nodiscard]] int           currentEnableMode() const;

    /// True after boost() and until the next releaseToAuto() /
    /// setPwmPercent() with a non-boost value.
    [[nodiscard]] bool boostActive() const noexcept { return _boostActive; }

    /// Configured limits. Tunable from main.cpp based on env vars.
    [[nodiscard]] std::uint8_t boostPwm()   const noexcept { return _boostPwm; }
    [[nodiscard]] std::uint8_t minSafePwm() const noexcept { return _minSafePwm; }

    /// Adjust the boost target. Clamped to [kAbsoluteMinPwm, 255].
    /// Applied on the next boost() call, not retroactively.
    void setBoostPwm(std::uint8_t v) noexcept;

    /// Adjust the minimum-safe PWM floor. Clamped to
    /// [kAbsoluteMinPwm, 200] so we never allow "fan-off" as a safe
    /// floor. Applied on the next setPwmPercent() call.
    void setMinSafePwm(std::uint8_t v) noexcept;

    /// Switch to manual mode and drive PWM to boostPwm. Returns the
    /// verified read-back pwm value (i915-style: kernel may clamp).
    /// No-op when not available().
    std::uint8_t boost();

    /// Switch pwm_enable back to the mode observed at startup. The
    /// last-written pwm value is left in place — most EC firmware
    /// ignores it once auto mode resumes control.
    void releaseToAuto();

    /// Manual PWM at `percent` of full. Clamped to
    /// [minSafePercent(), 100]. Enables manual mode as a side effect.
    /// Returns the verified read-back pwm value.
    std::uint8_t setPwmPercent(std::uint8_t percent);

    /// Manual PWM at raw 0-255 value. Clamped to [minSafePwm, 255].
    /// Enables manual mode as a side effect.
    std::uint8_t setPwmRaw(std::uint8_t value);

private:
    void probe(std::string_view sysfsRoot);

    /// Absolute floor. Writing lower would stop the fan on most EC
    /// firmwares — thermal-shutdown risk on a passively cooled NUC14.
    /// Nothing (env vars included) can push minSafePwm below this.
    static constexpr std::uint8_t kAbsoluteMinPwm    = 32;   // ~12 %

    /// Sensible defaults. Overridden from env vars in main.cpp.
    static constexpr std::uint8_t kDefaultMinSafePwm = 64;   // ~25 %
    static constexpr std::uint8_t kDefaultBoostPwm   = 255;  // 100 %

    std::string   _chipPath{};
    std::string   _chipName{};
    std::string   _pwmPath{};
    std::string   _pwmEnablePath{};
    std::string   _fanInputPath{};
    std::string   _unavailableReason{};

    std::uint8_t  _originalPwm{0};
    int           _originalEnableMode{-1};

    std::uint8_t  _boostPwm{kDefaultBoostPwm};
    std::uint8_t  _minSafePwm{kDefaultMinSafePwm};

    bool          _available{false};
    bool          _boostActive{false};
};

} // namespace mimirmind::runtime