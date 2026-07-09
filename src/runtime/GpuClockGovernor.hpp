#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <string_view>

namespace mimirmind::runtime {

class SystemMonitor;

/**
 * Dynamic Intel iGPU frequency governor.
 *
 * Linux exposes the iGPU's frequency-control interface under
 * /sys/class/drm/cardN/gt_*_freq_mhz. The governor reads the
 * hardware limits (RP0 = hardware max, RPn = hardware min) at
 * startup and writes the software cap (gt_max_freq_mhz) on demand
 * so the iGPU thermals stay below a configured target.
 *
 * Compared to the per-token pause approach in ThermalGuard, this
 * gives smooth latency under sustained load — the GPU runs at a
 * lower steady frequency rather than full-blast + idle bursts.
 *
 * Sysfs write semantics: i915 sometimes returns an error to the
 * write syscall even when the new cap was applied. The governor
 * always re-reads after writing and uses the read-back value as
 * authoritative. Callers should never trust setMaxFreqMhz's return
 * to mean "operation rejected" — only "new cap was at most this
 * value" (the kernel may also have clamped further).
 *
 * Container access: the bind-mounts for /sys/class/drm AND the
 * actual /sys/devices subtree must both be writable from inside
 * the container. See doc/setup-ct.md for the LXC + Docker config.
 */
class GpuClockGovernor {
public:
    /// Probes /sys/class/drm/card* for an iGPU with the gt_*_freq_mhz
    /// attributes. Sets the cap to RP0 on success (no cap). Never
    /// throws; check available() after construction. Optional sysfsRoot
    /// override for tests against a synthetic /sys tree.
    explicit GpuClockGovernor(std::string_view sysfsRoot = "/sys/class/drm");

    /// Restores RP0 on shutdown so the cap doesn't outlive the
    /// process.
    ~GpuClockGovernor();

    GpuClockGovernor(const GpuClockGovernor&)            = delete;
    GpuClockGovernor& operator=(const GpuClockGovernor&) = delete;

    [[nodiscard]] bool             available()         const noexcept { return _available; }
    [[nodiscard]] std::string_view unavailableReason() const noexcept { return _unavailableReason; }
    [[nodiscard]] std::string_view cardPath()          const noexcept { return _cardPath; }
    [[nodiscard]] std::uint32_t    rp0Mhz()            const noexcept { return _rp0Mhz; }
    [[nodiscard]] std::uint32_t    rpnMhz()            const noexcept { return _rpnMhz; }
    [[nodiscard]] std::uint32_t    currentCapMhz()     const noexcept { return _currentCap; }
    [[nodiscard]] float            targetTempC()       const noexcept { return _targetTempC; }

    /// Override the target temperature the P-controller drives the
    /// chip toward. Set from the thermal profile at startup.
    void setTargetTempC(float c) noexcept { _targetTempC = c; }

    /// Write a new max-freq cap. Caller may pass any value; it is
    /// clamped to [RPn, RP0] before write. Returns the cap after
    /// write (kernel-verified via re-read).
    std::uint32_t setMaxFreqMhz(std::uint32_t mhz);

    /// Session-level clock pin (M9.11.a). Applies `mhz` (clamped by
    /// setMaxFreqMhz) and records the intent ("rp0" / "rpn" / "numeric")
    /// and the raw `governor.gpuClockPin` string from config.json that
    /// produced it. The InferenceEngine's decode loop consults pinned()
    /// and skips its P-controller tick when true, so a pinned cap
    /// survives the full session. Reported through /v1/system/info and
    /// /system/status so any perf-bench number can be attributed to the
    /// pin it ran under.
    std::uint32_t pin(std::uint32_t mhz,
                      std::string_view intent,
                      std::string_view rawSource);

    [[nodiscard]] bool             pinned()     const noexcept { return _pinned; }
    [[nodiscard]] std::uint32_t    pinnedMhz()  const noexcept { return _pinnedMhz; }
    [[nodiscard]] std::string_view pinIntent()  const noexcept { return _pinIntent; }
    [[nodiscard]] std::string_view pinRawEnv()  const noexcept { return _pinRawEnv; }

    /// M9.6.6.0 tick-sink. Opens `path` for append and writes one NDJSON
    /// line per tick() call afterwards. Unset path (empty) or a path we
    /// cannot open leaves the sink inactive — tick() then costs one
    /// extra branch instead of a file write. Returns true on successful
    /// open.
    ///
    /// Line format:
    ///   {"ts_ms":..., "temp_c":..., "cap_before":..., "cap_after":...,
    ///    "delta_mhz":..., "error_c":..., "target_c":..., "pinned":<bool>}
    bool setTickLogPath(const std::string& path);

    [[nodiscard]] std::string_view tickLogPath() const noexcept {
        return _tickLogPath;
    }
    [[nodiscard]] bool tickLogOpen() const noexcept {
        return _tickLog.is_open();
    }

    /// Asymmetric P-controller: nudges current cap toward keeping
    /// `current_temp_c` at targetTempC(). The gain is direction-
    /// dependent so we drop the cap fast when overshooting target
    /// (heat-up dynamics are quick on this iGPU + UMA package) but
    /// raise it slowly when undershooting (cool-down is much slower
    /// than the temp swing a cap-raise causes). A small deadband
    /// around target prevents 1-MHz-per-tick oscillation in steady
    /// state. Returns the new cap.
    std::uint32_t adjustForTemp(float current_temp_c);

    /// Convenience: read current temp from `monitor` and call
    /// adjustForTemp. Returns the new cap, or currentCapMhz() if the
    /// monitor has no package temp reading.
    std::uint32_t tick(SystemMonitor& monitor);

private:
    void probe(std::string_view sysfsRoot);
    [[nodiscard]] static std::uint32_t readFreqFile(const std::string& path);

    // Asymmetric gains. Heat-up on this package (NUC14 MTL, ~15-20 W
    // sustained) reaches steady-state in ~5-10 s, cool-down takes
    // 30-60 s once heat is in the heatsink. A symmetric controller
    // overshoots on the cool side and re-spikes on the next workload
    // burst. Drop fast (kGainDown), creep up (kGainUp) — ratio 1:10.
    //
    // M9.6.2 relaxed this to 50/25 (ratio 1:2) on the assumption that
    // M9.6.1's per-request cap reset covered the worst case. That
    // assumption failed under sustained decode on 2026-07-01 — the
    // machine ran into a hard thermal shutdown that the software brakes
    // did not catch. Rolled back to the paranoid M9.6 values *and*
    // dropped the per-request reset: an 8 °C overshoot now drops the
    // cap by 800 MHz per tick again, and a request that follows a
    // cap-down inherits the throttled cap instead of starting at RP0.
    // Slower decode, but no shutdown. The NUC14 cooling lesson still
    // applies — physical airflow is the real fix.
    static constexpr float kGainUpMhzPerC      = 10.0F;
    static constexpr float kGainDownMhzPerC    = 100.0F;
    // Deadband around target — within this band the cap doesn't move.
    // Stops the 1-MHz-per-tick wiggle when the chip is sitting near
    // target. ±0.5 °C is below the resolution of x86_pkg_temp anyway.
    static constexpr float kDeadbandC           = 0.5F;
    static constexpr float kDefaultTargetTempC   = 72.0F;

    std::string   _cardPath{};
    std::string   _unavailableReason{};
    std::uint32_t _rp0Mhz{0};
    std::uint32_t _rpnMhz{0};
    std::uint32_t _currentCap{0};
    float         _targetTempC{kDefaultTargetTempC};
    bool          _available{false};

    // M9.11.a session-level pin. Set once via pin(); read by the engine's
    // decode loop (skip P-controller tick) and by ApiServer (report to
    // /system/info + /system/status).
    bool          _pinned{false};
    std::uint32_t _pinnedMhz{0};
    std::string   _pinIntent{};
    std::string   _pinRawEnv{};

    // M9.6.6.0 tick sink. When _tickLog is open, tick() writes one line
    // per invocation. NDJSON so downstream analysis (jq, pandas) can
    // consume it without a schema. Held open for the process lifetime;
    // destructor closes it.
    std::string   _tickLogPath{};
    std::ofstream _tickLog{};
};

} // namespace mimirmind::runtime
