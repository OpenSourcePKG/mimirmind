#pragma once

#include <cstdint>
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

    /// Reset the cap to RP0 (hardware max). Called by InferenceEngine
    /// at the start of each generate() call so a request that follows
    /// an earlier cap-down isn't forced to start at the throttled cap.
    /// The asymmetric controller's slow up-gain (kGainUpMhzPerC = 10)
    /// would otherwise take many ticks to recover, and ticks only fire
    /// during decode — so without a per-request reset the cap stays
    /// pinned at wherever the previous request left it. Returns the
    /// cap after write (= RP0 unless the kernel refused).
    std::uint32_t resetToMax() noexcept;

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
    // burst. Drop faster (kGainDown), creep up (kGainUp) — ratio 1:2.
    //
    // M9.6 shipped at 1:10 (100/10) because there was no inter-request
    // recovery — once we hit RPn we stayed there. M9.6.1 added a
    // per-request resetToMax(), which catches the worst case at the
    // request boundary; the in-run controller can afford to be less
    // paranoid. M9.6.2 retunes to 50/25 so a 5 °C overshoot drops the
    // cap by 250 MHz instead of 500, keeping the iGPU at a usable
    // mid-frequency through ~6-7 ticks of sustained load instead of
    // floor-clamping after 3.
    static constexpr float kGainUpMhzPerC      = 25.0F;
    static constexpr float kGainDownMhzPerC    = 50.0F;
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
};

} // namespace mimirmind::runtime