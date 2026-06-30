#pragma once

#include <chrono>
#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

namespace mimirmind::runtime {

/**
 * Snapshot of the sensor readings the monitor knows how to expose.
 *
 * Each field is optional — if the corresponding sysfs/proc source is
 * not reachable (sensor not present, file not readable, parse failed),
 * the field stays nullopt and the caller decides whether that is a
 * problem (the thermal guard treats a missing required reading
 * conservatively).
 */
struct SystemReading {
    std::optional<float>       package_temp_c;
    std::optional<std::size_t> ram_total_mib;
    std::optional<std::size_t> ram_available_mib;
    std::chrono::steady_clock::time_point taken_at{};
};

/**
 * Reads host CPU package temperature and RAM availability from sysfs +
 * procfs. Designed to work inside an unprivileged Docker container —
 * sysfs and procfs are mounted by default; no `--privileged` or
 * `--cap-add` needed.
 *
 * Package temperature is probed in two tiers at construction time:
 *
 *   1. /sys/class/thermal/thermal_zone*\/type matching "x86_pkg_temp"
 *      (kernel-managed thermal subsystem, always present on Intel).
 *   2. /sys/class/hwmon/hwmon*\/name == "coretemp" with a temp*_label
 *      reading "Package id 0" (lower-level coretemp driver).
 *
 * If neither path is found, packageTempPath() returns empty and reads
 * will leave `package_temp_c` nullopt. Construct with `require=true`
 * to surface that as a hard error at startup instead.
 *
 * Reads are cached for `minRefreshInterval` to keep the per-call cost
 * negligible even when the engine polls between every decode token.
 */
class SystemMonitor {
public:
    /// `require*` toggles control whether the constructor throws when
    /// the corresponding sensor cannot be found. Pass true for each
    /// metric your thermal profile actually depends on.
    SystemMonitor(bool                      requirePackageTemp = false,
                  bool                      requireRam         = false,
                  std::chrono::milliseconds minRefreshInterval =
                      std::chrono::milliseconds{250});

    /// Fresh snapshot of the sensors. Reads cached within the refresh
    /// interval. Thread-unsafe — gate at the call site if needed.
    SystemReading read();

    [[nodiscard]] std::string_view packageTempPath()   const noexcept { return _packageTempPath; }
    [[nodiscard]] std::string_view packageTempSource() const noexcept { return _packageTempSource; }

private:
    void probePackageTempSource();
    [[nodiscard]] std::optional<float>       readPackageTempC() const;
    [[nodiscard]] std::pair<std::optional<std::size_t>,
                            std::optional<std::size_t>>
                                              readRam() const;

    std::string                            _packageTempPath{};
    std::string                            _packageTempSource{};
    SystemReading                          _cached{};
    std::chrono::steady_clock::time_point  _lastReadAt{};
    std::chrono::milliseconds              _minRefresh;
};

} // namespace mimirmind::runtime