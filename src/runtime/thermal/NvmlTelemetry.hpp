// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace mimirmind::runtime::thermal {

/// Read-only GPU telemetry via NVML (NVIDIA Management Library). This is
/// observability, NOT enforcement: it never changes clocks or power limits, it
/// only samples what the driver already exposes (the same numbers nvidia-smi
/// prints). Used to surface live power / temperature / clock / memory on the
/// CUDA (GB10 / Blackwell) backend, where the sysfs-based PowerMonitor and
/// GpuClockGovernor (Intel RAPL / AMD/Intel GPU sysfs) do not apply.
///
/// RAII: the ctor calls nvmlInit and resolves device 0; the dtor shuts NVML
/// down. `available()` is false if init failed (e.g. no driver) — every read
/// then returns std::nullopt and callers should omit the field.
class NvmlTelemetry {
public:
    struct Reading {
        std::optional<double>        power_w;         // GPU power draw (W)
        std::optional<double>        temp_c;          // GPU temperature (°C)
        std::optional<std::uint32_t> sm_clock_mhz;    // current SM clock (MHz)
        std::optional<std::uint32_t> mem_clock_mhz;   // current memory clock (MHz)
        std::optional<std::uint64_t> mem_used_mib;    // used device memory (MiB)
        std::optional<std::uint64_t> mem_total_mib;   // total device memory (MiB)
    };

    NvmlTelemetry();
    ~NvmlTelemetry();

    NvmlTelemetry(const NvmlTelemetry&)            = delete;
    NvmlTelemetry& operator=(const NvmlTelemetry&) = delete;

    [[nodiscard]] bool               available() const noexcept { return _ok; }
    [[nodiscard]] const std::string& unavailableReason() const noexcept {
        return _reason;
    }
    [[nodiscard]] const std::string& deviceName() const noexcept { return _name; }

    /// Sample the current readings. Cheap (a handful of NVML queries); safe to
    /// call per status request. Returns an all-nullopt Reading if unavailable.
    [[nodiscard]] Reading sample() const;

private:
    bool        _ok{false};
    std::string _reason{"not initialised"};
    std::string _name;
    void*       _device{nullptr};   // nvmlDevice_t (opaque, kept as void*)
};

} // namespace mimirmind::runtime::thermal
