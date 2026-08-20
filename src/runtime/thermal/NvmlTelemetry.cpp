// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
#include "runtime/thermal/NvmlTelemetry.hpp"

#include <nvml.h>

namespace mimirmind::runtime::thermal {

namespace {
inline nvmlDevice_t asDev(void* p) {
    return static_cast<nvmlDevice_t>(p);
}
} // namespace

NvmlTelemetry::NvmlTelemetry() {
    const nvmlReturn_t ir = nvmlInit_v2();
    if (ir != NVML_SUCCESS) {
        _reason = std::string{"nvmlInit failed: "} + nvmlErrorString(ir);
        return;
    }
    nvmlDevice_t dev{};
    const nvmlReturn_t dr = nvmlDeviceGetHandleByIndex_v2(0, &dev);
    if (dr != NVML_SUCCESS) {
        _reason = std::string{"nvmlDeviceGetHandleByIndex failed: "} +
                  nvmlErrorString(dr);
        nvmlShutdown();
        return;
    }
    _device = static_cast<void*>(dev);

    char name[NVML_DEVICE_NAME_BUFFER_SIZE] = {0};
    if (nvmlDeviceGetName(dev, name, sizeof(name)) == NVML_SUCCESS) {
        _name = name;
    }
    _ok     = true;
    _reason = "ok";
}

NvmlTelemetry::~NvmlTelemetry() {
    if (_ok) {
        nvmlShutdown();
    }
}

NvmlTelemetry::Reading NvmlTelemetry::sample() const {
    Reading r{};
    if (!_ok) {
        return r;
    }
    nvmlDevice_t dev = asDev(_device);

    unsigned int mW = 0;
    if (nvmlDeviceGetPowerUsage(dev, &mW) == NVML_SUCCESS) {
        r.power_w = static_cast<double>(mW) / 1000.0;
    }
    unsigned int tC = 0;
    if (nvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU, &tC) == NVML_SUCCESS) {
        r.temp_c = static_cast<double>(tC);
    }
    unsigned int smMhz = 0;
    if (nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &smMhz) == NVML_SUCCESS) {
        r.sm_clock_mhz = smMhz;
    }
    unsigned int memMhz = 0;
    if (nvmlDeviceGetClockInfo(dev, NVML_CLOCK_MEM, &memMhz) == NVML_SUCCESS) {
        r.mem_clock_mhz = memMhz;
    }
    nvmlMemory_t mem{};
    if (nvmlDeviceGetMemoryInfo(dev, &mem) == NVML_SUCCESS) {
        r.mem_used_mib  = mem.used  / (1024ULL * 1024ULL);
        r.mem_total_mib = mem.total / (1024ULL * 1024ULL);
    }
    // Cumulative energy counter (millijoules). NOT_SUPPORTED on some GPUs ->
    // leave nullopt so callers fall back to instantaneous power.
    unsigned long long ej = 0;
    if (nvmlDeviceGetTotalEnergyConsumption(dev, &ej) == NVML_SUCCESS) {
        r.total_energy_mj = static_cast<std::uint64_t>(ej);
    }
    return r;
}

const NvmlTelemetry& nvmlTelemetry() {
    static const NvmlTelemetry inst;
    return inst;
}

} // namespace mimirmind::runtime::thermal
