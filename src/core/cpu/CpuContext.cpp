// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/cpu/CpuContext.hpp"

namespace mimirmind::core::cpu {

CpuBackend::CpuBackend() {
    // Minimal descriptor — CPU model / core count / cache sizes would
    // come from parsing /proc/cpuinfo. Skeleton stage returns a static
    // "cpu" name; refinement lands with the M-CPU.4 fill-in commits if
    // /v1/system/status ends up wanting real numbers.
    _info.name             = "cpu";
    _info.uuid             = {};
    _info.vendorId         = 0;
    _info.deviceId         = 0;
    _info.kind             = ::mimirmind::core::backend::DeviceKind::Cpu;
    _info.numComputeUnits  = 0;
    _info.totalLocalMem    = 0;
    _info.maxMemAllocSize  = 0;
    _info.coreClockRate    = 0;
}

::mimirmind::core::backend::BackendKind
CpuBackend::kind() const noexcept {
    return ::mimirmind::core::backend::BackendKind::Cpu;
}

const ::mimirmind::core::backend::BackendDeviceInfo&
CpuBackend::deviceInfo() const noexcept {
    return _info;
}

bool CpuBackend::hasFeature(
    ::mimirmind::core::backend::BackendFeature f) const noexcept {
    using ::mimirmind::core::backend::BackendFeature;
    // Only UnifiedMemoryHost — a device pointer IS a host pointer here,
    // by definition. No matrix engine (SIMD is a separate later story),
    // no IPC handle export, no HW dot-product intrinsics via this path,
    // no MutableCommandLists (nothing to record).
    switch (f) {
        case BackendFeature::UnifiedMemoryHost:  return true;
        case BackendFeature::MutableCommandLists:
        case BackendFeature::IntegerDotProduct:
        case BackendFeature::IpcHandleExport:
        case BackendFeature::MatrixEngine:
            return false;
    }
    return false;
}

::mimirmind::core::backend::BackendProbe
probeBackend() noexcept {
    return ::mimirmind::core::backend::BackendProbe{
        ::mimirmind::core::backend::BackendKind::Cpu,
        /*compiledIn=*/true,
        /*available =*/true,
        "cpu reference backend — always available",
    };
}

std::unique_ptr<::mimirmind::core::backend::ComputeContext>
createComputeContext() {
    return std::make_unique<CpuContext>();
}

} // namespace mimirmind::core::cpu