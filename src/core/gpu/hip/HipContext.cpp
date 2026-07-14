// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/gpu/hip/HipContext.hpp"

#include "core/log/Log.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <string>
#include <string_view>

namespace mimirmind::core::hip {

namespace {

// Wrap a HIP call — throw HipError on non-Success. Kept tiny; consumers
// that expect specific error codes should check hipGetLastError directly.
inline void hipCheck(const std::string& call, hipError_t code) {
    if (code != hipSuccess) {
        throw HipError(call, code);
    }
}

// Prefix-match on the gfxArch string. RDNA family detection is enough
// for the initial feature-flag heuristic — precise arch enumeration
// (gfx1100 vs gfx1101 vs gfx1102) matters at kernel-selection time, not
// at feature-probe time.
[[nodiscard]] bool archIsRdna3OrNewer(std::string_view arch) {
    // gfx11xx = RDNA3 (RX 7000-series, Phoenix APU, Hawk Point iGPU).
    // gfx12xx = RDNA4 (RX 9000-series). Both have WMMA.
    return arch.starts_with("gfx11") || arch.starts_with("gfx12");
}

[[nodiscard]] bool archIsRdna2OrNewer(std::string_view arch) {
    // gfx10xx = RDNA/RDNA2, gfx11xx = RDNA3, gfx12xx = RDNA4.
    // v_dot4_i32_i8 landed on gfx1030 (RDNA2), and every RDNA generation
    // since has kept it. RDNA1 (gfx1010, gfx1012) lacks it — but those
    // predate ROCm consumer support anyway.
    return arch.starts_with("gfx103") || arch.starts_with("gfx104")
        || arch.starts_with("gfx11")  || arch.starts_with("gfx12");
}

[[nodiscard]] bool archIsCdna(std::string_view arch) {
    // gfx908 = MI100, gfx90a = MI200, gfx940/gfx941/gfx942 = MI300.
    // All CDNA and all have MFMA matrix engines.
    return arch == "gfx908" || arch == "gfx90a"
        || arch.starts_with("gfx94");
}

[[nodiscard]] ::mimirmind::core::gpu::DeviceKind classifyKind(bool integrated) {
    using ::mimirmind::core::gpu::DeviceKind;
    return integrated ? DeviceKind::GpuIntegrated : DeviceKind::GpuDiscrete;
}

// Populate the portable and the HIP-native info blocks from
// hipDeviceProp_t. Kept as a free function so the ctor stays short.
void populate(const hipDeviceProp_t& p,
              int index,
              DeviceInfo& info,
              ::mimirmind::core::gpu::BackendDeviceInfo& backendInfo)
{
    info.name             = p.name;
    info.gfxArch          = p.gcnArchName; // "gfx1101" etc.
    // hipDeviceProp_t doesn't expose PCI vendor/device ids directly on all
    // HIP versions; leave 0 unless we later read /sys. AMD is always
    // 0x1002 so the vendorId can be hardcoded at that point.
    info.vendorId         = 0x1002u;
    info.deviceId         = 0u;
    info.hipDeviceIndex   = index;
    info.numComputeUnits  = p.multiProcessorCount;
    info.totalVram        = static_cast<std::size_t>(p.totalGlobalMem);
    info.maxAllocSize     = static_cast<std::size_t>(p.totalGlobalMem);
    info.warpSize         = p.warpSize;
    info.coreClockRateKhz = p.clockRate;
    info.isIntegrated     = p.integrated != 0;

    backendInfo.name             = info.name;
    backendInfo.uuid             = ""; // HIP has hipDeviceProp_t::uuid on ROCm 6+ but skip for the skeleton
    backendInfo.vendorId         = info.vendorId;
    backendInfo.deviceId         = info.deviceId;
    backendInfo.kind             = classifyKind(info.isIntegrated);
    backendInfo.numComputeUnits  = static_cast<std::uint32_t>(info.numComputeUnits);
    backendInfo.totalLocalMem    = info.totalVram;
    backendInfo.maxMemAllocSize  = info.maxAllocSize;
    backendInfo.coreClockRate    = static_cast<std::uint32_t>(info.coreClockRateKhz / 1000);
}

// Pick a device: prefer first discrete GPU, fall back to first
// integrated. Returns -1 if HIP reports zero devices.
[[nodiscard]] int selectDefaultDevice(int count) {
    int fallback = -1;
    for (int i = 0; i < count; ++i) {
        hipDeviceProp_t p{};
        if (hipGetDeviceProperties(&p, i) != hipSuccess) continue;
        if (p.integrated == 0) {
            return i;               // first dGPU wins
        }
        if (fallback < 0) fallback = i; // remember first iGPU as backup
    }
    return fallback;
}

} // namespace

// ---------- HipError --------------------------------------------------------

HipError::HipError(const std::string& call, hipError_t code)
    : std::runtime_error(call + ": " + hipGetErrorString(code)),
      _code(code) {}

// ---------- HipContext ------------------------------------------------------

HipContext::HipContext(int deviceIndex) {
    int count = 0;
    hipCheck("hipGetDeviceCount", hipGetDeviceCount(&count));
    if (count <= 0) {
        throw HipError("hipGetDeviceCount", hipErrorNoDevice);
    }

    if (deviceIndex < 0) {
        deviceIndex = selectDefaultDevice(count);
    }
    if (deviceIndex < 0 || deviceIndex >= count) {
        throw HipError("HipContext(deviceIndex out of range)", hipErrorInvalidDevice);
    }

    hipCheck("hipSetDevice", hipSetDevice(deviceIndex));

    hipDeviceProp_t p{};
    hipCheck("hipGetDeviceProperties", hipGetDeviceProperties(&p, deviceIndex));

    _deviceIdx = deviceIndex;
    populate(p, deviceIndex, _info, _backendInfo);

    MM_LOG_INFO("HipContext",
                "bound to device #{} ({}, arch={}, CUs={}, warp={})",
                _deviceIdx, _info.name, _info.gfxArch,
                _info.numComputeUnits, _info.warpSize);
}

HipContext::~HipContext() = default;

::mimirmind::core::gpu::BackendKind HipContext::kind() const noexcept {
    return ::mimirmind::core::gpu::BackendKind::Hip;
}

const ::mimirmind::core::gpu::BackendDeviceInfo& HipContext::deviceInfo() const noexcept {
    return _backendInfo;
}

bool HipContext::hasFeature(::mimirmind::core::gpu::BackendFeature f) const noexcept {
    using ::mimirmind::core::gpu::BackendFeature;
    const std::string_view arch = _info.gfxArch;

    switch (f) {
    case BackendFeature::MutableCommandLists:
        // hipGraph exists on all supported ROCm versions. Whether the
        // CLR use-case ports 1:1 is a runtime property of the recorded
        // graph, not the driver — see the MoE+CLR incompatibility
        // synaipse note. The flag here is "the API is present".
        return true;

    case BackendFeature::IntegerDotProduct:
        return archIsRdna2OrNewer(arch) || archIsCdna(arch);

    case BackendFeature::IpcHandleExport:
        // hipIpcGetMemHandle exists across ROCm 5+. Actually working
        // across processes on consumer RDNA has historical rough
        // edges — trust the API surface for now, gate at Munin ADR
        // time if we hit issues.
        return true;

    case BackendFeature::UnifiedMemoryHost:
        // Integrated APU (Phoenix/Hawk Point): system RAM is the same
        // memory pool, hipMallocManaged / hipHostMalloc gives zero-copy.
        // dGPU: PCIe copies still needed.
        return _info.isIntegrated;

    case BackendFeature::MatrixEngine:
        // RDNA3+: WMMA (via __builtin_amdgcn_wmma_*_w32_*).
        // CDNA: MFMA. Both count for the neutral flag.
        return archIsRdna3OrNewer(arch) || archIsCdna(arch);
    }
    return false;
}

} // namespace mimirmind::core::hip