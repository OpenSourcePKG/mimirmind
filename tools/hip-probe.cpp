// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// hip_probe — enumerate all HIP-visible devices, instantiate HipContext
// on the default pick, print the portable ComputeBackend snapshot plus the
// HIP-native superset, and evaluate every BackendFeature flag. This is
// the "toolchain is live end-to-end" smoke signal for the HIP bringup,
// analogous to l0_ipc_testrig for the L0 side. Not installed in any
// runtime image.
//
// Exit codes:
//   0  success — at least one device found, HipContext constructed OK
//   1  no HIP devices visible
//   2  HipContext construction threw (see stderr)
//
// Run via:
//   cmake --build build --target hip_probe && ./build/hip_probe

#include "core/backend/ComputeBackend.hpp"
#include "core/gpu/hip/HipContext.hpp"

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <string_view>

namespace {

std::string_view backendFeatureName(mimirmind::core::backend::BackendFeature f) {
    using F = mimirmind::core::backend::BackendFeature;
    switch (f) {
        case F::MutableCommandLists: return "MutableCommandLists";
        case F::IntegerDotProduct:   return "IntegerDotProduct";
        case F::IpcHandleExport:     return "IpcHandleExport";
        case F::UnifiedMemoryHost:   return "UnifiedMemoryHost";
        case F::MatrixEngine:        return "MatrixEngine";
    }
    return "Unknown";
}

std::string_view deviceKindName(mimirmind::core::backend::DeviceKind k) {
    using K = mimirmind::core::backend::DeviceKind;
    switch (k) {
        case K::GpuIntegrated: return "GpuIntegrated";
        case K::GpuDiscrete:   return "GpuDiscrete";
        case K::Npu:           return "Npu";
        case K::Cpu:           return "Cpu";
        case K::Unknown:       return "Unknown";
    }
    return "Unknown";
}

} // namespace

int main() {
    int count = 0;
    hipError_t rc = hipGetDeviceCount(&count);
    if (rc != hipSuccess) {
        std::fprintf(stderr, "hipGetDeviceCount failed: %s\n", hipGetErrorString(rc));
        return 1;
    }
    std::printf("HIP devices visible: %d\n", count);
    if (count == 0) {
        std::fprintf(stderr, "no HIP devices — aborting\n");
        return 1;
    }
    for (int i = 0; i < count; ++i) {
        hipDeviceProp_t p{};
        if (hipGetDeviceProperties(&p, i) != hipSuccess) {
            std::printf("  [%d] <unavailable>\n", i);
            continue;
        }
        std::printf("  [%d] %-32s arch=%-10s CUs=%-3d VRAM=%.1f GB warp=%d integrated=%d\n",
                    i, p.name, p.gcnArchName, p.multiProcessorCount,
                    static_cast<double>(p.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0),
                    p.warpSize, p.integrated);
    }

    std::printf("\n=== HipContext (auto-select default device) ===\n");

    try {
        mimirmind::core::hip::HipContext ctx{};

        const auto& hipInfo = ctx.hipDeviceInfo();
        std::printf("HIP-native info:\n");
        std::printf("  index               : %d\n", hipInfo.hipDeviceIndex);
        std::printf("  name                : %s\n", hipInfo.name.c_str());
        std::printf("  gfxArch             : %s\n", hipInfo.gfxArch.c_str());
        std::printf("  vendorId            : 0x%04x\n", hipInfo.vendorId);
        std::printf("  deviceId            : 0x%04x\n", hipInfo.deviceId);
        std::printf("  numComputeUnits     : %d\n", hipInfo.numComputeUnits);
        std::printf("  totalVram (GB)      : %.2f\n",
                    static_cast<double>(hipInfo.totalVram) / (1024.0 * 1024.0 * 1024.0));
        std::printf("  warpSize            : %d\n", hipInfo.warpSize);
        std::printf("  coreClockRate (MHz) : %d\n", hipInfo.coreClockRateKhz / 1000);
        std::printf("  isIntegrated        : %s\n", hipInfo.isIntegrated ? "yes" : "no");

        std::printf("\nPortable BackendDeviceInfo (via ComputeBackend&):\n");
        const auto& info = ctx.deviceInfo();
        std::printf("  name                : %s\n", info.name.c_str());
        std::printf("  DeviceKind          : %s\n", std::string(deviceKindName(info.kind)).c_str());
        std::printf("  vendorId            : 0x%04x\n", info.vendorId);
        std::printf("  numComputeUnits     : %u\n", info.numComputeUnits);
        std::printf("  totalLocalMem (GB)  : %.2f\n",
                    static_cast<double>(info.totalLocalMem) / (1024.0 * 1024.0 * 1024.0));
        std::printf("  coreClockRate (MHz) : %u\n", info.coreClockRate);

        std::printf("\nBackendFeature flags:\n");
        using F = mimirmind::core::backend::BackendFeature;
        for (auto f : { F::MutableCommandLists,
                        F::IntegerDotProduct,
                        F::IpcHandleExport,
                        F::UnifiedMemoryHost,
                        F::MatrixEngine }) {
            std::printf("  %-24s : %s\n",
                        std::string(backendFeatureName(f)).c_str(),
                        ctx.hasFeature(f) ? "yes" : "no");
        }

        std::printf("\nhip_probe: OK\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "HipContext failed: %s\n", e.what());
        return 2;
    }
}