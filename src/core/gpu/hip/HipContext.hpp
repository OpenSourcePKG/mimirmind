// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/gpu/GpuBackend.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace mimirmind::core::hip {

/**
 * HIP-native device snapshot. Superset of `gpu::BackendDeviceInfo` —
 * carries HIP-typed fields (gfx arch string, warp size, integrated
 * flag) that only the HIP backend cares about. Consumers that only
 * need the portable subset should use `GpuBackend::deviceInfo()`.
 */
struct DeviceInfo {
    std::string   name;              // vendor-supplied device name
    std::string   gfxArch;           // "gfx1101" for Navi 32, "gfx90a" for MI250, ...
    std::uint32_t vendorId{0};       // PCI vendor id (0x1002 for AMD)
    std::uint32_t deviceId{0};       // PCI device id (0x747e for Navi 32)
    int           hipDeviceIndex{-1};// index for hipSetDevice
    int           numComputeUnits{0};// CUs on RDNA/CDNA
    std::size_t   totalVram{0};      // bytes; on dGPU = dedicated VRAM, on iGPU = shared RAM
    std::size_t   maxAllocSize{0};   // largest single allocation the driver will honour
    int           warpSize{0};       // 32 on RDNA/RDNA2/RDNA3, 64 on CDNA/GCN
    int           coreClockRateKhz{0};
    bool          isIntegrated{false};
};

class HipError : public std::runtime_error {
public:
    HipError(const std::string& call, hipError_t code);
    [[nodiscard]] hipError_t code() const noexcept { return _code; }

private:
    hipError_t _code;
};

/**
 * RAII wrapper around a HIP device selection, scoped to one AMD GPU
 * (dGPU if present, iGPU otherwise). Multi-device isolation is out of
 * scope for the initial HIP skeleton — the first port aims for
 * bit-parity against the Level Zero reference on a single device,
 * then adds multi-device once the kernel-set has landed.
 *
 * Implements `core::gpu::GpuBackend` — the backend-neutral interface.
 * Consumers that only need device-info + feature-flags should take
 * `GpuBackend&`; consumers that touch HIP-native handles (streams,
 * events, modules, USM) go through the concrete `HipContext&` type
 * once those wrappers exist. Today the skeleton exposes the device
 * index only — no stream, no memory manager, no ops. Kernels come
 * on the same branch as follow-up commits.
 *
 * Not thread-safe by contract; wrap in a mutex at the caller level if
 * you need shared access.
 */
class HipContext : public ::mimirmind::core::gpu::GpuBackend {
public:
    /// `deviceIndex = -1` means auto-select: prefer the first
    /// non-integrated (discrete) GPU, fall back to the integrated one
    /// if no dGPU is present. Explicit index is honoured verbatim.
    explicit HipContext(int deviceIndex = -1);
    ~HipContext() override;

    HipContext(const HipContext&)            = delete;
    HipContext& operator=(const HipContext&) = delete;
    HipContext(HipContext&&)                 = delete;
    HipContext& operator=(HipContext&&)      = delete;

    // ---- GpuBackend interface ------------------------------------------

    [[nodiscard]] ::mimirmind::core::gpu::BackendKind kind() const noexcept override;
    [[nodiscard]] const ::mimirmind::core::gpu::BackendDeviceInfo& deviceInfo() const noexcept override;
    [[nodiscard]] bool hasFeature(::mimirmind::core::gpu::BackendFeature f) const noexcept override;

    // ---- HIP-native accessors ------------------------------------------

    [[nodiscard]] int hipDeviceIndex() const noexcept { return _deviceIdx; }
    [[nodiscard]] const DeviceInfo& hipDeviceInfo() const noexcept { return _info; }

private:
    int _deviceIdx{-1};
    DeviceInfo _info{};
    ::mimirmind::core::gpu::BackendDeviceInfo _backendInfo{};
};

} // namespace mimirmind::core::hip