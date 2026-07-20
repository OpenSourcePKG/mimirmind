// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/backend/ComputeBackend.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace mimirmind::core::cuda {

/**
 * CUDA-native device snapshot. Superset of `backend::BackendDeviceInfo`
 * — carries CUDA-typed fields (compute capability, SM count, integrated
 * flag) that only the CUDA backend cares about. Consumers that only need
 * the portable subset should use `ComputeBackend::deviceInfo()`.
 */
struct DeviceInfo {
    std::string   name;              // vendor-supplied device name
    int           computeMajor{0};   // sm_major (e.g. 12 on GB10 Blackwell client)
    int           computeMinor{0};   // sm_minor (e.g. 1 on GB10)
    std::uint32_t vendorId{0x10DE};  // PCI vendor id — always NVIDIA
    std::uint32_t deviceId{0};       // PCI device id
    int           cudaDeviceIndex{-1};// index for cudaSetDevice
    int           numSms{0};         // multiProcessorCount
    std::size_t   totalVram{0};      // bytes; on dGPU = dedicated VRAM, on Grace-ARM = shared LPDDR5x
    std::size_t   maxAllocSize{0};   // largest single allocation the driver will honour
    int           warpSize{32};      // 32 on every NVIDIA arch since Kepler
    int           coreClockRateKhz{0};
    bool          isIntegrated{false};// Jetson/Tegra/DGX Spark = true, discrete card = false
};

class CudaError : public std::runtime_error {
public:
    CudaError(const std::string& call, cudaError_t code);
    [[nodiscard]] cudaError_t code() const noexcept { return _code; }

private:
    cudaError_t _code;
};

/**
 * RAII wrapper around a CUDA device selection, scoped to one NVIDIA GPU.
 * Multi-device isolation is out of scope for the initial CUDA skeleton
 * — the first port aims for bit-parity against the Level Zero / HIP
 * reference on a single device.
 *
 * Implements `core::backend::ComputeBackend` — the neutral interface.
 * Consumers that only need device-info + feature-flags should take
 * `ComputeBackend&`; consumers that touch CUDA-native handles (streams,
 * events, modules, memory) go through the concrete `CudaContext&` type.
 *
 * Not thread-safe by contract; wrap in a mutex at the caller level if
 * you need shared access.
 */
class CudaContext : public ::mimirmind::core::backend::ComputeBackend {
public:
    /// `deviceIndex = -1` means auto-select: prefer the first
    /// non-integrated (discrete) GPU, fall back to the integrated one
    /// if no dGPU is present. On DGX Spark there is exactly one device
    /// and it reports integrated=true (Grace ARM + GB10 UMA).
    explicit CudaContext(int deviceIndex = -1);
    ~CudaContext() override;

    CudaContext(const CudaContext&)            = delete;
    CudaContext& operator=(const CudaContext&) = delete;
    CudaContext(CudaContext&&)                 = delete;
    CudaContext& operator=(CudaContext&&)      = delete;

    // ---- ComputeBackend interface --------------------------------------

    [[nodiscard]] ::mimirmind::core::backend::BackendKind kind() const noexcept override;
    [[nodiscard]] const ::mimirmind::core::backend::BackendDeviceInfo& deviceInfo() const noexcept override;
    [[nodiscard]] bool hasFeature(::mimirmind::core::backend::BackendFeature f) const noexcept override;

    // ---- CUDA-native accessors -----------------------------------------

    [[nodiscard]] int cudaDeviceIndex() const noexcept { return _deviceIdx; }
    [[nodiscard]] const DeviceInfo& cudaDeviceInfo() const noexcept { return _info; }

private:
    int _deviceIdx{-1};
    DeviceInfo _info{};
    ::mimirmind::core::backend::BackendDeviceInfo _backendInfo{};
};

} // namespace mimirmind::core::cuda