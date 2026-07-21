// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/gpu/cuda/CudaContext.hpp"

#include "core/log/Log.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <string>

namespace mimirmind::core::cuda {

namespace {

inline void cudaCheck(const std::string& call, cudaError_t code) {
    if (code != cudaSuccess) {
        throw CudaError(call, code);
    }
}

[[nodiscard]] ::mimirmind::core::backend::DeviceKind classifyKind(bool integrated) {
    using ::mimirmind::core::backend::DeviceKind;
    return integrated ? DeviceKind::GpuIntegrated : DeviceKind::GpuDiscrete;
}

// Populate the portable and the CUDA-native info blocks from
// cudaDeviceProp. Kept as a free function so the ctor stays short.
void populate(const cudaDeviceProp& p,
              int index,
              DeviceInfo& info,
              ::mimirmind::core::backend::BackendDeviceInfo& backendInfo)
{
    info.name             = p.name;
    info.computeMajor     = p.major;
    info.computeMinor     = p.minor;
    // cudaDeviceProp doesn't expose PCI vendor/device ids the same way
    // hipDeviceProp_t does; NVIDIA vendor is always 0x10DE, deviceId
    // left 0 unless we later read /sys.
    info.vendorId         = 0x10DEu;
    info.deviceId         = 0u;
    info.cudaDeviceIndex  = index;
    info.numSms           = p.multiProcessorCount;
    info.totalVram        = static_cast<std::size_t>(p.totalGlobalMem);
    info.maxAllocSize     = static_cast<std::size_t>(p.totalGlobalMem);
    info.warpSize         = p.warpSize;
    int clockKhz = 0;
    cudaDeviceGetAttribute(&clockKhz, cudaDevAttrClockRate, index);
    info.coreClockRateKhz = clockKhz;
    info.isIntegrated     = p.integrated != 0;

    backendInfo.name             = info.name;
    backendInfo.uuid             = "";  // p.uuid is available but 16-byte struct; skip in skeleton
    backendInfo.vendorId         = info.vendorId;
    backendInfo.deviceId         = info.deviceId;
    backendInfo.kind             = classifyKind(info.isIntegrated);
    backendInfo.numComputeUnits  = static_cast<std::uint32_t>(info.numSms);
    backendInfo.totalLocalMem    = info.totalVram;
    backendInfo.maxMemAllocSize  = info.maxAllocSize;
    backendInfo.coreClockRate    = static_cast<std::uint32_t>(info.coreClockRateKhz / 1000);
}

// Pick a device: prefer first discrete GPU, fall back to first
// integrated. Returns -1 if CUDA reports zero devices. On DGX Spark
// there is only one device (integrated), so the fallback fires.
[[nodiscard]] int selectDefaultDevice(int count) {
    int fallback = -1;
    for (int i = 0; i < count; ++i) {
        cudaDeviceProp p{};
        if (cudaGetDeviceProperties(&p, i) != cudaSuccess) continue;
        if (p.integrated == 0) {
            return i;                    // first dGPU wins
        }
        if (fallback < 0) fallback = i;  // remember first iGPU as backup
    }
    return fallback;
}

} // namespace

// ---------- CudaError ------------------------------------------------------

CudaError::CudaError(const std::string& call, cudaError_t code)
    : std::runtime_error(call + ": " + cudaGetErrorString(code)),
      _code(code) {}

// ---------- CudaContext ----------------------------------------------------

CudaContext::CudaContext(int deviceIndex) {
    int count = 0;
    cudaCheck("cudaGetDeviceCount", cudaGetDeviceCount(&count));
    if (count <= 0) {
        throw CudaError("cudaGetDeviceCount", cudaErrorNoDevice);
    }

    if (deviceIndex < 0) {
        deviceIndex = selectDefaultDevice(count);
    }
    if (deviceIndex < 0 || deviceIndex >= count) {
        throw CudaError("CudaContext(deviceIndex out of range)", cudaErrorInvalidDevice);
    }

    cudaCheck("cudaSetDevice", cudaSetDevice(deviceIndex));

    cudaDeviceProp p{};
    cudaCheck("cudaGetDeviceProperties", cudaGetDeviceProperties(&p, deviceIndex));

    _deviceIdx = deviceIndex;
    populate(p, deviceIndex, _info, _backendInfo);

    MM_LOG_INFO("CudaContext",
                "bound to device #{} ({}, sm_{}{}, SMs={}, warp={}, integrated={})",
                _deviceIdx, _info.name, _info.computeMajor, _info.computeMinor,
                _info.numSms, _info.warpSize, _info.isIntegrated);
}

CudaContext::~CudaContext() = default;

::mimirmind::core::backend::BackendKind CudaContext::kind() const noexcept {
    return ::mimirmind::core::backend::BackendKind::Cuda;
}

const ::mimirmind::core::backend::BackendDeviceInfo& CudaContext::deviceInfo() const noexcept {
    return _backendInfo;
}

bool CudaContext::hasFeature(::mimirmind::core::backend::BackendFeature f) const noexcept {
    using ::mimirmind::core::backend::BackendFeature;
    const int major = _info.computeMajor;

    switch (f) {
    case BackendFeature::MutableCommandLists:
        // cudaGraph exists on all supported CUDA 10+ versions.
        return true;

    case BackendFeature::IntegerDotProduct:
        // __dp4a intrinsic since Pascal sm_61. Practically all
        // ROCm/CUDA HW mimirmind targets meets sm_60 already.
        return major >= 6;

    case BackendFeature::IpcHandleExport:
        // cudaIpcGetMemHandle is only supported on discrete cards.
        // Integrated GPUs (Jetson/Tegra/DGX Spark) explicitly do
        // NOT support IPC handles — cudaIpcGetMemHandle returns
        // cudaErrorNotSupported. Honest signal.
        return !_info.isIntegrated;

    case BackendFeature::UnifiedMemoryHost:
        // Integrated NVIDIA parts (Jetson, DGX Spark GB10) share a
        // single LPDDR5x pool with the CPU — cudaMallocManaged /
        // cudaMallocHost yields zero-copy. Discrete cards still
        // need PCIe copies.
        return _info.isIntegrated;

    case BackendFeature::MatrixEngine:
        // Tensor Cores landed on Volta sm_70. Every consumer/Ampere/
        // Ada/Hopper/Blackwell part has them. Not enabling on Pascal
        // and older.
        return major >= 7;
    }
    return false;
}

// Note: `probeBackend()` for CUDA lives in CudaProbe.cpp (Track 1
// shipped it there). This TU deliberately does not redefine it — a
// duplicate symbol would fail at link time. Same for
// `createComputeContext()` (Track 2 updates CudaProbe.cpp to return a
// real CudaComputeContext instance instead of throwing).

} // namespace mimirmind::core::cuda