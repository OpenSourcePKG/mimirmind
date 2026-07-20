// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/gpu/cuda/CudaComputeContext.hpp"

namespace mimirmind::core::cuda {

CudaComputeContext::CudaComputeContext(Options opts)
    : _ctx{opts.deviceIndex}
    , _alloc{_ctx}
    , _stream{_ctx, opts.streamKind}
{}

CudaComputeContext::~CudaComputeContext() = default;

std::size_t CudaComputeContext::bandwidthGBps() const noexcept {
    // Two-branch heuristic — integrated Grace/Jetson vs discrete
    // consumer/datacenter dGPU. DGX Spark (Bragi target) is
    // integrated → 273 GB/s. A precise per-device probe is future
    // work; for BatchCapacityProbe this coarse estimate is enough
    // to gate serving-class (M-Startup.CapacityProbe).
    using ::mimirmind::core::backend::DeviceKind;
    switch (_ctx.deviceInfo().kind) {
        case DeviceKind::GpuIntegrated:  return 273;   // Grace + GB10 / Jetson AGX Orin class
        case DeviceKind::GpuDiscrete:    return 500;   // consumer Blackwell/Ada average
        case DeviceKind::Npu:            return 30;
        case DeviceKind::Cpu:            return 50;
        case DeviceKind::Unknown:        return 0;
    }
    return 0;
}

} // namespace mimirmind::core::cuda