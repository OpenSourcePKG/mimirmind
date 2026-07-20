// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/gpu/hip/HipComputeContext.hpp"

#include <memory>

namespace mimirmind::core::hip {

HipComputeContext::HipComputeContext(Options opts)
    : _ctx{opts.deviceIndex}
    , _alloc{_ctx}
    , _stream{_ctx, opts.streamKind}
{}

HipComputeContext::~HipComputeContext() = default;

std::size_t HipComputeContext::bandwidthGBps() const noexcept {
    // Two-branch heuristic — integrated APU vs discrete dGPU.
    // A precise per-device probe is future work; for
    // BatchCapacityProbe (M-Startup.CapacityProbe) this coarse
    // estimate is enough to gate serving-class in Bragi.
    using ::mimirmind::core::backend::DeviceKind;
    switch (_ctx.deviceInfo().kind) {
        case DeviceKind::GpuIntegrated:  return 100;   // Phoenix / Hawk Point / Strix Halo LPDDR5x
        case DeviceKind::GpuDiscrete:    return 500;   // RDNA3 average (gfx1101=624, gfx1030=512)
        case DeviceKind::Npu:            return 30;
        case DeviceKind::Cpu:            return 50;
        case DeviceKind::Unknown:        return 0;
    }
    return 0;
}

// Factory hook consumed by BackendRegistry::createContext(). Lives in
// this translation unit so the common BackendRegistry TU never pulls
// in <hip/hip_runtime.h>. Uses default Options: auto-select device,
// BlockingDefault stream.
std::unique_ptr<::mimirmind::core::backend::ComputeContext>
createComputeContext() {
    return std::make_unique<HipComputeContext>();
}

} // namespace mimirmind::core::hip