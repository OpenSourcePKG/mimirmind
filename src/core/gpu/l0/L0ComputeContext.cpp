// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/gpu/l0/L0ComputeContext.hpp"

#include <memory>

namespace mimirmind::core::l0 {

L0ComputeContext::L0ComputeContext(Options opts)
    : _ctx{std::move(opts.spvDirOverride)}
    , _alloc{_ctx,
             opts.usmProbeTotalGiB,
             opts.usmKindOverride.value_or(selectUsmAllocKind(_ctx))}
    , _queue{_ctx}
{}

L0ComputeContext::~L0ComputeContext() = default;

std::size_t L0ComputeContext::bandwidthGBps() const noexcept {
    // Two-branch heuristic — integrated iGPU vs discrete dGPU.
    // A precise per-device measurement lands in M-Probe.1; for the
    // BatchCapacityProbe consumer (M-Startup.CapacityProbe) this
    // coarse estimate is enough to decide serving-class gating.
    using ::mimirmind::core::backend::DeviceKind;
    switch (_ctx.deviceInfo().kind) {
        case DeviceKind::GpuIntegrated:  return 70;    // Meteor/Arrow/Lunar/Panther Lake Xe-LPG
        case DeviceKind::GpuDiscrete:    return 450;   // Arc B70 / A770 / Battlemage class
        case DeviceKind::Npu:            return 30;    // VPU 3720 conservative
        case DeviceKind::Cpu:            return 50;    // unlikely under L0 but safe
        case DeviceKind::Unknown:        return 0;
    }
    return 0;
}

// Factory hook consumed by BackendRegistry::createContext(). Lives in
// this translation unit so the common BackendRegistry TU never pulls
// in level_zero/ze_api.h. Options are the defaults — the entry point
// deliberately does not expose them because runtime backend-selection
// happens before config.json is fully parsed; the tuned Options struct
// gets passed by callers that build an L0ComputeContext directly.
std::unique_ptr<::mimirmind::core::backend::ComputeContext>
createComputeContext() {
    return std::make_unique<L0ComputeContext>();
}

} // namespace mimirmind::core::l0