// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "compute/l0/MoeTopKRouteDevice.hpp"

#include "core/gpu/l0/L0ComputeContext.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace mimirmind::compute::l0 {

namespace {
// Must match reqd_work_group_size(MOE_TOPK_LOCAL) in kernels/moe_topk.cl.
constexpr std::uint32_t kLocalSize = 32;
} // namespace

MoeTopKRouteDevice::MoeTopKRouteDevice(core::l0::L0ComputeContext& ctx)
    : _ctx{ctx},
      _module{ctx.l0Context(), "moe_topk"},
      _kernel{_module.kernel("moe_topk")} {}

void MoeTopKRouteDevice::launch(const float*  logits,
                                std::int32_t* outIdx,
                                float*        outWeight,
                                std::size_t   T,
                                std::size_t   nExperts,
                                std::size_t   K,
                                float         wScale) {
    if (T == 0 || nExperts == 0 || K == 0) {
        return;
    }
    if (nExperts > kMaxExperts || K > kMaxK) {
        throw std::runtime_error(
            "compute::l0::MoeTopKRouteDevice::launch: nExperts/K exceed the "
            "kernel ceilings (nExperts=" + std::to_string(nExperts) + " max " +
            std::to_string(kMaxExperts) + ", K=" + std::to_string(K) +
            " max " + std::to_string(kMaxK) + ") — bump both moe_topk.cl and "
            "kMaxExperts/kMaxK together");
    }

    // Args match the moe_topk kernel signature exactly.
    _kernel.setPtr  (0, logits);
    _kernel.setPtr  (1, outIdx);
    _kernel.setPtr  (2, outWeight);
    _kernel.setValue(3, static_cast<std::int32_t>(nExperts));
    _kernel.setValue(4, static_cast<std::int32_t>(K));
    _kernel.setValue(5, wScale);
    _kernel.setGroupSize(kLocalSize, 1, 1);

    // One work-group per token; only local id 0 computes in the v1 kernel.
    // appendLaunch is async — the surrounding GpuOps flush()/sync executes it.
    _ctx.queue().appendLaunch(_kernel,
                              static_cast<std::uint32_t>(T), 1, 1);
}

} // namespace mimirmind::compute::l0