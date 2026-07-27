// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "compute/l0/MoeGateUpFusedKDevice.hpp"

#include "core/gpu/l0/L0ComputeContext.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace mimirmind::compute::l0 {

namespace {
// Must match reqd_work_group_size(MOE_GU_LOCAL) in
// kernels/moe_gateup_fused_k_q6k.cl, and the 4 outputs per workgroup
// (MOE_GU_LOCAL / MOE_GU_SG).
constexpr std::uint32_t kLocalSize        = 64;
constexpr std::uint32_t kOutputsPerGroup  = 4;
} // namespace

MoeGateUpFusedKDevice::MoeGateUpFusedKDevice(core::l0::L0ComputeContext& ctx)
    : _ctx{ctx},
      _module{ctx.l0Context(), "moe_gateup_fused_k_q6k"},
      _kernel{_module.kernel("moe_gateup_fused_k_q6k")} {}

void MoeGateUpFusedKDevice::launch(const float*        x,
                                   const void*         W,
                                   const std::int32_t* expIdx,
                                   const float*        downScale,
                                   float*              gateActOut,
                                   std::size_t         dModel,
                                   std::size_t         nFf,
                                   std::size_t         kActive,
                                   std::size_t         expertBytes) {
    if (kActive == 0 || dModel == 0 || nFf == 0) {
        return;
    }
    if (dModel > kMaxDModel) {
        throw std::runtime_error(
            "compute::l0::MoeGateUpFusedKDevice::launch: dModel=" +
            std::to_string(dModel) + " exceeds SLM ceiling " +
            std::to_string(kMaxDModel) +
            " — bump MOE_GU_XMAX in the kernel + kMaxDModel together");
    }
    // The Q6_K row dot assumes dModel is a whole number of 256-element
    // super-blocks; a ragged tail would silently drop weights.
    if (dModel % 256 != 0) {
        throw std::runtime_error(
            "compute::l0::MoeGateUpFusedKDevice::launch: dModel=" +
            std::to_string(dModel) + " is not a multiple of the Q6_K "
            "super-block (256)");
    }

    // Args match the moe_gateup_fused_k_q6k kernel signature exactly.
    _kernel.setPtr        (0, x);
    _kernel.setPtr        (1, W);
    _kernel.setPtr        (2, expIdx);
    _kernel.setPtr        (3, downScale);
    _kernel.setPtr        (4, gateActOut);
    _kernel.setValue<std::int32_t>(5, static_cast<std::int32_t>(dModel));
    _kernel.setValue<std::int32_t>(6, static_cast<std::int32_t>(nFf));
    _kernel.setValue<std::int32_t>(7, static_cast<std::int32_t>(kActive));
    _kernel.setValue<std::int32_t>(8, static_cast<std::int32_t>(expertBytes));
    _kernel.setGroupSize(kLocalSize, 1, 1);

    // 4 outputs per workgroup — mirrors moe_down_fused_k / matmul_q6k_vec.
    const std::uint32_t nGroups =
        static_cast<std::uint32_t>((nFf + kOutputsPerGroup - 1) /
                                   kOutputsPerGroup);
    _ctx.queue().appendLaunch(_kernel, nGroups, 1, 1);
}

} // namespace mimirmind::compute::l0
