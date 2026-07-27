// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/gpu/l0/GpuKernel.hpp"
#include "core/gpu/l0/GpuModule.hpp"

#include <cstddef>
#include <cstdint>

namespace mimirmind::core::l0 {
class L0ComputeContext;
}

namespace mimirmind::compute::l0 {

/**
 * Standalone launcher for the M-CLR.MoE Increment 2 device-indexed fused
 * gate+up projection (`kernels/moe_gateup_fused_k_q6k.cl` -> .spv). Level
 * Zero peer of the fused-K down kernel in GpuMatmul, but for the gate+up
 * half of the SwiGLU: it reads the router pick `expIdx[k]` on the device
 * so the decode MoE block has no host read of the routing between the
 * device top-K and the expert matmuls — the precondition for
 * Command-List-Replay capture of the MoE decode path (Meteor Lake /
 * Xe-LPG, the 100 tok/s NUC target).
 *
 * Kept a separate class (not inline in GpuMatmul) so it can land without
 * touching the shared GpuMatmul translation unit — the same rationale as
 * MoeTopKRouteDevice. The L0 GpuOps owns an instance and forwards its
 * moeGateUpFusedKGeluAsync override here.
 *
 * Only Q6_K expert gate_up banks are supported today (Gemma 4 26B-A4B);
 * other quant types fall back to the host per-k GEMV dispatch.
 *
 * Immovable (holds a non-movable GpuModule); a direct member / local only.
 * Not thread-safe (L0 argument binding lives on the kernel handle).
 */
class MoeGateUpFusedKDevice {
public:
    // Mirror MOE_GU_XMAX in the kernel — the SLM-resident X ceiling. A
    // dModel above this must fall back to the host path (the launcher
    // returns false from available()/throws so the caller can catch).
    static constexpr std::size_t kMaxDModel = 2560;

    explicit MoeGateUpFusedKDevice(core::l0::L0ComputeContext& ctx);

    /// True when this dModel fits the SLM-resident X ceiling. The Q6_K
    /// module itself is always loaded at construction.
    [[nodiscard]] bool supports(std::size_t dModel) const noexcept {
        return dModel != 0 && dModel <= kMaxDModel;
    }

    /**
     * Append an async launch to the context queue. Every pointer is a
     * device (USM) pointer.
     *
     *   x          [dModel]        F32   shared token input (path-B norm)
     *   W          Q6_K fused gate_up expert bank base
     *   expIdx     [K]             int32 device top-K picks
     *   downScale  [nExperts]      F32   ffn_down_exps.scale (folded in)
     *   gateActOut [K, nFf]        F32   fused gelu(gate)*up*downScale
     *
     * No-op if K/dModel/nFf is zero. Throws std::runtime_error if dModel
     * exceeds kMaxDModel (caller should gate on supports() first).
     */
    void launch(const float*        x,
                const void*         W,
                const std::int32_t* expIdx,
                const float*        downScale,
                float*              gateActOut,
                std::size_t         dModel,
                std::size_t         nFf,
                std::size_t         kActive,
                std::size_t         expertBytes);

private:
    core::l0::L0ComputeContext& _ctx;
    runtime::GpuModule          _module;
    runtime::GpuKernel          _kernel;
};

} // namespace mimirmind::compute::l0
