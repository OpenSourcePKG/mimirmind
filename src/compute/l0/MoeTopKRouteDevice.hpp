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
 * Standalone launcher for the M-Q3N.5 device-side MoE top-K router
 * (`kernels/moe_topk.cl` -> moe_topk.spv). Level Zero peer of
 * `compute::cuda::MoeTopKRouteDevice` / `compute::hip::HipMoeTopKRouteDevice`.
 * Loads the `moe_topk` SPV module + kernel once and appends one launch per
 * call to the context command queue.
 *
 * The L0 `GpuOps` owns an instance and forwards its `moeTopKRouteDeviceAsync`
 * override here. Kept a separate class (not inline in GpuOps) so it can land
 * without touching the shared L0 GpuOps translation unit while another
 * session edits it.
 *
 * This is the primary path for the 100 tok/s NUC target (Meteor Lake,
 * Xe-LPG): it replaces the host `compute::moeTopKRoute` + host->USM copy
 * (`Qwen35MoeBackend.cpp:516,579`), the per-layer host round trip that keeps
 * decode launch-bound — the precondition for Command-List-Replay capture.
 *
 * Immovable (holds a non-movable GpuModule); a direct member / local only.
 * Not thread-safe (L0 argument binding lives on the kernel handle).
 */
class MoeTopKRouteDevice {
public:
    // Mirror the kernel's compile-time ceilings (moe_topk.cl). Dispatch of a
    // larger routing table must bump both sides in lockstep.
    static constexpr std::size_t kMaxExperts = 256;
    static constexpr std::size_t kMaxK       = 16;

    explicit MoeTopKRouteDevice(core::l0::L0ComputeContext& ctx);

    /**
     * Append an async launch to the context queue (flush()ed by the caller /
     * the surrounding GpuOps sync). Every pointer is a device (USM) pointer.
     *
     *   logits    [T, nExperts]  F32   router scores (router-matmul output)
     *   outIdx    [T, K]         int32 expert indices, descending probability
     *   outWeight [T, K]         F32   renormalised weights, *wScale
     *
     * Throws std::runtime_error if nExperts/K exceed the kernel ceilings;
     * no-op if T/nExperts/K is zero.
     */
    void launch(const float*  logits,
                std::int32_t* outIdx,
                float*        outWeight,
                std::size_t   T,
                std::size_t   nExperts,
                std::size_t   K,
                float         wScale);

private:
    core::l0::L0ComputeContext& _ctx;
    runtime::GpuModule          _module;
    runtime::GpuKernel          _kernel;
};

} // namespace mimirmind::compute::l0