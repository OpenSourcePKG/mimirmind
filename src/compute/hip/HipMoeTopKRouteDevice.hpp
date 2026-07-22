// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/gpu/hip/HipKernel.hpp"
#include "core/gpu/hip/HipModule.hpp"

#include <cstddef>
#include <cstdint>

namespace mimirmind::core::hip {
class HipComputeContext;
}

namespace mimirmind::compute::hip {

/**
 * Standalone launcher for the M-Q3N.5 device-side MoE top-K router
 * (`kernels_hip/moe_topk.hip`). HIP peer of
 * `compute::cuda::MoeTopKRouteDevice`. Loads the `moe_topk` module +
 * kernel once and drives one async launch per call.
 *
 * The HIP `GpuOps` owns an instance and forwards its
 * `moeTopKRouteDeviceAsync` override here, so all HIP-specific launch
 * detail lives in this file. Kept a separate class (not inline in GpuOps)
 * so it can land without touching the shared GpuOps translation unit while
 * another session edits it.
 *
 * Replaces the host `compute::moeTopKRoute` + the host->USM copy loop
 * (`Qwen35MoeBackend.cpp:516,579`) — the per-layer host round trip that
 * keeps decode launch-bound. Removing it is the precondition for HipGraph
 * capture on gfx1101 (see M-HipGraph shelf, unshelved by this milestone).
 *
 * Single-threaded by contract (the kernel's arg buffer is shared state).
 */
class HipMoeTopKRouteDevice {
public:
    // Mirror the kernel's compile-time ceilings (moe_topk.hip). Dispatch of
    // a larger routing table must bump both sides in lockstep.
    static constexpr std::size_t kMaxExperts = 256;
    static constexpr std::size_t kMaxK       = 16;

    explicit HipMoeTopKRouteDevice(core::hip::HipComputeContext& ctx);

    /**
     * Async launch on the context stream. Every pointer is a device (USM)
     * pointer; nothing is read back to the host.
     *
     *   logits    [T, nExperts]  F32   router scores (router-matmul output)
     *   outIdx    [T, K]         int32 expert indices, descending probability
     *   outWeight [T, K]         F32   renormalised weights, pre-multiplied
     *                                  by `wScale` (== the kwSlot layout)
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
    core::hip::HipComputeContext& _ctx;
    core::hip::HipModule          _module;
    core::hip::HipKernel          _kernel;
};

} // namespace mimirmind::compute::hip