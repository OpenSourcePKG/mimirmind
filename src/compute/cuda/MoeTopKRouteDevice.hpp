// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/gpu/cuda/CudaKernel.hpp"
#include "core/gpu/cuda/CudaModule.hpp"

#include <cstddef>
#include <cstdint>

namespace mimirmind::core::cuda {
class CudaComputeContext;
}

namespace mimirmind::compute::cuda {

/**
 * Standalone launcher for the M-Q3N.5 device-side MoE top-K router
 * (`kernels_cuda/moe_topk.cu`). Loads the `moe_topk` module + function once
 * and drives one async launch per call.
 *
 * `GpuOps` owns an instance and forwards its `moeTopKRouteDeviceAsync`
 * override here, so all CUDA-specific launch detail lives in this file
 * rather than bloating `GpuOps`. Kept a separate class (not inline in
 * GpuOps) so it can land without touching the shared GpuOps/GpuMatmul
 * translation units while another session is editing them.
 *
 * Replaces the host `compute::moeTopKRoute` + the host->USM copy loop that
 * forces a D2H/host/H2D round trip per MoE layer
 * (`Qwen35MoeBackend.cpp:516,579`) — the ~96 ms/tok host_sync wall that
 * keeps decode launch-bound. See milestone M-Q3N.5.
 *
 * Move-only via its members; single-threaded by contract (the kernel's arg
 * buffer is shared state, same as every other CudaKernel consumer).
 */
class MoeTopKRouteDevice {
public:
    // Mirror the kernel's compile-time ceilings (moe_topk.cu). Dispatch of a
    // larger routing table must bump both sides in lockstep.
    static constexpr std::size_t kMaxExperts = 256;
    static constexpr std::size_t kMaxK       = 16;

    explicit MoeTopKRouteDevice(core::cuda::CudaComputeContext& ctx);

    /**
     * Async launch on the context stream. Every pointer is a device (USM)
     * pointer; nothing is read back to the host — that is the whole point.
     *
     *   logits    [T, nExperts]  F32   router scores (router-matmul output)
     *   outIdx    [T, K]         int32 expert indices, descending probability
     *   outWeight [T, K]         F32   renormalised weights, pre-multiplied
     *                                  by `wScale` (== the kwSlot layout the
     *                                  fused-K down kernel consumes)
     *
     * `outIdx`/`outWeight` use the exact contiguous [T, K] layout of
     * `expIdxSlot`/`kwSlot`, so the fused-K consumers need no change.
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
    core::cuda::CudaComputeContext& _ctx;
    core::cuda::CudaModule          _module;
    core::cuda::CudaKernel          _kernel;
};

} // namespace mimirmind::compute::cuda