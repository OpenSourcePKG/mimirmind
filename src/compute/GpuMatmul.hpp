#pragma once

#include "model/GgufTypes.hpp"
#include "runtime/CommandQueue.hpp"
#include "runtime/GpuKernel.hpp"
#include "runtime/GpuModule.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>

namespace mimirmind::runtime {
class L0Context;
}

namespace mimirmind::compute {

/**
 * Drop-in replacement for compute::matmul that dispatches to GPU kernels
 * for weight types whose QuantType advertises a `gpuMatmulModule()`. For
 * unsupported types it falls back to the scalar CPU compute::matmul.
 *
 * Owns the SPIR-V modules + kernels. The command queue is shared with
 * the rest of the engine (passed in by reference) so element-wise GPU
 * ops can be appended into the same command list as matmuls. Not
 * thread-safe (the underlying ze_kernel_handle_t is mutated by
 * setArgumentValue). Construct once at startup, share across the engine.
 */
class GpuMatmul {
public:
    GpuMatmul(runtime::L0Context& ctx, runtime::CommandQueue& queue);
    ~GpuMatmul() = default;

    GpuMatmul(const GpuMatmul&)            = delete;
    GpuMatmul& operator=(const GpuMatmul&) = delete;
    GpuMatmul(GpuMatmul&&)                 = delete;
    GpuMatmul& operator=(GpuMatmul&&)      = delete;

    /// True if this dispatcher will run `type` on the GPU.
    [[nodiscard]] bool supports(model::GgmlType type) const noexcept;

    /// Y [M, N] = X [M, K] * W [N, K]^T. Synchronous version (mirrors
    /// compute::matmul signature). For supported `type` the dispatch goes
    /// to the GPU kernel and we sync immediately. For unsupported types
    /// the call falls back to CPU.
    ///
    /// `scratch` (K floats) is only consumed on the CPU fallback path.
    void matmul(model::GgmlType type,
                const void*     W,
                std::size_t     N,
                std::size_t     K,
                const float*    X,
                std::size_t     M,
                float*          Y,
                float*          scratch);

    /**
     * Same as matmul() but doesn't sync. The Y buffer is NOT yet valid
     * when this returns; call sync() before the CPU reads from it. Use
     * to batch independent matmul calls into a single Level-Zero
     * submission and save the per-dispatch sync cost.
     *
     * For unsupported types we transparently flush any pending GPU work
     * first (to preserve ordering vs prior async appends), then run the
     * CPU fallback synchronously.
     */
    void matmulAsync(model::GgmlType type,
                     const void*     W,
                     std::size_t     N,
                     std::size_t     K,
                     const float*    X,
                     std::size_t     M,
                     float*          Y,
                     float*          scratch);

    /// Flush any pending appends (close + execute + sync + reset). Safe
    /// to call when there's no pending work — cheap no-op.
    void sync();

private:
    struct Entry {
        std::unique_ptr<runtime::GpuModule> module;
        runtime::GpuKernel                  kernel;
    };

    runtime::L0Context&    _ctx;
    runtime::CommandQueue& _queue;

    // One Entry per GgmlType that has a `gpuMatmulModule()` registered.
    // Populated at construction by iterating the QuantType registry.
    std::unordered_map<model::GgmlType, Entry> _entries;

    // M5h: workgroup of 64 threads = 4 subgroups of 16, each subgroup
    // co-computes ONE output via sub_group_reduce_add. So a workgroup
    // emits 4 outputs and we need ceil(N/4) workgroups. Keep in sync
    // with the kernel macros `MATMUL_*_LOCAL` / `MATMUL_*_SG`.
    static constexpr std::uint32_t kLocalSize        = 64;
    static constexpr std::uint32_t kSubgroupSize     = 16;
    static constexpr std::uint32_t kOutputsPerGroup  = kLocalSize / kSubgroupSize;
};

} // namespace mimirmind::compute