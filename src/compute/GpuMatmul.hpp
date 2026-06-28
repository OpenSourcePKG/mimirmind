#pragma once

#include "model/GgufTypes.hpp"
#include "runtime/CommandQueue.hpp"
#include "runtime/GpuKernel.hpp"
#include "runtime/GpuModule.hpp"

#include <cstddef>

namespace mimirmind::runtime {
class L0Context;
}

namespace mimirmind::compute {

/**
 * Drop-in replacement for compute::matmul that dispatches to GPU kernels
 * for the weight types we have kernels for (Q4_K, Q6_K). For unsupported
 * types it falls back to the scalar CPU compute::matmul.
 *
 * Owns the SPIR-V modules + kernels + a single command queue. Not
 * thread-safe (the underlying ze_kernel_handle_t is mutated by
 * setArgumentValue). Construct once at startup, share across the engine.
 */
class GpuMatmul {
public:
    explicit GpuMatmul(runtime::L0Context& ctx);
    ~GpuMatmul() = default;

    GpuMatmul(const GpuMatmul&)            = delete;
    GpuMatmul& operator=(const GpuMatmul&) = delete;
    GpuMatmul(GpuMatmul&&)                 = delete;
    GpuMatmul& operator=(GpuMatmul&&)      = delete;

    /// True if this dispatcher will run `type` on the GPU.
    [[nodiscard]] bool supports(model::GgmlType type) const noexcept;

    /// Y [M, N] = X [M, K] * W [N, K]^T. Mirrors compute::matmul signature.
    /// For supported `type` and M==1 the dispatch goes straight to the GPU
    /// kernel; M>1 currently loops the matvec M times (one dispatch per
    /// row of X). For unsupported types the call falls back to CPU.
    ///
    /// `scratch` (K floats) is only consumed on the CPU fallback path;
    /// GPU path ignores it.
    void matmul(model::GgmlType type,
                const void*     W,
                std::size_t     N,
                std::size_t     K,
                const float*    X,
                std::size_t     M,
                float*          Y,
                float*          scratch);

private:
    runtime::L0Context&    _ctx;
    runtime::CommandQueue  _queue;
    runtime::GpuModule     _q4kModule;
    runtime::GpuKernel     _q4kKernel;
    runtime::GpuModule     _q6kModule;
    runtime::GpuKernel     _q6kKernel;

    static constexpr std::uint32_t kLocalSize = 64;
};

} // namespace mimirmind::compute