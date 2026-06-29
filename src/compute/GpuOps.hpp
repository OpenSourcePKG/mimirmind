#pragma once

#include "runtime/CommandQueue.hpp"
#include "runtime/GpuKernel.hpp"
#include "runtime/GpuModule.hpp"

#include <cstddef>

namespace mimirmind::runtime {
class L0Context;
}

namespace mimirmind::compute {

/**
 * GPU element-wise + normalisation kernels.
 *
 * Shares the engine's CommandQueue with GpuMatmul so the entire block
 * forward can eventually be appended into one command list (M5f.4).
 * Each public method appends a kernel launch to the queue WITHOUT
 * syncing — call queue.flush() / GpuMatmul::sync() before reading
 * results on the CPU.
 *
 * Not thread-safe (the underlying ze_kernel_handle_t is mutated by
 * setArgumentValue). Construct once at startup, share across the engine.
 */
class GpuOps {
public:
    GpuOps(runtime::L0Context& ctx, runtime::CommandQueue& queue);
    ~GpuOps() = default;

    GpuOps(const GpuOps&)            = delete;
    GpuOps& operator=(const GpuOps&) = delete;
    GpuOps(GpuOps&&)                 = delete;
    GpuOps& operator=(GpuOps&&)      = delete;

    /// Per-row RMSNorm. y = x * weight / sqrt(mean(x^2) + eps)
    /// x: [M, K] f32, weight: [K] f32, y: [M, K] f32. M and K must be
    /// representable as int32_t for the kernel.
    void rmsNormAsync(const float* x,
                      std::size_t  M,
                      std::size_t  K,
                      const float* weight,
                      float        eps,
                      float*       y);

    /// In-place broadcast bias: y[m, k] += bias[k].
    void addBiasAsync(float*       y,
                      std::size_t  M,
                      std::size_t  K,
                      const float* bias);

    /// In-place residual: y[i] += x[i] for i in [0, n).
    void addResidualAsync(float*       y,
                          const float* x,
                          std::size_t  n);

    /// Fused SwiGLU step: gate[i] = silu(gate[i]) * up[i] for i in [0, n).
    void siluMulAsync(float*       gate,
                      const float* up,
                      std::size_t  n);

private:
    runtime::L0Context&    _ctx;
    runtime::CommandQueue& _queue;

    runtime::GpuModule     _rmsnormModule;
    runtime::GpuKernel     _rmsnormKernel;

    runtime::GpuModule     _addBiasModule;
    runtime::GpuKernel     _addBiasKernel;

    runtime::GpuModule     _addResidualModule;
    runtime::GpuKernel     _addResidualKernel;

    runtime::GpuModule     _siluMulModule;
    runtime::GpuKernel     _siluMulKernel;

    static constexpr std::uint32_t kRmsnormLocalSize    = 128;
    static constexpr std::uint32_t kElementwiseLocalSize = 256;
};

} // namespace mimirmind::compute