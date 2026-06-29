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

    /// Gemma-family variant: y = x * (1 + weight) / sqrt(mean(x^2) + eps).
    /// Used for all proper norm weights in Gemma 2/3/4 (init at 0,
    /// (1+w) keeps the norm starting as identity). The non-Gemma
    /// rmsNormAsync stays in use for Qwen-family and for any
    /// multiplicative-scale step that doesn't follow the Gemma init.
    void rmsNormGemmaAsync(const float* x,
                           std::size_t  M,
                           std::size_t  K,
                           const float* weight,
                           float        eps,
                           float*       y);

    /// Bare RMS-normalize without a learned scale: y = x / sqrt(mean(x^2) + eps).
    /// Used by Gemma 4 for the V projection (V passes through ggml_rms_norm
    /// before going into the KV cache, with no per-element weight).
    void rmsNormNoWeightAsync(const float* x,
                              std::size_t  M,
                              std::size_t  K,
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

    /// In-place RoPE on a [seqLen, numHeads, headDim] f32 buffer. The
    /// per-position angle uses `startPos` as the absolute offset of
    /// row 0 — pass cache.length() in decode mode.
    void ropeInPlaceAsync(float*       x,
                          std::size_t  seqLen,
                          std::size_t  numHeads,
                          std::size_t  headDim,
                          std::size_t  startPos,
                          float        base);

    /// In-place RoPE with per-pair frequency factors (ggml_rope_ext's
    /// `freq_factors` argument). `freqFactors` points at [headDim/2] f32
    /// values; the rotation angle becomes
    ///   theta_i = pos * base^(-2i/headDim) / freqFactors[i]
    /// Used by Gemma 3/4 global-attention layers for proportional RoPE.
    void ropeInPlaceWithFactorsAsync(float*       x,
                                     const float* freqFactors,
                                     std::size_t  seqLen,
                                     std::size_t  numHeads,
                                     std::size_t  headDim,
                                     std::size_t  startPos,
                                     float        base);

    /// In-place scalar multiply: y[i] *= s for i in [0, n).
    /// Used by Gemma 4 for layer_output_scale.
    void mulScalarAsync(float*       y,
                        float        s,
                        std::size_t  n);

    /// GELU-flavoured SwiGLU: gate[i] = gelu_tanh(gate[i]) * up[i].
    /// Used by Gemma 4's FFN paths (vs Qwen's siluMulAsync).
    void geluMulAsync(float*       gate,
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

    runtime::GpuModule     _ropeModule;
    runtime::GpuKernel     _ropeKernel;

    runtime::GpuModule     _mulScalarModule;
    runtime::GpuKernel     _mulScalarKernel;

    runtime::GpuModule     _geluMulModule;
    runtime::GpuKernel     _geluMulKernel;

    runtime::GpuModule     _rmsnormGemmaModule;
    runtime::GpuKernel     _rmsnormGemmaKernel;

    runtime::GpuModule     _rmsnormNoWeightModule;
    runtime::GpuKernel     _rmsnormNoWeightKernel;

    runtime::GpuModule     _ropeFfModule;
    runtime::GpuKernel     _ropeFfKernel;

    static constexpr std::uint32_t kRmsnormLocalSize    = 128;
    static constexpr std::uint32_t kElementwiseLocalSize = 256;
    static constexpr std::uint32_t kRopeLocalSize        = 256;
};

} // namespace mimirmind::compute