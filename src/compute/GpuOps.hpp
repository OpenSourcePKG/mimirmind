#pragma once

#include "runtime/CommandQueue.hpp"
#include "runtime/GpuKernel.hpp"
#include "runtime/GpuModule.hpp"

#include <cstddef>
#include <string>
#include <string_view>

namespace mimirmind::runtime {
class L0Context;
class UsmAllocator;
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
    GpuOps(runtime::L0Context&    ctx,
           runtime::UsmAllocator& alloc,
           runtime::CommandQueue& queue);
    ~GpuOps();

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

    /// Fused scaled accumulate: dst[i] += scale * src[i]. Replaces a
    /// mulScalarAsync(src, scale) + addResidualAsync(dst, src) pair
    /// where the intermediate scaled src is not read elsewhere. Used by
    /// the Gemma 4 MoE per-expert loop.
    void scaledAddResidualAsync(float*       dst,
                                const float* src,
                                float        scale,
                                std::size_t  n);

    /// Scatter the output of a fused QKV matmul into the separate Q, K,
    /// V destinations. `fused` has shape [M, Nq + Nkv * (1 + hasV)];
    /// `Yq` [M, Nq]; `Yk` [M, Nkv]; `Yv` [M, Nkv] — the latter may be
    /// any valid pointer if `hasV == false` (never dereferenced).
    void qkvSplitAsync(const float* fused,
                       float*       Yq,
                       float*       Yk,
                       float*       Yv,
                       std::size_t  M,
                       std::size_t  Nq,
                       std::size_t  Nkv,
                       bool         hasV);

    /// Load-time self-test — run every GPU op with a known input and
    /// compare against a CPU reference within a tight tolerance. Catches
    /// broken SPV loads, driver miscompilation, and layout mismatches
    /// on unfamiliar iGPU µarchs before the first block runs. Throws
    /// std::runtime_error on any parity failure so loadModel() aborts
    /// with a clear error rather than silently corrupting inference.
    ///
    /// Currently exercises: qkv_split (full QKV + alt-attention paths).
    /// Runs in < 5 ms on any Intel iGPU. Idempotent, cheap to repeat.
    void selfTest(runtime::UsmAllocator& allocator);

    /// "pending" | "ok" — populated by selfTest(). Exposed via
    /// /v1/system/status so the deploy can be verified without pulling
    /// docker logs.
    [[nodiscard]] std::string_view selfTestStatus() const noexcept {
        return _selfTestStatus;
    }

    /// Multi-head GQA causal attention on the GPU. Layout-equivalent to
    /// compute::multiHeadAttention. q/k/v/out are all f32 USM:
    ///   q   [T_q, nHeads,    headDim]
    ///   k   [T_k, nKvHeads,  headDim]
    ///   v   [T_k, nKvHeads,  headDim]
    ///   out [T_q, nHeads,    headDim]
    /// scale is applied to Q·K before softmax (Qwen passes
    /// 1/sqrt(headDim); Gemma 4 passes 1.0 since it pre-scaled Q nowhere
    /// — see backend). Throws if T_k > kAttentionMaxTk (8192) or if
    /// nHeads is not a positive multiple of nKvHeads.
    void attentionAsync(const float* q,
                        const float* k,
                        const float* v,
                        std::size_t  T_q,
                        std::size_t  T_k,
                        std::size_t  nHeads,
                        std::size_t  nKvHeads,
                        std::size_t  headDim,
                        std::size_t  positionOffset,
                        float        scale,
                        float*       out);

    /// Compile-time bound on T_k matching ATTN_MAX_TK in attention.cl.
    /// Exposed so callers (and the engine config validator) can check
    /// max-context budgets up front. At 16384 the plain-attention SLM
    /// use (`scores[ATTN_MAX_TK] * 4 B`) sits at 64 KiB — the standard
    /// Intel Xe-LPG per-work-group SLM budget. Raising further requires
    /// the M9.8b architectural refactor (online-softmax plain attention
    /// or chunked-T_q flash prefill).
    static constexpr std::size_t kAttentionMaxTk = 16384;

    /// Accessor to the underlying command queue. Backends use this to
    /// construct UnorderedScope around clearly-parallel kernel groups
    /// (M5f.4). The queue is also shared with GpuMatmul, so wrapping
    /// matmul + GpuOps launches in the same scope works as expected.
    [[nodiscard]] runtime::CommandQueue& queue() noexcept { return _queue; }

private:
    runtime::L0Context&    _ctx;
    runtime::CommandQueue& _queue;
    runtime::UsmAllocator& _alloc;

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

    runtime::GpuModule     _attentionModule;
    runtime::GpuKernel     _attentionKernel;

    runtime::GpuModule     _attentionFlashPartialModule;
    runtime::GpuKernel     _attentionFlashPartialKernel;

    runtime::GpuModule     _attentionFlashMergeModule;
    runtime::GpuKernel     _attentionFlashMergeKernel;

    runtime::GpuModule     _scaledAddResidualModule;
    runtime::GpuKernel     _scaledAddResidualKernel;

    runtime::GpuModule     _qkvSplitModule;
    runtime::GpuKernel     _qkvSplitKernel;

    std::string            _selfTestStatus{"pending"};

    // Persistent USM scratch for the FlashAttention partial/merge
    // pipeline. Sized at construction for the worst case across the
    // models we run; reused across every decode call (the engine
    // serialises calls via engineMutex so no aliasing is possible).
    void*                  _flashPartialUsm{nullptr};
    std::size_t            _flashPartialBytes{0};

    static constexpr std::uint32_t kRmsnormLocalSize     = 128;
    static constexpr std::uint32_t kElementwiseLocalSize = 256;
    static constexpr std::uint32_t kRopeLocalSize        = 256;
    // Must match ATTN_LOCAL / ATTN_SG in kernels/attention.cl.
    static constexpr std::uint32_t kAttentionLocalSize   = 16;

    // Must match ATTN_FLASH_KTILE in kernels/attention_flash_partial.cl
    // and ATTN_FLASH_MAX_KTILES in kernels/attention_flash_merge.cl.
    static constexpr std::size_t kFlashKTileSize    = 256;
    static constexpr std::size_t kFlashMaxKTiles    = kAttentionMaxTk /
                                                      kFlashKTileSize;
    // Worst-case dims sized for Gemma 4 26B full-attention layers
    // (head_dim=512 on 5 of 30 layers; SWA layers are head_dim=256).
    // Bumping these only costs the static USM allocation size below.
    static constexpr std::size_t kFlashMaxHeads     = 64;
    static constexpr std::size_t kFlashMaxHeadDim   = 512;

    // Internal dispatch — variant (a) for T_q > 1, FlashAttention for
    // T_q == 1. Hidden behind attentionAsync.
    void attentionPlainAsync(const float* q,
                             const float* k,
                             const float* v,
                             std::size_t  T_q,
                             std::size_t  T_k,
                             std::size_t  nHeads,
                             std::size_t  nKvHeads,
                             std::size_t  headDim,
                             std::size_t  positionOffset,
                             float        scale,
                             float*       out);
    void attentionDecodeFlashAsync(const float* q,
                                   const float* k,
                                   const float* v,
                                   std::size_t  T_k,
                                   std::size_t  nHeads,
                                   std::size_t  nKvHeads,
                                   std::size_t  headDim,
                                   std::size_t  positionOffset,
                                   float        scale,
                                   float*       out);
};

} // namespace mimirmind::compute