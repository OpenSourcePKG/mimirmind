// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "compute/ComputeOps.hpp"
#include "core/config/Config.hpp"
#include "runtime/KvCache.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace mimirmind::core::l0 {
class L0Context;
class L0ComputeContext;
class UsmAllocator;
}

namespace mimirmind::runtime {
class CommandQueue;
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
class GpuOps : public ComputeOps {
public:
    /// `flashPrefillEnabled` maps to `features.flashPrefill` in config.json.
    /// When false, T_q > 1 dispatches fall back to the plain attention.cl
    /// kernel — rollback lever for flash-prefill bugs.
    ///
    /// Takes `L0ComputeContext&` for the backend-neutralisation layer
    /// (Schicht 2). The concrete L0Context / UsmAllocator / CommandQueue
    /// references are pulled from it at ctor time and stored as members
    /// — the kernel-launch API is still L0-native. When compute:: gets
    /// its own neutral launch API (Schicht 4) this ctor will change
    /// again; until then the ComputeContext is the ownership marker.
    GpuOps(core::l0::L0ComputeContext& ctx,
           bool                        flashPrefillEnabled      = true,
           bool                        flashPrefillGqaQ8Enabled = true,
           std::size_t                 flashPrefillKTileQ8      = 128,
           core::config::TriState      q8_0ReorderMode          =
               core::config::TriState::Disable);
    ~GpuOps() override;

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
                      float*       y) override;

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
                           float*       y) override;

    /// Bare RMS-normalize without a learned scale: y = x / sqrt(mean(x^2) + eps).
    /// Used by Gemma 4 for the V projection (V passes through ggml_rms_norm
    /// before going into the KV cache, with no per-element weight).
    void rmsNormNoWeightAsync(const float* x,
                              std::size_t  M,
                              std::size_t  K,
                              float        eps,
                              float*       y) override;

    /// Fused Q + K + V RMSNorm — one dispatch instead of three. Q and K
    /// normalize with per-`head_dim` weights (plain rmsNorm semantics),
    /// V normalizes without a weight (matches `rmsNormNoWeightAsync`).
    /// Every buffer is `[rows, head_dim]` and every write is in-place-
    /// safe (dst == src OK). Row counts differ per buffer: qRows =
    /// `T * nHeads`, kRows = vRows = `T * nKvHeads`. Saves 2 dispatch
    /// launches per own-KV attention block (Gemma family).
    /// M-CLR.2: `kBase` / `vBase` point at the layer's K/V cache BASE
    /// pointer (not the per-token write slot). The kernel adds
    /// `writeOffset * kvDim` to reach the current write row. This keeps
    /// the pointer arguments stable across replays; `writeOffset` and
    /// `kvDim` fold into the shared USM `curLen` slot + a per-layer
    /// scalar. `writeOffset` = `cache.length()` at the caller.
    /// M10.2 Phase 0 Commit 4/5 — `kvDtype` selects between the f32
    /// fused kernel (default; K/V destinations = fp32 cache) and the
    /// fp16-KV variant. `kBase`/`vBase` are typed as `void*` so both
    /// fp32 and fp16 backing storage can flow through the same call
    /// without reinterpret_cast at every site — the kernel body uses
    /// `vload_half` / `vstore_half` when `kvDtype == FP16`. Q
    /// workspace stays fp32 in both paths.
    void rmsNormQkvAsync(float*           qBuf,   const float* qWeight,
                         void*            kBase,  const float* kWeight,
                         void*            vBase,
                         std::size_t      qRows,
                         std::size_t      kvRows,
                         std::size_t      headDim,
                         float            eps,
                         std::size_t      writeOffset,
                         std::size_t      kvDim,
                         runtime::KvDtype kvDtype         = runtime::KvDtype::F32,
                         bool             useStagingSlot  = false) override;

    /// Fused residual-add + RMSNorm. `x[m, k] += delta[m, k]` in place,
    /// then `y[m, k] = x[m, k] * weight[k] / sqrt(mean(x[m, :]^2) + eps)`.
    /// `y` may alias `x`. Every backend hits this pattern at the end of
    /// the attention section (`x += attn_out; ffn_norm(x) → normBuf`),
    /// so the fusion is architecture-agnostic. Saves one dispatch per
    /// transformer block. K must be representable as int32.
    void addRmsNormAsync(float*       x,
                         const float* delta,
                         std::size_t  M,
                         std::size_t  K,
                         const float* weight,
                         float        eps,
                         float*       y) override;

    /// In-place broadcast bias: y[m, k] += bias[k].
    void addBiasAsync(float*       y,
                      std::size_t  M,
                      std::size_t  K,
                      const float* bias) override;

    /// In-place residual: y[i] += x[i] for i in [0, n).
    void addResidualAsync(float*       y,
                          const float* x,
                          std::size_t  n) override;

    /// Fused SwiGLU step: gate[i] = silu(gate[i]) * up[i] for i in [0, n).
    void siluMulAsync(float*       gate,
                      const float* up,
                      std::size_t  n) override;

    /// In-place RoPE on a [seqLen, numHeads, headDim] f32 buffer. The
    /// per-position angle uses `startPos` as the absolute offset of
    /// row 0 — pass cache.length() in decode mode.
    ///
    /// M-CLR.2 Wave 3b: `writeOffsetStride` shifts `xBase` inside the
    /// kernel by `startPos * writeOffsetStride`. Q-rope leaves it at 0
    /// (workspace buffer, stable); K-rope passes the layer's kvDim so
    /// `xBase = cache.baseK(L)` reaches the current write slot at
    /// startPos*kvDim without the host having to bind a token-varying
    /// pointer.
    /// M10.2 Phase 0 Commit 4/5 — `kvDtype` selects between the f32
    /// rope kernel (default) and the fp16-KV variant. `xBase` is
    /// typed as `void*` so both Q-rope (fp32 workspace) and K-rope
    /// (fp32 or fp16 cache slot base) can flow through the same call.
    /// Q-rope callers always pass `KvDtype::F32` because Q-rope
    /// targets the fp32 workspace regardless of KV storage.
    void ropeInPlaceAsync(void*            xBase,
                          std::size_t      seqLen,
                          std::size_t      numHeads,
                          std::size_t      headDim,
                          std::size_t      startPos,
                          float            base,
                          std::size_t      writeOffsetStride = 0,
                          runtime::KvDtype kvDtype           = runtime::KvDtype::F32) override;

    /// In-place RoPE with per-pair frequency factors (ggml_rope_ext's
    /// `freq_factors` argument). `freqFactors` points at [headDim/2] f32
    /// values; the rotation angle becomes
    ///   theta_i = pos * base^(-2i/headDim) / freqFactors[i]
    /// Used by Gemma 3/4 global-attention layers for proportional RoPE.
    /// `writeOffsetStride` semantics match `ropeInPlaceAsync`.
    /// M10.2 Phase 0 Commit 4/5 — `kvDtype` selects between the f32
    /// rope kernel and the fp16-KV variant. See `ropeInPlaceAsync`
    /// for the void* rationale.
    void ropeInPlaceWithFactorsAsync(void*            xBase,
                                     const float*     freqFactors,
                                     std::size_t      seqLen,
                                     std::size_t      numHeads,
                                     std::size_t      headDim,
                                     std::size_t      startPos,
                                     float            base,
                                     std::size_t      writeOffsetStride = 0,
                                     runtime::KvDtype kvDtype           = runtime::KvDtype::F32) override;

    /// In-place scalar multiply: y[i] *= s for i in [0, n).
    /// Used by Gemma 4 for layer_output_scale.
    void mulScalarAsync(float*       y,
                        float        s,
                        std::size_t  n) override;

    /// GELU-flavoured SwiGLU: gate[i] = gelu_tanh(gate[i]) * up[i].
    /// Used by Gemma 4's FFN paths (vs Qwen's siluMulAsync).
    void geluMulAsync(float*       gate,
                      const float* up,
                      std::size_t  n) override;

    /// Fused scaled accumulate: dst[i] += scale * src[i]. Replaces a
    /// mulScalarAsync(src, scale) + addResidualAsync(dst, src) pair
    /// where the intermediate scaled src is not read elsewhere. Used by
    /// the Gemma 4 MoE per-expert loop.
    void scaledAddResidualAsync(float*       dst,
                                const float* src,
                                float        scale,
                                std::size_t  n) override;

    /// Per-row symmetric int8 quantisation of an [M, K] f32 matrix.
    /// Writes int8 quants into `y` and per-row scales into `scale`
    /// (scale[m] = max_k(|x[m, k]|) / 127). Feeds the DP4A Q8_0 matmul
    /// (M8.H.1) — the matmul consumes `y` + `scale` alongside the Q8_0
    /// weight blocks. Zero-input rows produce scale=0 and all-zero
    /// quants, which round-trip back to zero through the matmul.
    void xQuantI8Async(const float*   x,
                       std::int8_t*   y,
                       float*         scale,
                       std::size_t    M,
                       std::size_t    K) override;

    /// M10.2 Phase 1a Commit 3 — write `T` fp32 rows into a Q8_0-encoded
    /// KV cache slot. `xSrc` is a [T, kvDim] fp32 workspace (typically
    /// the per-layer K or V after rmsnorm + RoPE); `kvDst` is the
    /// layer's Q8_0 K or V cache base pointer; the kernel writes rows
    /// [writeOffset, writeOffset+T) into 32-element blocks (fp16 scale
    /// + 32 int8) matching `compute::quant::Q8_0::quantizeRow`.
    /// `kvDim` MUST be a multiple of 32 (KvCache ctor asserts the same;
    /// this method double-checks so a miswired caller trips loudly).
    /// Same immediate/replay semantics as ropeInPlaceAsync — the shared
    /// `curLenSlot()` USM int drives the row offset.
    void kvQuantCommitQ8Async(const float* xSrc,
                              void*        kvDst,
                              std::size_t  T,
                              std::size_t  kvDim,
                              std::size_t  writeOffset) override;

    /// Scatter the output of a fused QKV matmul into the separate Q, K,
    /// V destinations. `fused` has shape [M, Nq + Nkv * (1 + hasV)];
    /// `Yq` [M, Nq]; `YkBase` / `YvBase` point at the layer's K/V cache
    /// base — the kernel adds `writeOffset * Nkv` to reach the current
    /// write row. `Yv` may be any valid pointer when `hasV == false`.
    /// `writeOffset` = current KV cache length (0 for the qkvSplit
    /// self-test path).
    /// M10.2 Phase 0 Commit 4/5 — `kvDtype` selects between the f32
    /// scatter kernel and the fp16-KV variant. `YkBase`/`YvBase` are
    /// typed as `void*` so both fp32 and fp16 backing storage flow
    /// through without reinterpret_cast at the call site — the fp16
    /// kernel uses `vstore_half` on the K/V destinations. `Yq` stays
    /// fp32 in both paths.
    void qkvSplitAsync(const float*     fused,
                       float*           Yq,
                       void*            YkBase,
                       void*            YvBase,
                       std::size_t      M,
                       std::size_t      Nq,
                       std::size_t      Nkv,
                       bool             hasV,
                       std::size_t      writeOffset     = 0,
                       runtime::KvDtype kvDtype         = runtime::KvDtype::F32,
                       bool             useStagingSlot  = false) override;

    /// Load-time self-test — run every GPU op with a known input and
    /// compare against a CPU reference within a tight tolerance. Catches
    /// broken SPV loads, driver miscompilation, and layout mismatches
    /// on unfamiliar iGPU µarchs before the first block runs. Throws
    /// std::runtime_error on any parity failure so loadModel() aborts
    /// with a clear error rather than silently corrupting inference.
    ///
    /// Currently exercises: qkv_split (full QKV + alt-attention paths).
    /// Runs in < 5 ms on any Intel iGPU. Idempotent, cheap to repeat.
    void selfTest(core::l0::UsmAllocator& allocator);

    /// "pending" | "ok" — populated by selfTest(). Exposed via
    /// /v1/system/status so the deploy can be verified without pulling
    /// docker logs.
    [[nodiscard]] std::string_view selfTestStatus() const noexcept override {
        return _selfTestStatus;
    }

    /// Runtime rollback states cached at construction from
    /// features.flashPrefill / features.flashPrefillGqaQ8. Surfaced via
    /// /v1/system/status.kernels.prefill_flash so an operator can
    /// verify the deployed config took effect without reading logs.
    [[nodiscard]] bool prefillFlashEnabled() const noexcept override {
        return !_prefillFlashDisabled;
    }
    [[nodiscard]] bool prefillFlashGqaQ8Enabled() const noexcept override {
        return !_prefillFlashGqaQ8Disabled;
    }
    [[nodiscard]] std::size_t prefillFlashKTileQ8() const noexcept override {
        return _prefillFlashKTileQ8;
    }
    [[nodiscard]] std::string_view prefillFlashKTileQ8Source() const noexcept override {
        return _prefillFlashKTileQ8Source;
    }

    /// M8.K.Q8_0-Reorder — how features.q8_0Reorder was resolved at
    /// startup. Returned as an enum so callers (SystemStatusBuilder,
    /// future GpuMatmul dispatch guard) can distinguish Auto/Force/
    /// Disable without string-matching. String form for status JSON
    /// via `q8_0ReorderModeName()`.
    [[nodiscard]] core::config::TriState q8_0ReorderMode() const noexcept override {
        return _q8_0ReorderMode;
    }
    [[nodiscard]] std::string_view q8_0ReorderModeName() const noexcept override;

    /// Registration hook for backends: called once per tensor that was
    /// successfully reordered at load time. Increments the internal
    /// counter + accumulated byte total surfaced through
    /// q8_0ReorderTensorCount() / q8_0ReorderTotalBytes(). The label
    /// is emitted in an info log so operators can see which weights
    /// went through the reorder path without grepping backend code.
    /// Safe to call from any load-time context; no-op is not offered
    /// because the caller already knows they reordered something.
    void noteQ8_0ReorderApplied(std::size_t bytes,
                                std::string_view label) noexcept override;

    [[nodiscard]] std::size_t q8_0ReorderTensorCount() const noexcept override {
        return _q8_0ReorderTensorCount;
    }
    [[nodiscard]] std::size_t q8_0ReorderTotalBytes() const noexcept override {
        return _q8_0ReorderTotalBytes;
    }

    /// Post-model-load K-tile bench. Called from InferenceEngine once
    /// model dims are known. When `features.flashPrefillKTileQ8 == 0`
    /// (autotune requested) AND the model has a GQA shape (nQPerKv > 1)
    /// AND kvDtype is Q8_0, run both KTILE=64 and KTILE=128 variants
    /// against synthetic Q8_0 K/V at the model's actual head geometry,
    /// pin the winner in `_prefillFlashKTileQ8`, and update
    /// `_prefillFlashKTileQ8Source`. When any precondition fails, log
    /// why and leave the ctor-time value in place.
    void autotuneKTileQ8(core::l0::UsmAllocator& allocator,
                         std::size_t            nHeads,
                         std::size_t            nKvHeads,
                         std::size_t            headDim,
                         runtime::KvDtype       kvDtype);

    /// Test-only. Flips the runtime rollback of the GQA-head-packed
    /// Q8_0 prefill kernel. Callers must not depend on this in
    /// production paths — the ctor-time toggle is authoritative.
    /// Exposed so parity tests can pin either the plain per-Q-head or
    /// the packed variant without instantiating two GpuOps.
    void setPrefillFlashGqaQ8DisabledForTest(bool disabled) noexcept {
        _prefillFlashGqaQ8Disabled = disabled;
    }
    /// Test-only. Pin the K-tile variant of the packed Q8_0 kernel.
    /// Valid: {64, 128}. Parity tests use this to exercise both SPVs
    /// without instantiating two GpuOps.
    void setPrefillFlashKTileQ8ForTest(std::size_t kTile) noexcept {
        _prefillFlashKTileQ8 = kTile;
    }

    /// Multi-head GQA causal attention on the GPU. Layout-equivalent to
    /// compute::multiHeadAttention. q/k/v/out are all f32 USM:
    ///   q   [T_q, nHeads,    headDim]
    ///   k   [T_k, nKvHeads,  headDim]
    ///   v   [T_k, nKvHeads,  headDim]
    ///   out [T_q, nHeads,    headDim]
    /// scale is applied to Q·K before softmax (Qwen passes
    /// 1/sqrt(headDim); Gemma 4 passes 1.0 since it pre-scaled Q nowhere
    /// — see backend).
    ///
    /// `slidingWindow == 0` (default) means pure causal attention. A
    /// positive value clamps each query's K-range to the last
    /// `slidingWindow` causal keys — used by Gemma-family SWA layers
    /// (Gemma 4 sw=512). Non-SWA architectures (Qwen 2.5) pass 0.
    ///
    /// `kvDtype` selects between the f32 kernels (default, bit-parity
    /// with pre-M10.2 behaviour), the fp16-KV kernels landed in M10.2
    /// Phase 0 Commit 3, and the Q8_0-KV kernels landed in M10.2 Phase 1a
    /// Commit 4. `k`/`v` are typed as `const void*` (Commit 5 of Phase 0)
    /// so fp32, fp16 and Q8_0 KV backing storage all flow through without
    /// a cast — the fp16 kernels use `vload_half` at read time and the
    /// Q8_0 kernels dequantise (fp16 scale × int8) per 32-element block.
    ///
    /// Throws if nHeads is not a positive multiple of nKvHeads.
    ///
    /// T_k is not bounded here: the flash prefill / decode paths (the
    /// default for all Prod configs after M9.8b) use per-WG tile SLM
    /// (~2.5 KiB) independent of context length. The plain-attention
    /// fallback in `attentionPlainAsync` still enforces
    /// `T_k <= kAttentionMaxTk` where it is dispatched — reached only
    /// when `features.prefillFlash: false` or
    /// `headDim > kFlashMaxHeadDim` route to it.
    void attentionAsync(const float*      q,
                        const void*       k,
                        const void*       v,
                        std::size_t       T_q,
                        std::size_t       T_k,
                        std::size_t       nHeads,
                        std::size_t       nKvHeads,
                        std::size_t       headDim,
                        std::size_t       positionOffset,
                        float             scale,
                        float*            out,
                        std::size_t       slidingWindow = 0,
                        runtime::KvDtype  kvDtype       = runtime::KvDtype::F32) override;

    /// Compile-time bound on T_k for the **plain-attention fallback**
    /// path (kernels/attention.cl, attention_fp16.cl, attention_q8_0.cl).
    /// At 16384 the plain-attention SLM use (`scores[ATTN_MAX_TK] * 4 B`)
    /// sits at 64 KiB — the standard Intel Xe-LPG per-work-group SLM
    /// budget. Not a Prod-path limit: M9.8b routes all default configs
    /// through flash prefill / decode, which are per-WG-tile-SLM and
    /// unaffected by this constant. The limit only matters when a caller
    /// forces the plain path via `features.prefillFlash: false` or hits
    /// the `headDim > kFlashMaxHeadDim` fallback in `attentionAsync`.
    /// Raising further requires either a plain-kernel rewrite to
    /// online-softmax (variant A of the M9.8b design) or removing the
    /// plain path entirely.
    static constexpr std::size_t kAttentionMaxTk = 16384;

    /// Accessor to the underlying command queue. Backends use this to
    /// construct UnorderedScope around clearly-parallel kernel groups
    /// (M5f.4). The queue is also shared with GpuMatmul, so wrapping
    /// matmul + GpuOps launches in the same scope works as expected.
    [[nodiscard]] runtime::CommandQueue& queue() noexcept { return _queue; }

    /// Accessor to the USM allocator. Exposed so architecture backends
    /// with variant-specific persistent scratch (e.g. the E-series PLE
    /// slice buffer in `Gemma4E4BBackend`) can allocate directly instead
    /// of receiving yet another constructor argument.
    [[nodiscard]] core::l0::UsmAllocator& allocator() noexcept { return _alloc; }

    /// Persistent USM slot that holds the current KV-cache length (or,
    /// for RoPE, the current decode position). Kernels take it as
    /// `__global const int*` and dereference at execution time so a
    /// recorded command list stays valid after the host updates the
    /// slot. In immediate mode `ropeInPlaceAsync` etc. write the value
    /// before every dispatch; nothing changes for callers. In replay
    /// mode the host updates this slot between replays.
    ///
    /// **Invariant** (relied on by immediate-mode correctness): within
    /// one flush cycle every kernel that reads this slot must want the
    /// same value. Current callers satisfy this — every layer in a
    /// forward pass shares the same curLen — but adding a caller that
    /// mixes curLen values inside a single flush would silently give
    /// all launches the last-written value.
    [[nodiscard]] std::int32_t* curLenSlot() noexcept { return _curLenSlotUsm; }

    /// M-CLR.4 follow-up: right-size the FlashAttention partial launch
    /// geometry during recording. When 0 (default), the launch uses the
    /// actual `nKTiles` derived from `positionOffset+1` — optimal for
    /// immediate mode. When > 0, recording launches with this fixed
    /// upper bound instead of `kFlashMaxKTiles`, saving 32× empty
    /// workgroups per attention call for typical short-context chats.
    /// The caller (InferenceEngine) sets this per-generate based on
    /// `prompt_len + params.maxNewTokens`.
    void setReplayMaxKTiles(std::size_t n) noexcept override {
        _replayMaxKTiles = n;
    }

    /// M8.K.Q8_0-Reorder — matmul with Q8_0 weights in REORDERED row
    /// layout (scales-then-quants, see compute::quant::Q8_0::reorderRow).
    /// Structurally identical to the native Q8_0 vec matmul dispatched
    /// through GpuMatmul::matmul(Q8_0, ...) but reads the reordered
    /// layout so subgroup loads coalesce cleanly (the native 34-byte
    /// block stride breaks Xe iGPU wide-vector loads — see llama.cpp
    /// PR #21527 for the same fix in the SYCL backend).
    ///
    /// Test-facing entry point: not yet wired into the production
    /// GpuMatmul dispatcher. Phase 3 of the M8.K.Q8_0-Reorder ticket
    /// picks that up along with a config toggle. `wReordered` must
    /// point at Q8_0 weights transformed via `Q8_0::reorderRow`
    /// row-by-row; feeding a native-layout buffer yields garbage.
    /// `M == 1` (matvec), K a multiple of 32.
    void matmulQ8_0VecReorderAsync(const void*  wReordered,
                                   std::size_t  N,
                                   std::size_t  K,
                                   const float* x,
                                   float*       y) override;

    // FlashAttention tuning constants — publicly readable so callers can
    // compute the launch upper bound to hand to setReplayMaxKTiles().
    // M5f.3.2 (2026-07-07): shrunk from 256 to 64 to quadruple concurrent
    // workgroups at typical decode lengths on Xe-LPG (was 16 WGs for
    // curLen=500 on 8-head E4B → 64 WGs = matches 8 Xe-Cores × 8 VEs).
    // M9.8b (2026-07-12): compile-time context envelope raised from
    // 16384 to 32768 tokens by doubling `kFlashMaxKTiles`. Scratch buffer
    // (`_flashPartialUsm`) picks up the new size automatically at
    // GpuOps construction: kFlashMaxHeads * kFlashMaxKTiles *
    // (2 + kFlashMaxHeadDim) * 4 B = 64 × 512 × 514 × 4 ≈ 64 MiB (was
    // ~33 MiB). Kernels are unchanged — `nKTiles` is derived at runtime
    // from `positionOffset+1`; the constant only bounds the persistent
    // scratch allocation and the decode-flash dispatch guard in
    // attentionDecodeFlashAsync.
    static constexpr std::size_t kFlashKTileSize = 64;
    static constexpr std::size_t kFlashMaxKTiles = 32768 / kFlashKTileSize;

private:
    core::l0::L0Context&    _ctx;
    runtime::CommandQueue& _queue;
    core::l0::UsmAllocator& _alloc;

    // All Level-Zero-typed GpuModule + GpuKernel handles live in an
    // Impl struct defined in GpuOps.cpp (pimpl). Keeps GpuKernel.hpp
    // + GpuModule.hpp — and their transitive `<level_zero/ze_api.h>`
    // pull-in — out of the ~11 files that include this header. Ctor
    // constructs one Impl (loads every SPV, looks up every kernel by
    // name); dtor tears them down in reverse. Adding a new op-kernel
    // is a one-line change inside Impl in the .cpp — no header edit.
    struct Impl;
    std::unique_ptr<Impl>  _pimpl;

    std::string            _selfTestStatus{"pending"};

    // Persistent USM scratch for the FlashAttention partial/merge
    // pipeline. Sized at construction for the worst case across the
    // models we run; reused across every decode call (the engine
    // serialises calls via engineMutex so no aliasing is possible).
    void*                  _flashPartialUsm{nullptr};

    // M-CLR.2: single-int USM slot shared by all kernels that consume
    // the current decode position / KV-cache length. Host-writable,
    // device-readable, allocated once at construction.
    std::int32_t*          _curLenSlotUsm{nullptr};

    // M10.2 Phase 1a Commit 5 follow-up: second USM int slot, initialised
    // to 0 at construction and NEVER touched again. Used by the Q8_0
    // fp32-staging write path: qkv_split and rmsnorm_qkv under
    // KvDtype::Q8_0 target the fp32 K/V scratch at row 0 (no history to
    // skip), so they need a `writeOffset` slot pointing at a constant 0.
    // The main `_curLenSlotUsm` advances to `curLen` per-token to drive
    // rope/kv_quant_commit/attention, and under command-list-replay that
    // second host-write races the earlier staging dispatches (their
    // recording captured only the slot pointer, not the value, so
    // replay reads the last-written value — curLen — instead of 0).
    // Two slots eliminate the race by construction: main slot for
    // curLen-advancing ops, staging slot for the always-0 staging ops.
    std::int32_t*          _stagingOffsetSlotUsm{nullptr};

    // M-CLR.4 follow-up: right-sized flash launch bound during
    // recording. 0 = disabled (immediate-mode uses actual nKTiles).
    std::size_t            _replayMaxKTiles{0};
    std::size_t            _flashPartialBytes{0};

    // M5i.J — flash-prefill rollback via `features.flashPrefill=false`
    // in config.json. Cached at construction so the dispatcher hot-path
    // stays branch-free after startup. When true, T_q > 1 falls back to
    // attention.cl variant (a).
    bool                   _prefillFlashDisabled{false};

    // Independent rollback for the R1 GQA-head-packed Q8_0 prefill
    // kernel via `features.flashPrefillGqaQ8=false`. Cached at ctor time.
    // When true, the Q8_0 prefill path stays on the plain per-Q-head
    // kernel even for GQA-shaped models.
    bool                   _prefillFlashGqaQ8Disabled{false};

    // K-tile size the packed Q8_0 kernel dispatch will use.
    // `_prefillFlashKTileQ8Configured` = raw config value {0, 64, 128}
    // stored so autotuneKTileQ8 can tell whether the operator asked for
    // autotune (0) or an explicit pin. `_prefillFlashKTileQ8` = resolved
    // active pick used by the dispatcher; equals the configured value
    // when pinned, or the autotune winner (64 or 128) after
    // autotuneKTileQ8 runs. Between ctor and autotune the resolved
    // value falls back to 128 silently — dispatch is safe from the
    // first request.
    std::size_t            _prefillFlashKTileQ8Configured{128};
    std::size_t            _prefillFlashKTileQ8{128};
    // "pinned (config)" | "pending (autotune)" | "bench" | "skipped (no GQA)"
    std::string            _prefillFlashKTileQ8Source{"pinned (config)"};

    // M8.K.Q8_0-Reorder — features.q8_0Reorder as-configured. Read by
    // SystemStatusBuilder for the /v1/system/status.kernels payload
    // and (in Phase 4) by the GpuMatmul dispatcher to route Q8_0
    // matvecs through the reordered kernel when the weights are
    // available. Default matches the Config default (Disable).
    core::config::TriState      _q8_0ReorderMode{core::config::TriState::Disable};
    // M8.K.Q8_0-Reorder Phase 5 — counters bumped by
    // noteQ8_0ReorderApplied() from each backend that actually
    // reorders a Q8_0 tensor at load time. Zero when the feature is
    // Disable, or when the mode is on but no backend eligible for the
    // reorder path is currently loaded.
    std::size_t            _q8_0ReorderTensorCount{0};
    std::size_t            _q8_0ReorderTotalBytes{0};

    static constexpr std::uint32_t kRmsnormLocalSize     = 128;
    static constexpr std::uint32_t kElementwiseLocalSize = 256;
    static constexpr std::uint32_t kRopeLocalSize        = 256;
    // Must match X_QUANT_I8_LOCAL in kernels/x_quant_i8.cl.
    static constexpr std::uint32_t kXQuantI8LocalSize    = 128;
    // Must match KV_QUANT_COMMIT_LOCAL in kernels/kv_quant_commit_q8_0.cl
    // AND the Q8_0 block size (32 elements per block).
    static constexpr std::uint32_t kKvQuantCommitLocalSize = 32;
    // Must match ATTN_LOCAL / ATTN_SG in kernels/attention.cl.
    static constexpr std::uint32_t kAttentionLocalSize   = 16;

    // (kFlashKTileSize / kFlashMaxKTiles are declared public above.)
    // Must match ATTN_FLASH_KTILE in kernels/attention_flash_partial.cl
    // and ATTN_FLASH_MAX_KTILES in kernels/attention_flash_merge.cl.
    // Worst-case dims sized for Gemma 4 26B full-attention layers
    // (head_dim=512 on 5 of 30 layers; SWA layers are head_dim=256).
    // Bumping these only costs the static USM allocation size below.
    static constexpr std::size_t kFlashMaxHeads     = 64;
    static constexpr std::size_t kFlashMaxHeadDim   = 512;
    // Must match ATTN_FLASH_PREFILL_N_Q_PER_KV_MAX in
    // kernels/attention_prefill_flash_q8_0_gqa.cl. Bounds the per-Q-head
    // register-array + SLM allocation inside the packed kernel. Dispatch
    // falls back to the plain Q8_0 kernel when nQPerKv exceeds this.
    static constexpr std::size_t kFlashPrefillGqaMaxQPerKv = 8;

    // Internal dispatch — variant (a) for T_q > 1 when the flash-prefill
    // path is disabled or unsupported, single-WG streaming FlashAttention
    // (M5i.J) for T_q > 1 by default, and two-kernel FlashAttention
    // (M5f.3.2) for T_q == 1. All hidden behind attentionAsync.
    void attentionPlainAsync(const float*     q,
                             const void*      k,
                             const void*      v,
                             std::size_t      T_q,
                             std::size_t      T_k,
                             std::size_t      nHeads,
                             std::size_t      nKvHeads,
                             std::size_t      headDim,
                             std::size_t      positionOffset,
                             float            scale,
                             float*           out,
                             std::size_t      slidingWindow,
                             runtime::KvDtype kvDtype);
    void attentionPrefillFlashAsync(const float*     q,
                                    const void*      k,
                                    const void*      v,
                                    std::size_t      T_q,
                                    std::size_t      nHeads,
                                    std::size_t      nKvHeads,
                                    std::size_t      headDim,
                                    std::size_t      positionOffset,
                                    float            scale,
                                    float*           out,
                                    std::size_t      slidingWindow,
                                    runtime::KvDtype kvDtype);
    void attentionDecodeFlashAsync(const float*     q,
                                   const void*      k,
                                   const void*      v,
                                   std::size_t      T_k,
                                   std::size_t      nHeads,
                                   std::size_t      nKvHeads,
                                   std::size_t      headDim,
                                   std::size_t      positionOffset,
                                   float            scale,
                                   float*           out,
                                   std::size_t      slidingWindow,
                                   runtime::KvDtype kvDtype);
};

} // namespace mimirmind::compute