// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "compute/ComputeBuffer.hpp"
#include "core/config/Config.hpp"
#include "runtime/KvCache.hpp"

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace mimirmind::compute {

/**
 * Backend-neutral kernel-launch interface. Every element-wise +
 * normalisation + attention kernel that the transformer block hits
 * shows up here as a pure virtual — one method per fused-shape.
 *
 * Concrete backends (`GpuOps` on Level-Zero, `HipGpuOps` on ROCm/HIP)
 * implement the launches against their native APIs and own the module
 * loading / kernel handles / persistent scratch buffers. Backend-
 * specific escape hatches (raw command queue, USM/HIP allocator,
 * `curLenSlot()` USM int) stay on the concrete class — consumers that
 * need them still downcast to the concrete backend, exactly the same
 * way they downcast `ComputeContext&` to `L0ComputeContext&`.
 *
 * Schicht 3 of the HW-abstraction strategy: introducing the abstract
 * base establishes the interface contract without yet forcing every
 * caller to migrate to it. `GpuOps` inherits and marks its overrides;
 * backends can keep holding `GpuOps&` (or `HipGpuOps&` when it lands)
 * until Schicht 4 unifies the launch API. This split keeps the diff
 * bounded — the alternative was a mass-rename of every backend at the
 * same time as the interface split, which was rejected as too risky
 * for one commit.
 *
 * All async methods append a kernel to the backend's queue WITHOUT
 * syncing — the caller flushes/syncs on the concrete backend before
 * reading results. Not thread-safe; construct once at startup.
 */
class ComputeOps {
public:
    virtual ~ComputeOps() = default;

    ComputeOps(const ComputeOps&)            = delete;
    ComputeOps& operator=(const ComputeOps&) = delete;
    ComputeOps(ComputeOps&&)                 = delete;
    ComputeOps& operator=(ComputeOps&&)      = delete;

    // ---- Element-wise + normalisation ----------------------------------

    virtual void rmsNormAsync(const float* x,
                              std::size_t  M,
                              std::size_t  K,
                              const float* weight,
                              float        eps,
                              float*       y) = 0;

    virtual void rmsNormGemmaAsync(const float* x,
                                   std::size_t  M,
                                   std::size_t  K,
                                   const float* weight,
                                   float        eps,
                                   float*       y) = 0;

    virtual void rmsNormNoWeightAsync(const float* x,
                                      std::size_t  M,
                                      std::size_t  K,
                                      float        eps,
                                      float*       y) = 0;

    virtual void rmsNormQkvAsync(float*           qBuf,   const float* qWeight,
                                 void*            kBase,  const float* kWeight,
                                 void*            vBase,
                                 std::size_t      qRows,
                                 std::size_t      kvRows,
                                 std::size_t      headDim,
                                 float            eps,
                                 std::size_t      writeOffset,
                                 std::size_t      kvDim,
                                 runtime::KvDtype kvDtype        = runtime::KvDtype::F32,
                                 bool             useStagingSlot = false) = 0;

    virtual void addRmsNormAsync(float*       x,
                                 const float* delta,
                                 std::size_t  M,
                                 std::size_t  K,
                                 const float* weight,
                                 float        eps,
                                 float*       y) = 0;

    virtual void addBiasAsync(float*       y,
                              std::size_t  M,
                              std::size_t  K,
                              const float* bias) = 0;

    virtual void addResidualAsync(float*       y,
                                  const float* x,
                                  std::size_t  n) = 0;

    virtual void siluMulAsync(float*       gate,
                              const float* up,
                              std::size_t  n) = 0;

    virtual void geluMulAsync(float*       gate,
                              const float* up,
                              std::size_t  n) = 0;

    virtual void mulScalarAsync(float*       y,
                                float        s,
                                std::size_t  n) = 0;

    virtual void scaledAddResidualAsync(float*       dst,
                                        const float* src,
                                        float        scale,
                                        std::size_t  n) = 0;

    // De-interleave a fused per-head [block_a | block_b] projection into
    // two contiguous [seqLen, numHeads, headDim] buffers. Qwen3-Next fuses
    // the query and a per-head output gate into one `attn_q` weight whose
    // output is `[Q_h | gate_h]` per head (head stride 2*headDim); this
    // splits it so RoPE/attention see a contiguous Q and the gate is
    // applied post-attention. `a` = first block per head, `b` = second.
    virtual void splitHeadPairAsync(const float* src,
                                    float*       a,
                                    float*       b,
                                    std::size_t  seqLen,
                                    std::size_t  numHeads,
                                    std::size_t  headDim) = 0;

    // In-place sigmoid gating: y[r,c] *= sigmoid(g[r, gateDim==1?0:c]).
    // `gateDim == dim` = per-element gate (Qwen3-Next attention output
    // gate); `gateDim == 1` = per-row scalar broadcast (shared-expert
    // gate). y is [rows, dim] f32 row-major; g is [rows, gateDim].
    virtual void sigmoidGateMulAsync(float*       y,
                                     const float* g,
                                     std::size_t  rows,
                                     std::size_t  dim,
                                     std::size_t  gateDim) = 0;

    // ---- GatedDeltaNet (Qwen3-Next linear attention, M-Q3N.3) ---------
    //
    // GPU counterparts of the compute/GatedDeltaNet.* CPU reference. The
    // CPU backend delegates straight to that reference; the GPU backends
    // dispatch dedicated kernels validated against it. All F32.

    /// In-place L2 normalisation over the innermost `dim` (head_dim):
    /// `x[r,:] /= max(sqrt(sum_j x[r,j]^2), eps)`. `rows` = number of
    /// length-`dim` vectors (== T*H). Applied to q/k in the linear layer.
    virtual void l2NormInPlaceAsync(float*      x,
                                    std::size_t rows,
                                    std::size_t dim,
                                    float       eps) = 0;

    /// Causal depthwise 1-D convolution + SiLU (Qwen3-Next `ssm_conv1d`):
    /// `out[t,c] = silu( sum_{k} convInput[t+k, c] * kernel[k, c] )`.
    /// `convInput` is [(kernelSize-1)+T, channels]; `kernel` is
    /// [kernelSize, channels] (tap-major); `out` is [T, channels].
    virtual void causalConv1dSiluAsync(const float* convInput,
                                       const float* kernel,
                                       float*       out,
                                       std::size_t  T,
                                       std::size_t  channels,
                                       std::size_t  kernelSize) = 0;

    /// Autoregressive gated delta-rule recurrence over `T` tokens, per
    /// value-head. Updates `state` [H, S, S] in place and writes `out`
    /// [T, H, S]. q/k/v are [T, H, S]; gLog/beta are [T, H] (gLog is the
    /// raw log-decay — exp applied internally). Reference:
    /// compute::gatedDeltaNetRecurrent. Single sequence (n_seqs == 1).
    virtual void gatedDeltaNetRecurrentAsync(const float* q,
                                             const float* k,
                                             const float* v,
                                             const float* gLog,
                                             const float* beta,
                                             float*       state,
                                             float*       out,
                                             std::size_t  T,
                                             std::size_t  H,
                                             std::size_t  S) = 0;

    /// GatedDeltaNet decay gate: gLog[t,h] = -exp(ssmA[h]) *
    /// softplus(alpha[t,h] + ssmDt[h]). alpha [T,H]; ssmA/ssmDt [H];
    /// gLog (out) [T,H]. Reference: compute::deltanetGate.
    virtual void deltanetGateAsync(const float* alpha,
                                   const float* ssmA,
                                   const float* ssmDt,
                                   float*       gLog,
                                   std::size_t  T,
                                   std::size_t  H) = 0;

    /// In-place logistic sigmoid: y[i] = 1/(1+exp(-y[i])). GatedDeltaNet
    /// `beta` gate. Reference: compute::sigmoidInPlace.
    virtual void sigmoidInPlaceAsync(float* y, std::size_t n) = 0;

    // ---- RoPE ---------------------------------------------------------

    virtual void ropeInPlaceAsync(void*            xBase,
                                  std::size_t      seqLen,
                                  std::size_t      numHeads,
                                  std::size_t      headDim,
                                  std::size_t      startPos,
                                  float            base,
                                  std::size_t      writeOffsetStride = 0,
                                  runtime::KvDtype kvDtype           = runtime::KvDtype::F32) = 0;

    virtual void ropeInPlaceWithFactorsAsync(void*            xBase,
                                             const float*     freqFactors,
                                             std::size_t      seqLen,
                                             std::size_t      numHeads,
                                             std::size_t      headDim,
                                             std::size_t      startPos,
                                             float            base,
                                             std::size_t      writeOffsetStride = 0,
                                             runtime::KvDtype kvDtype           = runtime::KvDtype::F32) = 0;

    // Interleaved multi-axis RoPE (IMRoPE) — Qwen3-Next / Qwen3.5-VL
    // full-attention layers (`LLM_ROPE_TYPE_IMROPE`). Same split-pair
    // rotation as ropeInPlaceAsync but the per-pair angle base is chosen
    // across four position axes via the IMRoPE sector rule. `sections`
    // points at 4 int32 dimension-section widths (GGUF
    // `<arch>.rope.dimension_sections`). For text-only positions (all
    // axes equal) this is bit-identical to ropeInPlaceAsync. Only the
    // F32 storage path is implemented in M-Q3N.2; FP16/Q8_0 KV throws.
    virtual void mropeInPlaceAsync(void*              xBase,
                                   std::size_t        seqLen,
                                   std::size_t        numHeads,
                                   std::size_t        headDim,
                                   std::size_t        startPos,
                                   float              base,
                                   const std::int32_t* sections,
                                   std::size_t        writeOffsetStride = 0,
                                   runtime::KvDtype   kvDtype           = runtime::KvDtype::F32) = 0;

    // ---- Quantisation + KV commit -------------------------------------

    virtual void xQuantI8Async(const float* x,
                               std::int8_t* y,
                               float*       scale,
                               std::size_t  M,
                               std::size_t  K) = 0;

    virtual void kvQuantCommitQ8Async(const float* xSrc,
                                      void*        kvDst,
                                      std::size_t  T,
                                      std::size_t  kvDim,
                                      std::size_t  writeOffset) = 0;

    virtual void qkvSplitAsync(const float*     fused,
                               float*           Yq,
                               void*            YkBase,
                               void*            YvBase,
                               std::size_t      M,
                               std::size_t      Nq,
                               std::size_t      Nkv,
                               bool             hasV,
                               std::size_t      writeOffset    = 0,
                               runtime::KvDtype kvDtype        = runtime::KvDtype::F32,
                               bool             useStagingSlot = false) = 0;

    // ---- Attention ----------------------------------------------------

    virtual void attentionAsync(const float*     q,
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
                                std::size_t      slidingWindow = 0,
                                runtime::KvDtype kvDtype       = runtime::KvDtype::F32) = 0;

    // ---- Reordered-Q8_0 matvec (test-facing) --------------------------

    virtual void matmulQ8_0VecReorderAsync(const void*  wReordered,
                                           std::size_t  N,
                                           std::size_t  K,
                                           const float* x,
                                           float*       y) = 0;

    // ---- Recording-side knobs -----------------------------------------

    /// Right-size the FlashAttention partial launch geometry during
    /// recording. When 0 (default), the launch uses the actual `nKTiles`
    /// derived from `positionOffset+1`. See `GpuOps::setReplayMaxKTiles`
    /// for the CLR context; HIP's hipGraph equivalent will consume the
    /// same knob when it lands.
    virtual void setReplayMaxKTiles(std::size_t n) noexcept = 0;

    // ---- Feature-flag + status accessors ------------------------------
    //
    // These reflect the ctor-time resolution of `features.*` in
    // config.json plus any post-load autotune result. Numerics-neutral
    // bookkeeping — every backend surfaces the same values through
    // `/v1/system/status`, so the interface pins them here.

    [[nodiscard]] virtual std::string_view selfTestStatus() const noexcept = 0;

    [[nodiscard]] virtual bool prefillFlashEnabled() const noexcept = 0;
    [[nodiscard]] virtual bool prefillFlashGqaQ8Enabled() const noexcept = 0;
    [[nodiscard]] virtual std::size_t prefillFlashKTileQ8() const noexcept = 0;
    [[nodiscard]] virtual std::string_view prefillFlashKTileQ8Source() const noexcept = 0;

    [[nodiscard]] virtual core::config::TriState q8_0ReorderMode() const noexcept = 0;
    [[nodiscard]] virtual std::string_view q8_0ReorderModeName() const noexcept = 0;
    [[nodiscard]] virtual std::size_t q8_0ReorderTensorCount() const noexcept = 0;
    [[nodiscard]] virtual std::size_t q8_0ReorderTotalBytes() const noexcept = 0;

    /// Load-time hook: backends invoke this once per Q8_0 tensor they
    /// actually reordered so the counters + info log stay accurate. See
    /// `GpuOps::noteQ8_0ReorderApplied` for the full contract.
    virtual void noteQ8_0ReorderApplied(std::size_t bytes,
                                        std::string_view label) noexcept = 0;

    // ---- Stream / recording ops (Schritt 3c.1) ------------------------
    //
    // Neutral wrappers around the two dominant L0-specific operations
    // that backends previously reached through `_ops.queue()`:
    // `UnorderedScope` (concurrent-dispatch window on the command
    // queue) and `appendMemoryCopy` (device↔device or device↔host
    // copy queued into the same stream as kernel launches).
    //
    // L0 impl forwards to `runtime::CommandQueue::{push,pop}Unordered`
    // and `appendMemoryCopy`. HIP impl leaves push/pop as no-ops
    // (streams already reorder freely at driver level) and routes
    // the copy through `hipMemcpyAsync` on the shared HIP stream.

    /// Enter concurrent-dispatch scope. Launches appended inside the
    /// scope may reorder / overlap; the runtime restores strict order
    /// on `popUnorderedScope()`. Use the RAII `compute::UnorderedScope`
    /// helper below rather than calling these two directly.
    virtual void pushUnorderedScope() = 0;
    virtual void popUnorderedScope()  = 0;

    /// Queue a memory copy into the same stream as kernel launches.
    /// Both pointers must be reachable from the device (backend-
    /// dependent — for L0 USM they can be host/shared/device; for HIP
    /// they can be host-pinned or device). Not synchronous — call
    /// `flush()` before the CPU reads the result.
    virtual void appendMemoryCopy(void*       dst,
                                  const void* src,
                                  std::size_t bytes) = 0;

    /// Flush pending work + wait for it. On L0 this closes the current
    /// command list, executes it, and syncs; on HIP it calls
    /// `hipStreamSynchronize`. Same semantic as `ComputeMatmul::sync()`
    /// but reachable through the ops interface for callers that only
    /// hold `ComputeOps&`.
    virtual void flush() = 0;

    /// Copy `bytes` from a device buffer to a plain host buffer
    /// synchronously — the CPU can dereference `dst` immediately after
    /// return.
    ///
    /// **Why this exists (Session 2026-07-18 finding):** L0 on Meteor
    /// Lake uses UMA/USM so a device pointer is directly host-readable
    /// with cache-line-speed access. HIP on discrete gfx1101 does NOT
    /// share memory: every host-side `logits[i]` on a `hipMalloc`'d
    /// buffer traverses PCIe, ~700 ns/access × 152 k vocab = ~110 ms
    /// per token. That was the dominant unaccounted decode cost.
    /// Callers that need to CPU-scan a device buffer (argmax, top-k,
    /// etc.) must go through this method — L0 falls back to std::memcpy
    /// (which the compiler often elides when src == dst), HIP issues a
    /// single bulk `hipMemcpy(D→H)`.
    virtual void readbackToHost(void*       hostDst,
                                const void* deviceSrc,
                                std::size_t bytes) = 0;

    // ---- Allocation (Schritt 3c.2) ------------------------------------
    //
    // Neutral buffer factory. Consumers that used to construct
    // `core::l0::UsmHandle{allocator(), bytes}` now call
    // `ops.allocate(bytes)` and hold the resulting `ComputeBuffer` as
    // a value member (see `runtime::BlockBuffers`). The concrete
    // backend installs a deleter closure at allocate() time; the
    // buffer releases itself on destruction without depending on any
    // backend type.
    //
    // A zero-byte request returns an empty `ComputeBuffer` (deleter
    // unset, dtor is a no-op) — mirrors the current UsmHandle default-
    // ctor semantics.

    /// Allocate `bytes` of device-visible memory. Throws (backend-
    /// dependent exception type) on driver failure. Returned buffer
    /// owns the allocation and frees it via the deleter installed by
    /// this ops instance's backing allocator.
    [[nodiscard]] virtual ComputeBuffer allocate(std::size_t bytes) = 0;

    /// Synchronous host-to-device copy. Blocks until the transfer is
    /// visible to subsequent GPU dispatches. Used by loaders that need
    /// "the mmap bytes must be on the device before we return".
    ///
    /// L0 impl is a plain `std::memcpy` — USM is host-visible on the
    /// target GPU. HIP impl is a blocking `hipMemcpy(hipMemcpyHostToDevice)`.
    /// Callers should batch multi-tensor loads with a single trailing
    /// `flush()` if they need cross-tensor ordering; per-tensor this
    /// call is already ordered against subsequent kernel launches.
    virtual void uploadHostBytes(void*       deviceDst,
                                 const void* hostSrc,
                                 std::size_t bytes) = 0;

protected:
    ComputeOps() = default;
};

/**
 * RAII helper: enters an unordered-dispatch scope on construction,
 * exits on destruction. Non-copyable + non-movable to avoid dangling
 * push/pop pairings. Same shape as the L0-native
 * `runtime::UnorderedScope` — direct drop-in for callers that migrate
 * from `_ops.queue()` to `_ops`.
 */
class UnorderedScope {
public:
    explicit UnorderedScope(ComputeOps& ops) : _ops{ops} {
        _ops.pushUnorderedScope();
    }
    ~UnorderedScope() { _ops.popUnorderedScope(); }

    UnorderedScope(const UnorderedScope&)            = delete;
    UnorderedScope& operator=(const UnorderedScope&) = delete;
    UnorderedScope(UnorderedScope&&)                 = delete;
    UnorderedScope& operator=(UnorderedScope&&)      = delete;

private:
    ComputeOps& _ops;
};

} // namespace mimirmind::compute