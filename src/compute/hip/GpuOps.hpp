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

namespace mimirmind::core::hip {
class HipComputeContext;
class HipStream;
class HipMemoryAllocator;
}

namespace mimirmind::compute::hip {

/**
 * HIP/ROCm implementation of the backend-neutral `compute::ComputeOps`
 * interface. Parallel to the Level-Zero `compute::l0::GpuOps` — same
 * public surface (all 25 virtuals from `ComputeOps` plus the L0-mirrored
 * feature-flag constructor arguments), but every kernel launch goes
 * through `HipModule` + `HipKernel` + `HipStream` instead of L0
 * modules / command queues.
 *
 * Same class name (`GpuOps`) as the L0 side; disambiguation happens
 * through the `compute::hip::` namespace vs `compute::l0::`. This
 * mirrors the `src/core/gpu/{l0,hip}/` primitive layout.
 *
 * Skeleton stage (Schritt 3b sub-A): the ctor loads every `.hsaco`
 * from the resolved directory and allocates the persistent
 * FlashAttention partial scratch + the two shared `curLen` USM slots
 * that L0 uses for command-list-replay parity. Feature-flag getters
 * are fully functional; `noteQ8_0ReorderApplied` maintains its
 * counters. Kernel-launch methods land group-by-group in the sub-B..E
 * commits — the remaining stubs throw with an actionable message.
 *
 * The class is deliberately structured so a partial implementation
 * still links and lets the L0 backend keep running unchanged. Once a
 * consumer holds `ComputeOps&` (Schritt 3c) and the runtime picks
 * this backend on HIP_TARGET_HOST, any stubbed method surfaces as a
 * clear runtime error rather than a link-time hole.
 *
 * Not thread-safe. Construct once at startup, share across the engine.
 */
class GpuOps : public ::mimirmind::compute::ComputeOps {
public:
    /// Same 4-arg shape as `GpuOps` — the config knobs propagate 1:1
    /// so a config.json that steered the L0 backend keeps steering
    /// the HIP one. The ctor also allocates persistent scratch and
    /// resolves the K-tile pick up-front so the dispatch hot path
    /// stays branch-cheap.
    GpuOps(core::hip::HipComputeContext& ctx,
              bool                          flashPrefillEnabled      = true,
              bool                          flashPrefillGqaQ8Enabled = true,
              std::size_t                   flashPrefillKTileQ8      = 128,
              core::config::TriState        q8_0ReorderMode          =
                  core::config::TriState::Disable);
    ~GpuOps() override;

    GpuOps(const GpuOps&)            = delete;
    GpuOps& operator=(const GpuOps&) = delete;
    GpuOps(GpuOps&&)                 = delete;
    GpuOps& operator=(GpuOps&&)      = delete;

    // ---- ComputeOps overrides (all stubbed in the skeleton) -----------

    void rmsNormAsync(const float* x, std::size_t M, std::size_t K,
                      const float* weight, float eps, float* y) override;

    void rmsNormGemmaAsync(const float* x, std::size_t M, std::size_t K,
                           const float* weight, float eps, float* y) override;

    void rmsNormNoWeightAsync(const float* x, std::size_t M, std::size_t K,
                              float eps, float* y) override;

    void rmsNormQkvAsync(float* qBuf, const float* qWeight,
                         void* kBase, const float* kWeight,
                         void* vBase,
                         std::size_t qRows, std::size_t kvRows,
                         std::size_t headDim, float eps,
                         std::size_t writeOffset, std::size_t kvDim,
                         runtime::KvDtype kvDtype = runtime::KvDtype::F32,
                         bool useStagingSlot = false) override;

    void addRmsNormAsync(float* x, const float* delta,
                         std::size_t M, std::size_t K,
                         const float* weight, float eps, float* y) override;

    void addBiasAsync(float* y, std::size_t M, std::size_t K,
                      const float* bias) override;

    void addResidualAsync(float* y, const float* x, std::size_t n) override;

    void siluMulAsync(float* gate, const float* up, std::size_t n) override;

    void geluMulAsync(float* gate, const float* up, std::size_t n) override;

    void mulScalarAsync(float* y, float s, std::size_t n) override;

    void scaledAddResidualAsync(float* dst, const float* src,
                                float scale, std::size_t n) override;

    void ropeInPlaceAsync(void* xBase,
                          std::size_t seqLen, std::size_t numHeads,
                          std::size_t headDim, std::size_t startPos,
                          float base,
                          std::size_t writeOffsetStride = 0,
                          runtime::KvDtype kvDtype = runtime::KvDtype::F32) override;

    void ropeInPlaceWithFactorsAsync(void* xBase, const float* freqFactors,
                                     std::size_t seqLen, std::size_t numHeads,
                                     std::size_t headDim, std::size_t startPos,
                                     float base,
                                     std::size_t writeOffsetStride = 0,
                                     runtime::KvDtype kvDtype = runtime::KvDtype::F32) override;

    void mropeInPlaceAsync(void* xBase, std::size_t seqLen,
                           std::size_t numHeads, std::size_t headDim,
                           std::size_t startPos, float base,
                           const std::int32_t* sections,
                           std::size_t writeOffsetStride = 0,
                           runtime::KvDtype kvDtype = runtime::KvDtype::F32) override;

    void splitHeadPairAsync(const float* src, float* a, float* b,
                            std::size_t seqLen, std::size_t numHeads,
                            std::size_t headDim) override;

    void sigmoidGateMulAsync(float* y, const float* g, std::size_t rows,
                             std::size_t dim, std::size_t gateDim) override;

    void l2NormInPlaceAsync(float* x, std::size_t rows, std::size_t dim,
                            float eps) override;
    void causalConv1dSiluAsync(const float* convInput, const float* kernel,
                               float* out, std::size_t T, std::size_t channels,
                               std::size_t kernelSize) override;
    void gatedDeltaNetRecurrentAsync(const float* q, const float* k,
                                     const float* v, const float* gLog,
                                     const float* beta, float* state,
                                     float* out, std::size_t T, std::size_t H,
                                     std::size_t S) override;
    void deltanetGateAsync(const float* alpha, const float* ssmA,
                           const float* ssmDt, float* gLog,
                           std::size_t T, std::size_t H) override;
    void sigmoidInPlaceAsync(float* y, std::size_t n) override;
    void gatherHeadsFromChannelsAsync(const float* src, float* dst,
                                      std::size_t T, std::size_t offset,
                                      std::size_t srcHeads, std::size_t dstHeads,
                                      std::size_t S,
                                      std::size_t convTotalWidth) override;

    void xQuantI8Async(const float* x, std::int8_t* y, float* scale,
                       std::size_t M, std::size_t K) override;

    void kvQuantCommitQ8Async(const float* xSrc, void* kvDst,
                              std::size_t T, std::size_t kvDim,
                              std::size_t writeOffset) override;

    void qkvSplitAsync(const float* fused, float* Yq,
                       void* YkBase, void* YvBase,
                       std::size_t M, std::size_t Nq, std::size_t Nkv,
                       bool hasV,
                       std::size_t writeOffset = 0,
                       runtime::KvDtype kvDtype = runtime::KvDtype::F32,
                       bool useStagingSlot = false) override;

    void attentionAsync(const float* q, const void* k, const void* v,
                        std::size_t T_q, std::size_t T_k,
                        std::size_t nHeads, std::size_t nKvHeads,
                        std::size_t headDim,
                        std::size_t positionOffset,
                        float scale, float* out,
                        std::size_t slidingWindow = 0,
                        runtime::KvDtype kvDtype = runtime::KvDtype::F32) override;

    void matmulQ8_0VecReorderAsync(const void* wReordered,
                                   std::size_t N, std::size_t K,
                                   const float* x, float* y) override;

    void setReplayMaxKTiles(std::size_t n) noexcept override {
        _replayMaxKTiles = n;
    }

    // ---- Feature-flag + status (functional, not stubbed) --------------

    [[nodiscard]] std::string_view selfTestStatus() const noexcept override {
        return _selfTestStatus;
    }

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

    [[nodiscard]] core::config::TriState q8_0ReorderMode() const noexcept override {
        return _q8_0ReorderMode;
    }
    [[nodiscard]] std::string_view q8_0ReorderModeName() const noexcept override;

    [[nodiscard]] std::size_t q8_0ReorderTensorCount() const noexcept override {
        return _q8_0ReorderTensorCount;
    }
    [[nodiscard]] std::size_t q8_0ReorderTotalBytes() const noexcept override {
        return _q8_0ReorderTotalBytes;
    }

    void noteQ8_0ReorderApplied(std::size_t bytes,
                                std::string_view label) noexcept override;

    // Schritt 3c.1 — stream / recording ops. push/pop are no-ops
    // (HIP streams reorder freely at the driver level; the L0
    // UnorderedScope has no direct equivalent to enable). Memcpy
    // goes through hipMemcpyAsync on the shared stream; flush
    // syncs the stream.
    void pushUnorderedScope() override;
    void popUnorderedScope()  override;
    void appendMemoryCopy(void* dst, const void* src, std::size_t bytes) override;
    void flush() override;
    void readbackToHost(void* hostDst, const void* deviceSrc,
                        std::size_t bytes) override;

    // Schritt 3c.2 — neutral buffer factory. Routes through the shared
    // `HipMemoryAllocator` in `HipAllocKind::Device` mode and installs
    // a deleter closure that calls back with the same kind so the
    // deallocate side hits the matching `hipFree` path.
    [[nodiscard]] compute::ComputeBuffer allocate(std::size_t bytes) override;
    void uploadHostBytes(void* deviceDst, const void* hostSrc,
                         std::size_t bytes) override;

    // ---- HIP-native accessors ----------------------------------------
    //
    // Mirror `GpuOps::queue()` / `allocator()` — consumers that need
    // the raw HIP handle downcast to `GpuOps&`, exactly how L0
    // consumers downcast to `GpuOps&`. Kept out of the `ComputeOps`
    // base for the same reason.

    [[nodiscard]] core::hip::HipStream&           stream() noexcept;
    [[nodiscard]] core::hip::HipMemoryAllocator&  allocator() noexcept;

    /// Persistent USM-equivalent slot for the current KV-cache length /
    /// decode position. Mirrors `GpuOps::curLenSlot()`; kernels bind
    /// it as a device pointer and dereference at launch time. In HIP
    /// this is a plain `hipMalloc`'d int; the host must use
    /// `hipMemcpy(...H2D)` to update it (no zero-copy USM equivalent
    /// on discrete gfx1101).
    [[nodiscard]] std::int32_t* curLenSlot() noexcept { return _curLenSlotUsm; }

private:
    core::hip::HipComputeContext& _ctx;

    struct Impl;
    std::unique_ptr<Impl>         _pimpl;

    // Persistent scratch — layout identical to the L0 side for pattern
    // parity. Sizes are constants below.
    void*                _flashPartialUsm{nullptr};
    std::int32_t*        _curLenSlotUsm{nullptr};
    std::int32_t*        _stagingOffsetSlotUsm{nullptr};
    std::size_t          _flashPartialBytes{0};
    std::size_t          _replayMaxKTiles{0};

    // Pinned host ring buffer for scalar-int32 H2D updates. Each
    // `stagedInt32ToDevice` call cycles to the next slot, writes the
    // value, and issues `hipMemcpyAsync` from that slot — which is
    // truly async because the source is pinned (pageable-source
    // hipMemcpyAsync silently falls back to synchronous on ROCm). The
    // ring size is generous vs. the max in-flight copies in one
    // decode-step (< 100 per token, cycles cleanly).
    static constexpr std::size_t kScalarRingSize = 256;
    std::int32_t*        _scalarRing{nullptr};
    std::size_t          _scalarRingIdx{0};

    /// Stage `value` into the next pinned slot, issue an async H2D copy
    /// on the compute stream, and return. The device slot sees the
    /// value in stream-order — subsequent kernel launches on the same
    /// stream observe the update without a host sync.
    void stagedInt32ToDevice(std::int32_t* devicePtr, std::int32_t value);

    std::string          _selfTestStatus{"pending"};

    // Feature-flag cache — resolved once in the ctor, immutable
    // afterwards except for the two setter test hooks.
    bool                 _prefillFlashDisabled{false};
    bool                 _prefillFlashGqaQ8Disabled{false};
    std::size_t          _prefillFlashKTileQ8Configured{128};
    std::size_t          _prefillFlashKTileQ8{128};
    std::string          _prefillFlashKTileQ8Source{"pinned (config)"};

    core::config::TriState _q8_0ReorderMode{core::config::TriState::Disable};
    std::size_t          _q8_0ReorderTensorCount{0};
    std::size_t          _q8_0ReorderTotalBytes{0};

    // Local-size constants shared with the HIP kernel .hip sources.
    // Kept in the class rather than in a header the kernels also
    // include because .hip files are compiled by hipcc separately and
    // consume their own local macros; parity is maintained by
    // convention (mirrored from `GpuOps` kernel constants).
    static constexpr std::uint32_t kRmsnormLocalSize      = 128;
    static constexpr std::uint32_t kElementwiseLocalSize  = 256;
    static constexpr std::uint32_t kRopeLocalSize         = 256;
    static constexpr std::uint32_t kXQuantI8LocalSize     = 128;
    static constexpr std::uint32_t kKvQuantCommitLocalSize = 32;
    static constexpr std::uint32_t kAttentionLocalSize    = 16;
    // Matches QKV_SPLIT_LOCAL / QKV_SPLIT_FP16_LOCAL in
    // kernels_hip/qkv_split{,_fp16}.hip. Numerically equal to
    // kElementwiseLocalSize today; kept separate so a future retune of
    // the elementwise WG size doesn't silently un-tune qkv_split.
    // (2026-07-17 followup from matmul_q8_0_vec launch-geometry audit.)
    static constexpr std::uint32_t kQkvSplitLocalSize     = 256;

    // Cap on how many Q-heads share one KV-head that the head-packed
    // Q8_0 prefill kernel can serve. Must match the compile-time
    // register-array + LDS bound inside
    // `attention_prefill_flash_q8_0_gqa.hip`. Dispatch falls back to
    // the plain Q8_0 kernel when nQPerKv exceeds this.
    static constexpr std::size_t kFlashPrefillGqaMaxQPerKv = 8;

    // Attention fan-out. `attentionAsync` dispatches to one of these
    // three helpers based on T_q, T_k, positionOffset and kvDtype,
    // matching the L0 GpuOps routing logic. Kept private to make the
    // public API a single entry point.
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

public:
    // Publicly readable so callers can compute launch upper bounds
    // for `setReplayMaxKTiles`. Parity with `GpuOps`.
    static constexpr std::size_t kFlashKTileSize   = 64;
    static constexpr std::size_t kFlashMaxKTiles   = 32768 / kFlashKTileSize;
    static constexpr std::size_t kFlashMaxHeads    = 64;
    static constexpr std::size_t kFlashMaxHeadDim  = 512;
    static constexpr std::size_t kAttentionMaxTk   = 16384;
};

} // namespace mimirmind::compute::hip