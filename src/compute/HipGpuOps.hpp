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

namespace mimirmind::compute {

/**
 * HIP/ROCm implementation of the backend-neutral `compute::ComputeOps`
 * interface. Parallel to the Level-Zero `compute::GpuOps` — same public
 * surface (all 25 virtuals from `ComputeOps` plus the L0-mirrored
 * feature-flag constructor arguments), but every kernel launch goes
 * through `HipModule` + `HipKernel` + `HipStream` instead of L0
 * modules / command queues.
 *
 * Skeleton stage (Schritt 3b sub-A): the ctor loads every `.hsaco`
 * from the resolved directory and allocates the persistent
 * FlashAttention partial scratch + the two shared `curLen` USM slots
 * that L0 uses for command-list-replay parity. Feature-flag getters
 * are fully functional; `noteQ8_0ReorderApplied` maintains its
 * counters. Every kernel-launch method currently throws
 * `std::runtime_error("HipGpuOps::<name>: not yet implemented")` —
 * follow-up commits fill them in group-by-group (elementwise,
 * RoPE, Q8_0 KV, attention, matvec).
 *
 * The class is deliberately structured so a partial implementation
 * still links and lets the L0 backend keep running unchanged. Once a
 * consumer holds `ComputeOps&` (Schritt 3c) and the runtime picks
 * `HipGpuOps` on HIP_TARGET_HOST, any stubbed method surfaces as a
 * clear runtime error rather than a link-time hole.
 *
 * Not thread-safe. Construct once at startup, share across the engine.
 */
class HipGpuOps : public ComputeOps {
public:
    /// Same 4-arg shape as `GpuOps` — the config knobs propagate 1:1
    /// so a config.json that steered the L0 backend keeps steering
    /// the HIP one. The ctor also allocates persistent scratch and
    /// resolves the K-tile pick up-front so the dispatch hot path
    /// stays branch-cheap.
    HipGpuOps(core::hip::HipComputeContext& ctx,
              bool                          flashPrefillEnabled      = true,
              bool                          flashPrefillGqaQ8Enabled = true,
              std::size_t                   flashPrefillKTileQ8      = 128,
              core::config::TriState        q8_0ReorderMode          =
                  core::config::TriState::Disable);
    ~HipGpuOps() override;

    HipGpuOps(const HipGpuOps&)            = delete;
    HipGpuOps& operator=(const HipGpuOps&) = delete;
    HipGpuOps(HipGpuOps&&)                 = delete;
    HipGpuOps& operator=(HipGpuOps&&)      = delete;

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

    // ---- HIP-native accessors ----------------------------------------
    //
    // Mirror `GpuOps::queue()` / `allocator()` — consumers that need
    // the raw HIP handle downcast to `HipGpuOps&`, exactly how L0
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

public:
    // Publicly readable so callers can compute launch upper bounds
    // for `setReplayMaxKTiles`. Parity with `GpuOps`.
    static constexpr std::size_t kFlashKTileSize   = 64;
    static constexpr std::size_t kFlashMaxKTiles   = 32768 / kFlashKTileSize;
    static constexpr std::size_t kFlashMaxHeads    = 64;
    static constexpr std::size_t kFlashMaxHeadDim  = 512;
    static constexpr std::size_t kAttentionMaxTk   = 16384;
};

} // namespace mimirmind::compute