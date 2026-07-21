// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "compute/ComputeOps.hpp"

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace mimirmind::core::cpu {
class CpuContext;
}

namespace mimirmind::compute::cpu {

/**
 * CPU implementation of `compute::ComputeOps`. Every kernel dispatches
 * to the reference paths in the `compute` library — element-wise loops,
 * `compute::rmsNorm`, `compute::multiHeadAttention`, etc. — running on
 * the calling thread. There is no queue, no async work, no recording;
 * every "async" method completes before returning.
 *
 * Skeleton stage (M-CPU.0): the 1:1-mappable methods forward to the
 * matching `compute::*` free function, small inline loops fill the
 * trivial ones (mulScalar, scaledAddResidual, xQuantI8), and the fused
 * paths (rmsNormQkv, qkvSplit, kvQuantCommitQ8, matmulQ8_0VecReorder,
 * geluMul, ropeInPlaceWithFactors) throw a NotImplemented with a
 * pointer at the follow-up commit that will fill them in. Feature
 * flags return neutral defaults (flash disabled, no reorder, self-test
 * "cpu-noop") because none of those code paths apply to the CPU
 * reference.
 *
 * Takes `CpuContext&` for symmetry with the L0 / HIP GpuOps — the
 * context isn't read today but will be once fused paths land and
 * potentially need a shared scratch buffer.
 *
 * Not thread-safe. Construct once at startup, share across the engine.
 */
class GpuOps : public ::mimirmind::compute::ComputeOps {
public:
    explicit GpuOps(::mimirmind::core::cpu::CpuContext& ctx);
    ~GpuOps() override = default;

    GpuOps(const GpuOps&)            = delete;
    GpuOps& operator=(const GpuOps&) = delete;
    GpuOps(GpuOps&&)                 = delete;
    GpuOps& operator=(GpuOps&&)      = delete;

    // ---- Element-wise + normalisation -----------------------------------

    void rmsNormAsync(const float* x,
                      std::size_t  M,
                      std::size_t  K,
                      const float* weight,
                      float        eps,
                      float*       y) override;

    void rmsNormGemmaAsync(const float* x,
                           std::size_t  M,
                           std::size_t  K,
                           const float* weight,
                           float        eps,
                           float*       y) override;

    void rmsNormNoWeightAsync(const float* x,
                              std::size_t  M,
                              std::size_t  K,
                              float        eps,
                              float*       y) override;

    void rmsNormQkvAsync(float*           qBuf,   const float* qWeight,
                         void*            kBase,  const float* kWeight,
                         void*            vBase,
                         std::size_t      qRows,
                         std::size_t      kvRows,
                         std::size_t      headDim,
                         float            eps,
                         std::size_t      writeOffset,
                         std::size_t      kvDim,
                         runtime::KvDtype kvDtype        = runtime::KvDtype::F32,
                         bool             useStagingSlot = false) override;

    void addRmsNormAsync(float*       x,
                         const float* delta,
                         std::size_t  M,
                         std::size_t  K,
                         const float* weight,
                         float        eps,
                         float*       y) override;

    void addBiasAsync(float*       y,
                      std::size_t  M,
                      std::size_t  K,
                      const float* bias) override;

    void addResidualAsync(float*       y,
                          const float* x,
                          std::size_t  n) override;

    void siluMulAsync(float*       gate,
                      const float* up,
                      std::size_t  n) override;

    void geluMulAsync(float*       gate,
                      const float* up,
                      std::size_t  n) override;

    void mulScalarAsync(float*       y,
                        float        s,
                        std::size_t  n) override;

    void scaledAddResidualAsync(float*       dst,
                                const float* src,
                                float        scale,
                                std::size_t  n) override;

    void splitHeadPairAsync(const float* src,
                            float*       a,
                            float*       b,
                            std::size_t  seqLen,
                            std::size_t  numHeads,
                            std::size_t  headDim) override;

    void sigmoidGateMulAsync(float*       y,
                             const float* g,
                             std::size_t  rows,
                             std::size_t  dim,
                             std::size_t  gateDim) override;

    // ---- RoPE -----------------------------------------------------------

    void ropeInPlaceAsync(void*            xBase,
                          std::size_t      seqLen,
                          std::size_t      numHeads,
                          std::size_t      headDim,
                          std::size_t      startPos,
                          float            base,
                          std::size_t      writeOffsetStride = 0,
                          runtime::KvDtype kvDtype           = runtime::KvDtype::F32) override;

    void ropeInPlaceWithFactorsAsync(void*            xBase,
                                     const float*     freqFactors,
                                     std::size_t      seqLen,
                                     std::size_t      numHeads,
                                     std::size_t      headDim,
                                     std::size_t      startPos,
                                     float            base,
                                     std::size_t      writeOffsetStride = 0,
                                     runtime::KvDtype kvDtype           = runtime::KvDtype::F32) override;

    void mropeInPlaceAsync(void*               xBase,
                           std::size_t         seqLen,
                           std::size_t         numHeads,
                           std::size_t         headDim,
                           std::size_t         startPos,
                           float               base,
                           const std::int32_t* sections,
                           std::size_t         writeOffsetStride = 0,
                           runtime::KvDtype    kvDtype           = runtime::KvDtype::F32) override;

    // ---- Quantisation + KV commit --------------------------------------

    void xQuantI8Async(const float* x,
                       std::int8_t* y,
                       float*       scale,
                       std::size_t  M,
                       std::size_t  K) override;

    void kvQuantCommitQ8Async(const float* xSrc,
                              void*        kvDst,
                              std::size_t  T,
                              std::size_t  kvDim,
                              std::size_t  writeOffset) override;

    void qkvSplitAsync(const float*     fused,
                       float*           Yq,
                       void*            YkBase,
                       void*            YvBase,
                       std::size_t      M,
                       std::size_t      Nq,
                       std::size_t      Nkv,
                       bool             hasV,
                       std::size_t      writeOffset    = 0,
                       runtime::KvDtype kvDtype        = runtime::KvDtype::F32,
                       bool             useStagingSlot = false) override;

    // ---- Attention -----------------------------------------------------

    void attentionAsync(const float*     q,
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
                        runtime::KvDtype kvDtype       = runtime::KvDtype::F32) override;

    // ---- Reordered Q8_0 matvec -----------------------------------------

    void matmulQ8_0VecReorderAsync(const void*  wReordered,
                                   std::size_t  N,
                                   std::size_t  K,
                                   const float* x,
                                   float*       y) override;

    // ---- Recording-side knobs ------------------------------------------

    void setReplayMaxKTiles(std::size_t /*n*/) noexcept override {}

    // ---- Feature-flag + status accessors -------------------------------

    [[nodiscard]] std::string_view selfTestStatus() const noexcept override {
        return "cpu-noop";
    }
    [[nodiscard]] bool prefillFlashEnabled()      const noexcept override { return false; }
    [[nodiscard]] bool prefillFlashGqaQ8Enabled() const noexcept override { return false; }
    [[nodiscard]] std::size_t prefillFlashKTileQ8() const noexcept override { return 0; }
    [[nodiscard]] std::string_view prefillFlashKTileQ8Source() const noexcept override {
        return "cpu-noop";
    }

    [[nodiscard]] core::config::TriState q8_0ReorderMode() const noexcept override;
    [[nodiscard]] std::string_view       q8_0ReorderModeName() const noexcept override {
        return "disabled";
    }
    [[nodiscard]] std::size_t q8_0ReorderTensorCount() const noexcept override { return 0; }
    [[nodiscard]] std::size_t q8_0ReorderTotalBytes()  const noexcept override { return 0; }

    void noteQ8_0ReorderApplied(std::size_t /*bytes*/,
                                std::string_view /*label*/) noexcept override {}

    // ---- Stream / recording ops ----------------------------------------

    void pushUnorderedScope() override {}
    void popUnorderedScope()  override {}
    void appendMemoryCopy(void* dst, const void* src, std::size_t bytes) override;
    void flush() override {}
    void readbackToHost(void* hostDst, const void* deviceSrc,
                        std::size_t bytes) override;

    // ---- Allocation ----------------------------------------------------

    [[nodiscard]] ::mimirmind::compute::ComputeBuffer
        allocate(std::size_t bytes) override;

    void uploadHostBytes(void* deviceDst, const void* hostSrc,
                         std::size_t bytes) override;

private:
    ::mimirmind::core::cpu::CpuContext& _ctx;
};

} // namespace mimirmind::compute::cpu