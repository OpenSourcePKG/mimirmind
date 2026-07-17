// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "compute/cpu/GpuOps.hpp"

#include "compute/Activations.hpp"
#include "compute/Attention.hpp"
#include "compute/ComputeBuffer.hpp"
#include "compute/Matmul.hpp"
#include "compute/Norm.hpp"
#include "compute/Rope.hpp"
#include "core/cpu/CpuContext.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace mimirmind::compute::cpu {

namespace {

[[noreturn]] void throwNotImplemented(const char* method) {
    throw std::runtime_error(
        std::string{"compute::cpu::GpuOps::"} + method +
        ": not yet implemented in the M-CPU.0 skeleton — fill-in "
        "lands in a follow-up M-CPU.4 commit");
}

// Per-row RMS-mean-square factor: 1 / sqrt(mean(x²) + eps). Shared by
// the Gemma / no-weight / addRms variants below so the numerics stay
// consistent across the family. Double-accumulate matches the L0
// reference behaviour (compute::rmsNorm does the same).
[[nodiscard]] float rmsScale(const float* row,
                             std::size_t  K,
                             float        eps) {
    double sumSq = 0.0;
    for (std::size_t k = 0; k < K; ++k) {
        const double v = row[k];
        sumSq += v * v;
    }
    const double mean = sumSq / static_cast<double>(K);
    return 1.0F / std::sqrt(static_cast<float>(mean) + eps);
}

// GELU-tanh approximation used by Llama-family / Gemma FFNs:
//   gelu(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
[[nodiscard]] float geluTanh(float x) noexcept {
    constexpr float kSqrt2OverPi = 0.7978845608028654F;      // √(2/π)
    constexpr float kCoeff       = 0.044715F;
    const float inner = kSqrt2OverPi * (x + kCoeff * x * x * x);
    return 0.5F * x * (1.0F + std::tanh(inner));
}

} // namespace

GpuOps::GpuOps(::mimirmind::core::cpu::CpuContext& ctx)
    : _ctx{ctx} {}

// ---- Element-wise + normalisation ---------------------------------------

void GpuOps::rmsNormAsync(const float* x,
                          std::size_t  M,
                          std::size_t  K,
                          const float* weight,
                          float        eps,
                          float*       y) {
    ::mimirmind::compute::rmsNorm(x, M, K, weight, eps, y);
}

void GpuOps::rmsNormGemmaAsync(const float* x,
                               std::size_t  M,
                               std::size_t  K,
                               const float* weight,
                               float        eps,
                               float*       y) {
    // Gemma variant: y = x * (1 + weight) / sqrt(mean(x²) + eps).
    for (std::size_t m = 0; m < M; ++m) {
        const float* row = x + m * K;
        const float  inv = rmsScale(row, K, eps);
        float* outRow    = y + m * K;
        for (std::size_t k = 0; k < K; ++k) {
            outRow[k] = row[k] * (1.0F + weight[k]) * inv;
        }
    }
}

void GpuOps::rmsNormNoWeightAsync(const float* x,
                                  std::size_t  M,
                                  std::size_t  K,
                                  float        eps,
                                  float*       y) {
    // Bare RMS-normalize, no learned scale: y = x / sqrt(mean(x²) + eps).
    for (std::size_t m = 0; m < M; ++m) {
        const float* row = x + m * K;
        const float  inv = rmsScale(row, K, eps);
        float* outRow    = y + m * K;
        for (std::size_t k = 0; k < K; ++k) {
            outRow[k] = row[k] * inv;
        }
    }
}

void GpuOps::rmsNormQkvAsync(float*           /*qBuf*/,   const float* /*qWeight*/,
                             void*            /*kBase*/,  const float* /*kWeight*/,
                             void*            /*vBase*/,
                             std::size_t      /*qRows*/,
                             std::size_t      /*kvRows*/,
                             std::size_t      /*headDim*/,
                             float            /*eps*/,
                             std::size_t      /*writeOffset*/,
                             std::size_t      /*kvDim*/,
                             runtime::KvDtype /*kvDtype*/,
                             bool             /*useStagingSlot*/) {
    // Fused Q+K+V RMSNorm with KV-cache write-through and kvDtype
    // conversion. Non-trivial for CPU because we need to (a) rmsNorm
    // three sub-buffers with different row counts and weight semantics
    // (Q/K use per-headDim weights; V is no-weight), (b) route K/V
    // writes into the correct write-slot inside kBase/vBase, (c)
    // handle FP16 / Q8_0 destination formats. Left for M-CPU.4 fill-in.
    throwNotImplemented("rmsNormQkvAsync");
}

void GpuOps::addRmsNormAsync(float*       x,
                             const float* delta,
                             std::size_t  M,
                             std::size_t  K,
                             const float* weight,
                             float        eps,
                             float*       y) {
    // First x += delta in place, then rmsNorm(x) → y. y may alias x
    // (the interface docs allow it), so the second pass reads x AFTER
    // the delta accumulation is complete for that row.
    for (std::size_t m = 0; m < M; ++m) {
        float*       xRow   = x + m * K;
        const float* dRow   = delta + m * K;
        for (std::size_t k = 0; k < K; ++k) {
            xRow[k] += dRow[k];
        }
        const float inv = rmsScale(xRow, K, eps);
        float* yRow     = y + m * K;
        for (std::size_t k = 0; k < K; ++k) {
            yRow[k] = xRow[k] * weight[k] * inv;
        }
    }
}

void GpuOps::addBiasAsync(float*       y,
                          std::size_t  M,
                          std::size_t  K,
                          const float* bias) {
    ::mimirmind::compute::addBias(y, M, K, bias);
}

void GpuOps::addResidualAsync(float*       y,
                              const float* x,
                              std::size_t  n) {
    ::mimirmind::compute::addResidual(y, x, n);
}

void GpuOps::siluMulAsync(float*       gate,
                          const float* up,
                          std::size_t  n) {
    ::mimirmind::compute::siluInPlace(gate, n);
    ::mimirmind::compute::mulInPlace(gate, up, n);
}

void GpuOps::geluMulAsync(float*       gate,
                          const float* up,
                          std::size_t  n) {
    // Gemma FFN activation: gate[i] = gelu_tanh(gate[i]) * up[i].
    // Not exposed as a compute::* free function today (see grep at
    // ComputeOps.hpp:106) — inline here so the skeleton is usable for
    // any Gemma-family smoke run without waiting on a follow-up.
    for (std::size_t i = 0; i < n; ++i) {
        gate[i] = geluTanh(gate[i]) * up[i];
    }
}

void GpuOps::mulScalarAsync(float*       y,
                            float        s,
                            std::size_t  n) {
    for (std::size_t i = 0; i < n; ++i) {
        y[i] *= s;
    }
}

void GpuOps::scaledAddResidualAsync(float*       dst,
                                    const float* src,
                                    float        scale,
                                    std::size_t  n) {
    for (std::size_t i = 0; i < n; ++i) {
        dst[i] += scale * src[i];
    }
}

// ---- RoPE ---------------------------------------------------------------

void GpuOps::ropeInPlaceAsync(void*            xBase,
                              std::size_t      seqLen,
                              std::size_t      numHeads,
                              std::size_t      headDim,
                              std::size_t      startPos,
                              float            base,
                              std::size_t      writeOffsetStride,
                              runtime::KvDtype kvDtype) {
    if (kvDtype != runtime::KvDtype::F32) {
        throw std::runtime_error(
            "compute::cpu::GpuOps::ropeInPlaceAsync: only KvDtype::F32 "
            "is supported by the M-CPU.0 skeleton — FP16 / Q8_0 KV "
            "cache lands with a follow-up M-CPU.4 commit");
    }
    // writeOffsetStride is measured in F32 elements (matches the L0
    // kernel semantics — see GpuOps.cpp:627+ for the `kvDim` origin).
    // For Q-rope callers pass 0 and start at xBase; for K-rope with a
    // rolling cache, `xBase + startPos * writeOffsetStride` lands on
    // the row for the current token.
    float* effective = static_cast<float*>(xBase)
                     + startPos * writeOffsetStride;
    ::mimirmind::compute::applyRopeInPlace(effective,
                                           seqLen, numHeads, headDim,
                                           startPos, base);
}

void GpuOps::ropeInPlaceWithFactorsAsync(void*            /*xBase*/,
                                         const float*     /*freqFactors*/,
                                         std::size_t      /*seqLen*/,
                                         std::size_t      /*numHeads*/,
                                         std::size_t      /*headDim*/,
                                         std::size_t      /*startPos*/,
                                         float            /*base*/,
                                         std::size_t      /*writeOffsetStride*/,
                                         runtime::KvDtype /*kvDtype*/) {
    // Proportional-RoPE variant (Gemma 3/4 global-attention). CPU
    // reference doesn't have a freqFactors variant of applyRopeInPlace
    // today — implementing it means duplicating the RoPE inner loop
    // with per-pair 1/freqFactors[i] scaling. Left for M-CPU.4.
    throwNotImplemented("ropeInPlaceWithFactorsAsync");
}

// ---- Quantisation + KV commit ------------------------------------------

void GpuOps::xQuantI8Async(const float* x,
                           std::int8_t* y,
                           float*       scale,
                           std::size_t  M,
                           std::size_t  K) {
    // Per-row symmetric int8 quantisation matching the L0 xQuantI8
    // kernel (see l0::GpuOps::xQuantI8Async → xquant_i8 SPV). Zero-input
    // rows produce scale=0 + all-zero quants, which round-trip back to
    // zero through the matmul. Same rounding as Q8_0::quantizeRow but
    // over a full row's max-abs instead of per-32-block.
    for (std::size_t m = 0; m < M; ++m) {
        const float* row = x + m * K;
        float absMax = 0.0F;
        for (std::size_t k = 0; k < K; ++k) {
            absMax = std::max(absMax, std::fabs(row[k]));
        }
        const float s = (absMax > 0.0F) ? (absMax / 127.0F) : 0.0F;
        scale[m] = s;
        const float invS = (s > 0.0F) ? (1.0F / s) : 0.0F;
        std::int8_t* out = y + m * K;
        for (std::size_t k = 0; k < K; ++k) {
            const int q =
                static_cast<int>(std::lround(row[k] * invS));
            out[k] = static_cast<std::int8_t>(std::clamp(q, -127, 127));
        }
    }
}

void GpuOps::kvQuantCommitQ8Async(const float* /*xSrc*/,
                                  void*        /*kvDst*/,
                                  std::size_t  /*T*/,
                                  std::size_t  /*kvDim*/,
                                  std::size_t  /*writeOffset*/) {
    // Writes T fp32 rows into a Q8_0-encoded KV-cache slot. Semantics
    // match Q8_0::quantizeRow per 32-element block but with a slot-
    // offset and per-row stride into an interleaved cache. Left for
    // M-CPU.4 (adds a Q8_0-KV path to the reference alongside F32).
    throwNotImplemented("kvQuantCommitQ8Async");
}

void GpuOps::qkvSplitAsync(const float*     /*fused*/,
                           float*           /*Yq*/,
                           void*            /*YkBase*/,
                           void*            /*YvBase*/,
                           std::size_t      /*M*/,
                           std::size_t      /*Nq*/,
                           std::size_t      /*Nkv*/,
                           bool             /*hasV*/,
                           std::size_t      /*writeOffset*/,
                           runtime::KvDtype /*kvDtype*/,
                           bool             /*useStagingSlot*/) {
    // Scatter of a fused QKV matmul output into three destinations
    // with kvDtype conversion and rolling-slot arithmetic. Non-trivial
    // for CPU because of the FP16 / Q8_0 K/V write paths. Left for
    // M-CPU.4 fill-in.
    throwNotImplemented("qkvSplitAsync");
}

// ---- Attention ----------------------------------------------------------

void GpuOps::attentionAsync(const float*     q,
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
                            runtime::KvDtype kvDtype) {
    if (kvDtype != runtime::KvDtype::F32) {
        throw std::runtime_error(
            "compute::cpu::GpuOps::attentionAsync: only KvDtype::F32 "
            "is supported by the M-CPU.0 skeleton — FP16 / Q8_0 KV "
            "cache lands with a follow-up M-CPU.4 commit");
    }
    // compute::multiHeadAttention bakes in scale = 1/sqrt(headDim).
    // Callers that pass a different scale (Gemma 4 uses 1.0 because
    // Q was pre-scaled elsewhere) don't fit today's reference. When
    // the mismatch is small (< 1e-3 relative) we still delegate;
    // otherwise throw so bugs surface early.
    const float expected = 1.0F / std::sqrt(static_cast<float>(headDim));
    if (std::fabs(scale - expected) > 1e-3F * expected) {
        throw std::runtime_error(
            "compute::cpu::GpuOps::attentionAsync: custom scale (" +
            std::to_string(scale) + " vs baked-in " +
            std::to_string(expected) + ") not supported by the "
            "reference multiHeadAttention — M-CPU.4 lifts the "
            "restriction by threading `scale` through");
    }
    std::vector<float> scratch(T_k);
    ::mimirmind::compute::multiHeadAttention(
        q,
        static_cast<const float*>(k),
        static_cast<const float*>(v),
        T_q, T_k,
        nHeads, nKvHeads, headDim,
        positionOffset,
        scratch.data(),
        out,
        slidingWindow);
}

// ---- Reordered Q8_0 matvec ---------------------------------------------

void GpuOps::matmulQ8_0VecReorderAsync(const void*  /*wReordered*/,
                                       std::size_t  /*N*/,
                                       std::size_t  /*K*/,
                                       const float* /*x*/,
                                       float*       /*y*/) {
    // Reordered layout is an L0-Xe iGPU-specific coalesce trick (see
    // llama.cpp PR #21527 + Q8_0.hpp reorder API). Not meaningful on
    // CPU — reference matmul reads the native layout. Left as a throw
    // rather than delegating because the caller (test-facing today)
    // should be Q8_0::reorderRow-aware.
    throwNotImplemented("matmulQ8_0VecReorderAsync");
}

// ---- Feature-flag defaults ---------------------------------------------

core::config::TriState GpuOps::q8_0ReorderMode() const noexcept {
    return core::config::TriState::Disable;
}

// ---- Stream / memory ---------------------------------------------------

void GpuOps::appendMemoryCopy(void* dst, const void* src, std::size_t bytes) {
    // Everything is host memory here — synchronous memcpy is the same
    // ordering guarantee as the L0 / HIP appendMemoryCopy would give
    // to a subsequent kernel launch.
    std::memcpy(dst, src, bytes);
}

// ---- Allocation --------------------------------------------------------

::mimirmind::compute::ComputeBuffer
GpuOps::allocate(std::size_t bytes) {
    if (bytes == 0) {
        return ::mimirmind::compute::ComputeBuffer{};
    }
    // new-expression with std::byte gives us `delete[]`-compatible
    // storage. Captureless lambda decays to a plain function pointer,
    // which is what ComputeBuffer::DeleterFn expects. No allocator
    // context to thread through — CPU allocs are stateless.
    auto* raw = new std::byte[bytes];
    return ::mimirmind::compute::ComputeBuffer{
        raw,
        bytes,
        [](void* p, std::size_t /*bytes*/, void* /*ctx*/) noexcept {
            delete[] static_cast<std::byte*>(p);
        },
        /*ctx=*/nullptr,
    };
}

void GpuOps::uploadHostBytes(void* deviceDst, const void* hostSrc,
                             std::size_t bytes) {
    std::memcpy(deviceDst, hostSrc, bytes);
}

} // namespace mimirmind::compute::cpu