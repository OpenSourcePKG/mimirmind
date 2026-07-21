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

void GpuOps::rmsNormQkvAsync(float*           qBuf,   const float* qWeight,
                             void*            kBase,  const float* kWeight,
                             void*            vBase,
                             std::size_t      qRows,
                             std::size_t      kvRows,
                             std::size_t      headDim,
                             float            eps,
                             std::size_t      writeOffset,
                             std::size_t      kvDim,
                             runtime::KvDtype kvDtype,
                             bool             useStagingSlot) {
    if (kvDtype != runtime::KvDtype::F32) {
        throw std::runtime_error(
            "compute::cpu::GpuOps::rmsNormQkvAsync: only KvDtype::F32 "
            "is supported by the M-CPU.4 fills — FP16 / Q8_0 KV cache "
            "lands with a follow-up commit");
    }
    // useStagingSlot is an L0-CLR construct (mutable command list KV
    // slot indirection). No analogue on CPU.
    (void)useStagingSlot;

    // Q layout: workspace [qRows, headDim] where qRows = T*nHeads.
    // Compact, no writeOffset — one rmsNorm per row with per-headDim
    // qWeight, in place.
    for (std::size_t r = 0; r < qRows; ++r) {
        float* row = qBuf + r * headDim;
        const float inv = rmsScale(row, headDim, eps);
        for (std::size_t k = 0; k < headDim; ++k) {
            row[k] = row[k] * qWeight[k] * inv;
        }
    }

    // K / V layout: layer cache [maxSeq, kvDim] with kvDim =
    // nKvHeads * headDim. kvRows = T * nKvHeads rows to process,
    // distributed across T tokens starting at row writeOffset.
    // For row r in [0, kvRows): t = r / nKvHeads, h = r % nKvHeads,
    // physical offset = (writeOffset + t) * kvDim + h * headDim.
    if (headDim == 0 || kvDim % headDim != 0) {
        throw std::runtime_error(
            "compute::cpu::GpuOps::rmsNormQkvAsync: kvDim (" +
            std::to_string(kvDim) + ") must be a positive multiple "
            "of headDim (" + std::to_string(headDim) + ")");
    }
    const std::size_t nKvHeads = kvDim / headDim;

    // K: rmsNorm with per-headDim kWeight, in-place at cache slot.
    float* K = static_cast<float*>(kBase);
    for (std::size_t r = 0; r < kvRows; ++r) {
        const std::size_t t = r / nKvHeads;
        const std::size_t h = r % nKvHeads;
        float* row = K + (writeOffset + t) * kvDim + h * headDim;
        const float inv = rmsScale(row, headDim, eps);
        for (std::size_t k = 0; k < headDim; ++k) {
            row[k] = row[k] * kWeight[k] * inv;
        }
    }

    // V: rmsNorm without a learned weight (Gemma-4 semantics), same
    // physical layout as K in the cache.
    float* V = static_cast<float*>(vBase);
    for (std::size_t r = 0; r < kvRows; ++r) {
        const std::size_t t = r / nKvHeads;
        const std::size_t h = r % nKvHeads;
        float* row = V + (writeOffset + t) * kvDim + h * headDim;
        const float inv = rmsScale(row, headDim, eps);
        for (std::size_t k = 0; k < headDim; ++k) {
            row[k] = row[k] * inv;
        }
    }
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

void GpuOps::mropeInPlaceAsync(void*               xBase,
                               std::size_t         seqLen,
                               std::size_t         numHeads,
                               std::size_t         headDim,
                               std::size_t         startPos,
                               float               base,
                               const std::int32_t* sections,
                               std::size_t         writeOffsetStride,
                               runtime::KvDtype    kvDtype) {
    if (kvDtype != runtime::KvDtype::F32) {
        throw std::runtime_error(
            "compute::cpu::GpuOps::mropeInPlaceAsync: only KvDtype::F32 "
            "is supported (M-Q3N.2 F32-only IMRoPE path)");
    }
    float* effective = static_cast<float*>(xBase)
                     + startPos * writeOffsetStride;
    ::mimirmind::compute::applyMropeInPlace(effective,
                                            seqLen, numHeads, headDim,
                                            startPos, base, sections);
}

void GpuOps::splitHeadPairAsync(const float* src,
                                float*       a,
                                float*       b,
                                std::size_t  seqLen,
                                std::size_t  numHeads,
                                std::size_t  headDim) {
    for (std::size_t p = 0; p < seqLen; ++p) {
        for (std::size_t h = 0; h < numHeads; ++h) {
            const float* srcHead =
                src + (p * numHeads + h) * (2 * headDim);
            float* aHead = a + (p * numHeads + h) * headDim;
            float* bHead = b + (p * numHeads + h) * headDim;
            for (std::size_t d = 0; d < headDim; ++d) {
                aHead[d] = srcHead[d];
                bHead[d] = srcHead[headDim + d];
            }
        }
    }
}

void GpuOps::sigmoidGateMulAsync(float*       y,
                                 const float* g,
                                 std::size_t  rows,
                                 std::size_t  dim,
                                 std::size_t  gateDim) {
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < dim; ++c) {
            const std::size_t gcol = (gateDim == 1) ? 0 : c;
            const float gv  = g[r * gateDim + gcol];
            const float sig = 1.0F / (1.0F + std::exp(-gv));
            y[r * dim + c] *= sig;
        }
    }
}

void GpuOps::ropeInPlaceWithFactorsAsync(void*            xBase,
                                         const float*     freqFactors,
                                         std::size_t      seqLen,
                                         std::size_t      numHeads,
                                         std::size_t      headDim,
                                         std::size_t      startPos,
                                         float            base,
                                         std::size_t      writeOffsetStride,
                                         runtime::KvDtype kvDtype) {
    if (kvDtype != runtime::KvDtype::F32) {
        throw std::runtime_error(
            "compute::cpu::GpuOps::ropeInPlaceWithFactorsAsync: only "
            "KvDtype::F32 is supported by the M-CPU.4a fill — FP16 / "
            "Q8_0 KV cache lands with a follow-up commit");
    }
    // Same writeOffsetStride semantics as ropeInPlaceAsync: measured
    // in F32 elements. Q-rope callers pass 0; K-rope with a rolling
    // cache passes the layer's kvDim so `xBase + startPos*writeOffsetStride`
    // hits the row for the current token.
    float* effective = static_cast<float*>(xBase)
                     + startPos * writeOffsetStride;
    ::mimirmind::compute::applyRopeInPlaceWithFactors(
        effective, freqFactors,
        seqLen, numHeads, headDim,
        startPos, base);
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

void GpuOps::qkvSplitAsync(const float*     fused,
                           float*           Yq,
                           void*            YkBase,
                           void*            YvBase,
                           std::size_t      M,
                           std::size_t      Nq,
                           std::size_t      Nkv,
                           bool             hasV,
                           std::size_t      writeOffset,
                           runtime::KvDtype kvDtype,
                           bool             useStagingSlot) {
    if (kvDtype != runtime::KvDtype::F32) {
        throw std::runtime_error(
            "compute::cpu::GpuOps::qkvSplitAsync: only KvDtype::F32 is "
            "supported by the M-CPU.4 fills — FP16 / Q8_0 KV cache "
            "lands with a follow-up commit");
    }
    // useStagingSlot is an L0-CLR construct (mutable command list KV
    // slot indirection). No analogue on CPU where nothing is recorded.
    (void)useStagingSlot;

    // Fused row layout matches the L0 kernel contract in
    // qkv_split.cl: Q columns come first, then K, then (if hasV) V.
    // Total row width = Nq + Nkv + (hasV ? Nkv : 0).
    const std::size_t fusedStride = Nq + Nkv + (hasV ? Nkv : 0);

    // K / V cache bases already point at row 0 of the layer's cache;
    // the caller's `writeOffset` = current cache length in rows. We
    // advance to the write slot once here rather than adding
    // `writeOffset * Nkv` per row inside the loop.
    float* Yk = static_cast<float*>(YkBase) + writeOffset * Nkv;
    float* Yv = hasV
        ? static_cast<float*>(YvBase) + writeOffset * Nkv
        : nullptr;

    for (std::size_t m = 0; m < M; ++m) {
        const float* row = fused + m * fusedStride;
        std::memcpy(Yq + m * Nq,     row,             Nq  * sizeof(float));
        std::memcpy(Yk + m * Nkv,    row + Nq,        Nkv * sizeof(float));
        if (hasV) {
            std::memcpy(Yv + m * Nkv, row + Nq + Nkv, Nkv * sizeof(float));
        }
    }
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
            "is supported by the M-CPU.4 fills — FP16 / Q8_0 KV cache "
            "lands with a follow-up commit");
    }
    // M-CPU.4b: `scale` is threaded through to compute::multiHeadAttention
    // (which now accepts an optional override; positive value wins,
    // 0 / negative means "use default 1/sqrt(headDim)"). Gemma 4's 1.0F
    // and Qwen's baked-default both round-trip cleanly.
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
        slidingWindow,
        scale);
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

void GpuOps::readbackToHost(void* hostDst, const void* deviceSrc,
                            std::size_t bytes) {
    // CPU backend's "device" pointers are host memory already, so the
    // sampler can read them directly. Provide a plain memcpy for
    // callers that copy defensively regardless of backend.
    if (bytes != 0) {
        std::memcpy(hostDst, deviceSrc, bytes);
    }
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