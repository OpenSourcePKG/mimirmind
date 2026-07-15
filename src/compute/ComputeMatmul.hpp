// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/gguf/GgufTypes.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mimirmind::compute {

/**
 * M-bucket ladder shared by every matmul autotune (backend-neutral
 * constants — the buckets don't change between Level-Zero and HIP).
 *
 * Small values (16-64) cover single-request decode + short-prompt
 * prefill; large values (512-1024) cover realistic RAG-prompt prefill
 * on 26B where a batched matmul at M=400-500 is the dominant kernel
 * cost. M8.J original was {16, 64, 256} — too low for RAG-prefill.
 * The M=256 → gemmMinM threshold underfit prefill because Q6_K v2
 * loses at 256 but might win at 512+; without a bench data point
 * there, the auto-pick logic can't see the crossover.
 */
inline constexpr std::array<std::size_t, 5> kAutotuneMBuckets{
    16, 64, 256, 512, 1024,
};
inline constexpr std::size_t kAutotuneMBucketsCount =
    kAutotuneMBuckets.size();

// Legacy alias — many callers spell it `kAutotuneBucketCount` and the
// rename is not worth churning them all. Keep both names in the
// namespace so old code compiles unchanged.
inline constexpr std::size_t kAutotuneBucketCount = kAutotuneMBucketsCount;

/**
 * Per-QuantType autotune summary surfaced through /v1/system/status.
 * Backend-neutral shape: the M-bucket layout, DP4A/GEMM winner flags,
 * and the source-tag string are all decisions any backend surfaces
 * in the same JSON envelope. `HipGpuMatmul` would populate the same
 * struct — GEMM/DP4A availability just resolves against different
 * kernels.
 *
 * Was previously nested in `GpuMatmul::AutotuneReport`; moved to the
 * namespace level as part of the Schritt-3 interface extraction so
 * multiple backends can share it without a `friend`-declaration or a
 * per-backend copy.
 */
struct AutotuneReport {
    std::string name;                // e.g. "Q6_K"
    bool        gemmAvailable;       // GEMM kernel loaded for this type
    bool        gemmPicked;          // GEMM will be dispatched for at least some M
    // Legacy fields — the M=16 timing, kept so downstream consumers
    // don't break during M8.J rollout.
    double      vecMs;
    double      gemmMs;
    // M8.J — per-M-bucket timings + threshold. Buckets match
    // kAutotuneMBuckets (16, 64, 256, 512, 1024).
    std::array<std::size_t, kAutotuneBucketCount> mBuckets{};
    std::array<double, kAutotuneBucketCount>      vecMsAtM{};
    std::array<double, kAutotuneBucketCount>      gemmMsAtM{};
    std::size_t                gemmMinM{0};  // 0 = "not set" (rendered as null in JSON)
    // M8.H.3 — only meaningful on iGPUs where the DP4A module loaded.
    bool        dp4aAvailable{false};
    bool        dp4aPicked{false};
    double      dp4aMs{0.0};
    // M8.K.1 — v2 GEMM prototype (M_TILE=16) telemetry. Q8_0 only.
    bool        gemmV2Available{false};
    bool        gemmV2Picked{false};
    std::array<double, kAutotuneBucketCount> gemmV2MsAtM{};
    std::string source;              // "bench" | "env_force_gemm" | ...
};

/**
 * Backend-neutral matmul dispatcher interface. Mirrors `ComputeOps`
 * for the matmul + MoE-fused dispatch surface. Concrete backends
 * (`GpuMatmul` on Level-Zero, `HipGpuMatmul` on ROCm/HIP) load their
 * per-QuantType kernels at ctor time and route requests through the
 * same virtual entry points.
 *
 * The autotune method (`autotune(UsmAllocator&, ...)`) stays on the
 * concrete class because its allocator argument is backend-typed —
 * same rationale as `GpuOps::selfTest`. Every backend runs its own
 * bench and then reports through the neutral `autotuneReport()`
 * shape defined here.
 *
 * All async methods are non-blocking; the caller invokes `sync()` on
 * the concrete backend before reading a result buffer on the CPU.
 * Not thread-safe (the underlying kernel handles get argument
 * mutation per dispatch). Construct once at startup.
 */
class ComputeMatmul {
public:
    virtual ~ComputeMatmul() = default;

    ComputeMatmul(const ComputeMatmul&)            = delete;
    ComputeMatmul& operator=(const ComputeMatmul&) = delete;
    ComputeMatmul(ComputeMatmul&&)                 = delete;
    ComputeMatmul& operator=(ComputeMatmul&&)      = delete;

    /// True if this dispatcher will run `type` on the GPU.
    [[nodiscard]] virtual bool supports(core::gguf::GgmlType type) const noexcept = 0;

    /// Y [M, N] = X [M, K] * W [N, K]^T. Synchronous.
    virtual void matmul(core::gguf::GgmlType type,
                        const void*     W,
                        std::size_t     N,
                        std::size_t     K,
                        const float*    X,
                        std::size_t     M,
                        float*          Y,
                        float*          scratch) = 0;

    /// Same as matmul() but doesn't sync. Call sync() before the CPU
    /// reads Y.
    virtual void matmulAsync(core::gguf::GgmlType type,
                             const void*     W,
                             std::size_t     N,
                             std::size_t     K,
                             const float*    X,
                             std::size_t     M,
                             float*          Y,
                             float*          scratch) = 0;

    /// DP4A matvec with pre-quantised int8 activation.
    /// Y [M, N] = Xq [M, K] (int8) × Xscale [M] × W [N, K]^T (quantised).
    virtual void matmulDp4aAsync(core::gguf::GgmlType    type,
                                 const std::int8_t* Xq,
                                 const float*       Xscale,
                                 const void*        W,
                                 std::size_t        N,
                                 std::size_t        K,
                                 std::size_t        M,
                                 float*             Y) = 0;

    [[nodiscard]] virtual bool dp4aAvailable() const noexcept = 0;
    [[nodiscard]] virtual bool dp4aAvailable(core::gguf::GgmlType type) const noexcept = 0;

    /// Fused K-experts down projection for MoE T=1 decode.
    ///   accum[n] += sum_{k=0..kActive-1} kw[k] *
    ///                 sum_{l=0..ffPer-1} dequant<T>(W[expIdx[k]], n, l)
    ///                                   * gateAct[k, l]
    virtual void moeDownFusedKAsync(core::gguf::GgmlType type,
                                    const float*         gateAct,
                                    const void*          W,
                                    const std::int32_t*  expIdx,
                                    const float*         kw,
                                    float*               accum,
                                    std::size_t          ffPer,
                                    std::size_t          dModel,
                                    std::size_t          kActive,
                                    std::size_t          expertBytes) = 0;

    [[nodiscard]] virtual bool moeDownFusedKAvailable() const noexcept = 0;
    [[nodiscard]] virtual bool moeDownFusedKAvailable(core::gguf::GgmlType type) const noexcept = 0;

    /// Flush any pending appends. Safe to call when there's no
    /// pending work — cheap no-op.
    virtual void sync() = 0;

    /// Report the autotune decision + timings per QuantType. Rows are
    /// empty until autotune() has run on the concrete backend.
    [[nodiscard]] virtual std::vector<AutotuneReport> autotuneReport() const = 0;

protected:
    ComputeMatmul() = default;
};

} // namespace mimirmind::compute