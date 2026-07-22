// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "compute/ComputeMatmul.hpp"
#include "core/gguf/GgufTypes.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>

namespace mimirmind::core::cuda {
class CudaComputeContext;
class CudaMemoryAllocator;
}
namespace mimirmind::core::config {
struct FeatureSettings;
}

namespace mimirmind::compute::cuda {

class GpuOps;

/**
 * HIP/ROCm implementation of the backend-neutral `compute::ComputeMatmul`
 * interface. Parallel to the Level-Zero `compute::l0::GpuMatmul` — same
 * public surface (11 virtuals from `ComputeMatmul`), but every kernel
 * launch goes through `CudaModule` + `CudaKernel` + `CudaStream` instead
 * of L0 modules / command queues.
 *
 * Same class name (`GpuMatmul`) as the L0 side; disambiguation happens
 * through the `compute::cuda::` namespace vs `compute::l0::`, mirroring
 * the `src/core/gpu/{l0,hip}/` primitive layout and the `GpuOps` split.
 *
 * Skeleton stage (sub-F): ctor loads the 5 Q8_0 matmul kernel modules
 * (`matmul_q8_0_vec`, `_gemm`, `_gemm_v2`, `_vec_dp4a`,
 * `moe_down_fused_k_q8_0`) that already exist under `kernels_hip/`.
 * `supports()`, `dp4aAvailable()`, `moeDownFusedKAvailable()`, `sync()`
 * and `autotuneReport()` are all real. Every matmul-launch method
 * currently throws `std::runtime_error("compute::cuda::GpuMatmul::<name>:
 * not yet implemented ...")` — follow-up commits fill them in
 * group-by-group (vec + async, DP4A, MoE fused-K, GEMM + autotune).
 *
 * HIP only supports Q8_0 quantised weights today; the L0-side Q4_K /
 * Q5_K / Q6_K matmul kernels haven't been ported to HIP yet. The
 * dispatcher's `supports()` returns false for any non-Q8_0 type so
 * callers know to fall back (or the HIP backend simply refuses to
 * load models with unsupported quant weights).
 *
 * Not thread-safe. Construct once at startup, share across the engine.
 */
class GpuMatmul : public ::mimirmind::compute::ComputeMatmul {
public:
    /// Ctor takes `CudaComputeContext&` (matches the l0 side's
    /// `L0ComputeContext&` pattern) plus a `GpuOps&` because the
    /// DP4A path shares `xQuantI8Async` with the elementwise kernels
    /// living on GpuOps.
    GpuMatmul(::mimirmind::core::cuda::CudaComputeContext& ctx, GpuOps& ops);
    ~GpuMatmul() override;

    GpuMatmul(const GpuMatmul&)            = delete;
    GpuMatmul& operator=(const GpuMatmul&) = delete;
    GpuMatmul(GpuMatmul&&)                 = delete;
    GpuMatmul& operator=(GpuMatmul&&)      = delete;

    // ---- ComputeMatmul overrides -------------------------------------

    [[nodiscard]] bool supports(::mimirmind::core::gguf::GgmlType type)
        const noexcept override;

    void matmul(::mimirmind::core::gguf::GgmlType type,
                const void*     W,
                std::size_t     N,
                std::size_t     K,
                const float*    X,
                std::size_t     M,
                float*          Y,
                float*          scratch) override;

    void matmulAsync(::mimirmind::core::gguf::GgmlType type,
                     const void*     W,
                     std::size_t     N,
                     std::size_t     K,
                     const float*    X,
                     std::size_t     M,
                     float*          Y,
                     float*          scratch) override;

    void matmulDp4aAsync(::mimirmind::core::gguf::GgmlType type,
                         const std::int8_t* Xq,
                         const float*       Xscale,
                         const void*        W,
                         std::size_t        N,
                         std::size_t        K,
                         std::size_t        M,
                         float*             Y) override;

    [[nodiscard]] bool dp4aAvailable() const noexcept override;
    [[nodiscard]] bool dp4aAvailable(::mimirmind::core::gguf::GgmlType type)
        const noexcept override;

    void moeDownFusedKAsync(::mimirmind::core::gguf::GgmlType type,
                            const float*         gateAct,
                            const void*          W,
                            const std::int32_t*  expIdx,
                            const float*         kw,
                            float*               accum,
                            std::size_t          ffPer,
                            std::size_t          dModel,
                            std::size_t          kActive,
                            std::size_t          expertBytes) override;

    [[nodiscard]] bool moeDownFusedKAvailable() const noexcept override;
    [[nodiscard]] bool moeDownFusedKAvailable(::mimirmind::core::gguf::GgmlType type)
        const noexcept override;

    void moeGateUpFusedKAsync(::mimirmind::core::gguf::GgmlType type,
                              const float*         x,
                              const void*          Wg,
                              const void*          Wu,
                              const std::int32_t*  expIdx,
                              float*               gateActOut,
                              std::size_t          dModel,
                              std::size_t          nFf,
                              std::size_t          kActive,
                              std::size_t          expertBytesGate,
                              std::size_t          expertBytesUp) override;

    [[nodiscard]] bool moeGateUpFusedKAvailable(
        ::mimirmind::core::gguf::GgmlType type) const noexcept override;

    void ffnGateUpFusedQ8Async(const float* x,
                               const void*  Wg,
                               const void*  Wu,
                               float*       Y,
                               std::size_t  dModel,
                               std::size_t  nFf) override;

    [[nodiscard]] bool ffnGateUpFusedQ8Available() const noexcept override;

    /// M-Cuda.MMQ B1 — Q8_0 int8 quantized-matmul GEMM for prefill (M>1).
    /// Y[M,N] = X[M,K] (fp32, int8-quantised per 32-elem block) · W[N,K]
    /// (Q8_0), int8 dp4a dots scaled per k-block. Lossy vs fp32 (int8
    /// activations) — the compute-bound-prefill accelerator. CUDA-only, not on
    /// the ComputeMatmul interface yet (production dispatch wiring is step C1).
    void matmulQ8_0MmqAsync(const void*  W,
                            std::size_t  N,
                            std::size_t  K,
                            const float* X,
                            std::size_t  M,
                            float*       Y);

    /// M-Cuda.MMQ B1b — Q8_0 int8 TENSOR-CORE (wmma) matmul GEMM for prefill.
    /// Same contract as matmulQ8_0MmqAsync but runs the int8 dot on the
    /// Blackwell int8 tensor cores (16x16x16 s8->s32), scaled per Q8_0 block.
    void matmulQ8_0MmqTcAsync(const void*  W,
                              std::size_t  N,
                              std::size_t  K,
                              const float* X,
                              std::size_t  M,
                              float*       Y);

    /// M-Cuda.MMQ B2 — Q4_K int8 quantized-matmul GEMM for prefill (M>1).
    /// Q4_K has no CUDA GEMM otherwise (vec-only); this both tiles it and runs
    /// the dot in int8. Affine per-sub-block dequant folded into the scale.
    void matmulQ4KMmqAsync(const void*  W,
                           std::size_t  N,
                           std::size_t  K,
                           const float* X,
                           std::size_t  M,
                           float*       Y);

    /// M-Cuda.MMQ B2 — Q5_K int8 quantized-matmul GEMM for prefill (M>1).
    /// Q5_K = Q4_K + one high bit per quant; same affine int8 decomposition.
    void matmulQ5KMmqAsync(const void*  W,
                           std::size_t  N,
                           std::size_t  K,
                           const float* X,
                           std::size_t  M,
                           float*       Y);

    void sync() override;

    [[nodiscard]] std::vector<::mimirmind::compute::AutotuneReport>
        autotuneReport() const override;

    /// Bench-driven pick between the matvec-loop and the batched GEMM
    /// kernel per M-bucket. Populates `_gemmMinM` (smallest bucket at
    /// which GEMM beat matvec-loop with a 5 % margin), plus per-M
    /// timing arrays surfaced through `autotuneReport()`.
    ///
    /// Config overrides short-circuit the bench:
    ///   features.gemmMinM.has_value() → pin the crossover threshold,
    ///                                    skip bench
    ///   features.gemm    == Disable   → gemmMinM = MAX (matvec always)
    ///   features.gemm    == Force     → gemmMinM = 2   (GEMM whenever M>1)
    ///   features.dp4a    == Force     → useDp4a = true (skip bench)
    ///
    /// `hiddenDim` is the model's d_model; N=K=round_up(hiddenDim, 256)
    /// so the synthetic bench matches the actual matmul shape. `alloc`
    /// is used for temporary scratch (X, Y, W, S) that gets freed before
    /// return.
    ///
    /// Idempotent — call once from `InferenceEngine::loadModel` after
    /// the model dims are known. DP4A/V2 auto-pick + full parity gate
    /// are follow-up scope (matches the HipGpuOps sub-A → sub-E
    /// incremental rhythm).
    void autotune(::mimirmind::core::cuda::CudaMemoryAllocator& alloc,
                  std::size_t                                 hiddenDim,
                  const ::mimirmind::core::config::FeatureSettings& features);

private:
    ::mimirmind::core::cuda::CudaComputeContext& _ctx;
    GpuOps&                                    _ops;

    struct Impl;
    std::unique_ptr<Impl>                      _pimpl;

    // Kernel-launch geometry constants (must stay in sync with the
    // MATMUL_Q8_0_* macros in the .hip source files). GEMM and vec
    // kernels historically diverged on threads-per-WG — GEMM keeps the
    // Intel Xe-style 64 threads (SG=16 → 4 subgroups per WG), vec was
    // restructured for RDNA3 warpSize=32 with 128 threads (4 warps per
    // WG, one warp per output row). Both agree on 4 outputs per WG so a
    // single kOutputsPerGroup is fine.
    static constexpr std::uint32_t kLocalSize        = 64;   // gemm path
    static constexpr std::uint32_t kVecLocalSize     = 128;  // vec path (matches MATMUL_Q8_0_LOCAL)
    static constexpr std::uint32_t kSubgroupSize     = 16;
    static constexpr std::uint32_t kOutputsPerGroup  = kLocalSize / kSubgroupSize;

    static constexpr std::uint32_t kDp4aLocalSize    = 64;
    static constexpr std::uint32_t kDp4aSubgroupSize = 16;
    static constexpr std::uint32_t kDp4aOutputsPerGroup =
        kDp4aLocalSize / kDp4aSubgroupSize;

    static constexpr std::size_t   kGemmMTile        = 8;
    static constexpr std::size_t   kGemmV2MTile      = 8;

    // MoE fused-K down-projection: same 4-outputs-per-WG geometry as
    // the plain vec kernel by coincidence (MOE_DOWN_LOCAL=64, SG=16 in
    // moe_down_fused_k_q8_0.hip). Declared separately so a future
    // architecture-specific tuning of one doesn't drag the other.
    static constexpr std::uint32_t kMoeDownLocalSize      = 64;
    static constexpr std::uint32_t kMoeDownOutputsPerGroup = 4;
    // moe_gate_up_fused_k_q4k: LOCAL=128 (4 warps), 4 outputs/workgroup.
    static constexpr std::uint32_t kMoeGateUpLocalSize       = 128;
    static constexpr std::uint32_t kMoeGateUpOutputsPerGroup = 4;
    // ffn_gate_up_fused_q8_0: LOCAL=128 (4 warps), 4 outputs/workgroup.
    static constexpr std::uint32_t kFfnGuQ8LocalSize       = 128;
    static constexpr std::uint32_t kFfnGuQ8OutputsPerGroup = 4;

    // Sentinel for "GEMM never wins — always take matvec-loop". Same
    // pattern as L0's `kGemmMinMNever`.
    static constexpr std::size_t   kGemmMinMNever =
        std::numeric_limits<std::size_t>::max();

    // --- Autotune-populated state --------------------------------------
    // Set once by `autotune()` and read by `matmulAsync` on every
    // dispatch. Defaults leave dispatch at the safe matvec-loop path
    // until autotune runs — same as L0.
    std::size_t                    _gemmMinM{kGemmMinMNever};
    bool                           _useGemmV2{false};
    bool                           _useDp4a{false};
    // M-Cuda.MMQ C1: MIMIRMIND_MMQ routes the Q8_0 prefill (M>1) matmul
    // through the int8 dp4a MMQ GEMM instead of the fp32 gemm/matvec path.
    // Decode (M==1) is unaffected. Default off until the prefill A/B lands.
    bool                           _mmqEnabled{false};
    std::array<double, ::mimirmind::compute::kAutotuneBucketCount>
                                   _vecMsAtM{};
    std::array<double, ::mimirmind::compute::kAutotuneBucketCount>
                                   _gemmMsAtM{};
    std::array<double, ::mimirmind::compute::kAutotuneBucketCount>
                                   _gemmV2MsAtM{};
    double                         _dp4aMs{0.0};
    std::string                    _autotuneSource{"pending (hip skeleton)"};

    // One-shot flag for the CPU-fallback dispatch path (types other than
    // Q8_0). First dispatch through the fallback logs an INFO line so
    // the operator knows a slow correctness path is active; subsequent
    // dispatches stay silent to avoid per-block log spam during prefill.
    bool                           _cpuFallbackLogged{false};
};

} // namespace mimirmind::compute::cuda