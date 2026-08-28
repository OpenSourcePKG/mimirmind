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
 * `moe_down_fused_k_q8_0`) that already exist under `kernels/hip/llm/`.
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

    // M-Cuda.Batch Cat B: batched variant (Q5_K only for now) — nSeq
    // decode tokens, each with its own gateAct / routed experts / router
    // weights / RMW accumulator, in one launch. CUDA-only, parity-gated.
    void moeDownFusedKBatchedAsync(::mimirmind::core::gguf::GgmlType type,
                                   const float*         gateAct,
                                   const void*          W,
                                   const std::int32_t*  expIdx,
                                   const float*         kw,
                                   float*               accum,
                                   std::size_t          nSeq,
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

    // M-Cuda.Batch Cat B: batched variant — nSeq decode tokens, each with
    // its own x + routed-expert list, in one launch. Expert banks shared.
    // CUDA-only (parity-gated before wiring).
    void moeGateUpFusedKBatchedAsync(::mimirmind::core::gguf::GgmlType type,
                                     const float*         x,
                                     const void*          Wg,
                                     const void*          Wu,
                                     const std::int32_t*  expIdx,
                                     float*               gateActOut,
                                     std::size_t          nSeq,
                                     std::size_t          dModel,
                                     std::size_t          nFf,
                                     std::size_t          kActive,
                                     std::size_t          expertBytesGate,
                                     std::size_t          expertBytesUp) override;

    [[nodiscard]] bool moeGateUpFusedKAvailable(
        ::mimirmind::core::gguf::GgmlType type) const noexcept override;

    // E-d.5 FP4-TC decode (plain nibbles + swizzled SFB + per-expert global).
    void moeGateUpFusedKTcBatchedAsync(
            const float* x, const void* WgNib, const void* WuNib,
            const void* WgSfb, const void* WuSfb,
            const float* WgGlobal, const float* WuGlobal,
            const std::int32_t* expIdx, float* gateActOut,
            std::size_t nSeq, std::size_t dModel, std::size_t nFf,
            std::size_t kActive, std::size_t sfbStride) override;
    void moeDownFusedKTcBatchedAsync(
            const float* gateAct, const void* WNib, const void* WSfb,
            const float* WGlobal, const std::int32_t* expIdx, const float* kw,
            float* accum, std::size_t nSeq, std::size_t ffPer, std::size_t dModel,
            std::size_t kActive, std::size_t sfbStride) override;

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

    [[nodiscard]] bool f32TcAllowed() const noexcept override {
        return _f32TcAllowed;
    }
    void setF32TcAllowed(bool allowed) noexcept override {
        _f32TcAllowed = allowed;
    }

    // Layer-2 profile-apply (5.19). Set only from the runtime after a
    // fingerprint match, and only where the ctor did not read an explicit env
    // value (see the *EnvSet flags below) — env is the debug override and wins.
    void setF32TcPrefill(bool on) noexcept override { _useF32TcPrefill = on; }
    [[nodiscard]] bool f32TcPrefillEnvOverridden() const noexcept override {
        return _f32TcPrefillEnvSet;
    }
    void setCublasFp8Prefill(bool on) noexcept override {
        _useCublasFp8Prefill = on;
    }
    [[nodiscard]] bool cublasFp8PrefillEnvOverridden() const noexcept override {
        return _cublasFp8PrefillEnvSet;
    }
    void setMmq(bool on) noexcept override { _mmqEnabled = on; }
    void setMmqTc(bool on) noexcept override { _mmqTc = on; }
    [[nodiscard]] bool mmqEnvOverridden() const noexcept override {
        return _mmqEnvSet;
    }

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

    // warp32 revision (one full CUDA warp per output row, matching
    // llama.cpp's ggml-cuda MMVQ geometry) — see matmul_q8_0_vec_dp4a.cu /
    // matmul_q6k_vec_dp4a.cu. Was 64/16 (RDNA3-sub-group-16-derived,
    // two half-warps per row); that halved effective memory coalescing.
    static constexpr std::uint32_t kDp4aLocalSize    = 128;
    static constexpr std::uint32_t kDp4aWarpSize     = 32;
    static constexpr std::uint32_t kDp4aOutputsPerGroup =
        kDp4aLocalSize / kDp4aWarpSize;

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
    // MIMIRMIND_MMQ_TC (only meaningful with MIMIRMIND_MMQ): use the int8
    // TENSOR-CORE (wmma) Q8_0 MMQ kernel instead of the dp4a one. A/B knob to
    // measure tensor-core vs dp4a for the Q8_0 prefill.
    bool                           _mmqTc{false};
    // M-Cuda.MMQ Alternative F: skip MMQ for output dims larger than this, i.e.
    // keep the vocab-sized final logit projection (lm_head, N=vocab~248320) in
    // fp32. The greedy argmax is decided by the final logits, so int8 loss there
    // tips the top token; the intermediate projections (N<=d_model-ish) run
    // through error-tolerant norms/attention. Default 32768 excludes lm_head,
    // includes every other Q8_0 matmul. MIMIRMIND_MMQ_MAX_N overrides (0/large
    // = no exclusion, i.e. MMQ everything, for the A/B).
    std::size_t                    _mmqMaxN{32768};
    // E-FP4.3: BF16 tensor-core (wmma) for ALL batched (M>1) BF16 dense GEMMs
    // (shared expert / attn projections / lm-head). Activations are rounded
    // F32->BF16 (not bit-exact, ~0.2% relL2). Default ON (+11.7% serving
    // decode). Coherence gate (64 greedy tokens, 8 diverse prompts): 7/8
    // bit-identical to the scalar path, the one divergence a benign near-tie
    // flip whose continuation stays coherent — standard BF16-serving behaviour.
    // (Excluding the lm-head does NOT remove the flip — it is upstream
    // activation noise — and costs ~80% of the win, so the whole path uses TC.)
    // MIMIRMIND_BF16_TC=0 forces it off (A/B / rollback).
    bool                           _bf16Tc{true};
    // E-FP4.5 fidelity path: run the dense-GEMM tensor-core kernel in TF32
    // (10-bit mantissa) instead of BF16 (7-bit), so the F32 activations round
    // closer to the scalar reference. This removes the benign BF16-TC near-tie
    // flip (serving-parity 8/8 bit-identical to the scalar single-session path
    // vs BF16-TC's 7/8) at NO measurable perf cost: the M<=16 tile is staging-
    // /overhead-bound, not TC-throughput-bound, so TF32's lower mma throughput
    // does not bite (measured within +/-2% of BF16-TC @nSeq16, bracketed).
    // Default ON; MIMIRMIND_TF32_TC=0 rolls back to BF16-TC. Takes precedence
    // over BF16-TC when both are on.
    bool                           _tf32Tc{true};
    // Route the dense BF16 matmul (decode GEMV M==1 and batched GEMM M>1)
    // through cuBLASLt instead of the hand-written matmul_bf16_* kernels.
    // Opt-in for A/B (MIMIRMIND_CUBLAS=1); the hand kernels stay the default
    // and the safe fallback when cuBLASLt is unavailable or errors.
    bool                           _useCublas{false};
    // As _useCublas but routes the dense BF16 branch through cuBLASLt *FP8*
    // (per-tensor E4M3): W is quantised BF16->E4M3 once (cached) and X per call,
    // both with an amax/448 device scale. Opt-in MIMIRMIND_CUBLAS_FP8=1. Reads
    // half the weight bytes of the BF16 cuBLAS path; falls back to the hand
    // kernel if cuBLASLt does not support the shape (e.g. M=1 non-TC).
    bool                           _useCublasFp8{false};
    // Also route batched (M>1) prefill dense GEMMs through the per-tensor FP8
    // path. Opt-in MIMIRMIND_CUBLAS_FP8_PREFILL=1. The dense prefill projections
    // (GDN in_proj_qkv/gate/ssm_out, attn q/k/v/o) all consume RMSNorm outputs,
    // whose bounded magnitude keeps the per-tensor E4M3 activation scale from
    // crushing precision — so the M==1-only gate can be lifted for these.
    bool                           _useCublasFp8Prefill{false};
    // Whether the ctor read an explicit MIMIRMIND_{F32_TC,CUBLAS_FP8}_PREFILL
    // env var. If so, the Layer-2 per-HW profile must not override it (env is
    // the debug override and wins). See setF32TcPrefill / setCublasFp8Prefill.
    bool                           _f32TcPrefillEnvSet{false};
    bool                           _cublasFp8PrefillEnvSet{false};
    // Same guard for MIMIRMIND_MMQ / MIMIRMIND_MMQ_TC (int8 MMQ Q8_0 prefill
    // GEMM). An explicit env pins the value; otherwise the Layer-2 profile
    // may set it. See setMmq / setMmqTc / mmqEnvOverridden.
    bool                           _mmqEnvSet{false};
    // Route batched (M>1) F32 GEMMs through the BF16/TF32 tensor-core kernel
    // (weight cast to BF16 once, cached) instead of the per-row F32 vec launches.
    // Opt-in MIMIRMIND_F32_TC_PREFILL=1: huge win for models with small F32
    // weights hit at prefill M (the MoE router ffn_gate_inp, GDN ssm_beta/alpha,
    // shared-expert router), but TF32's 10-bit mantissa is too coarse for the
    // precision-sensitive encoder/reranker path, so it stays off by default.
    bool                           _useF32TcPrefill{false};
    // Runtime gate on the F32->TC downcast, defaulting to allowed so the LLM
    // prefill uses it whenever `_useF32TcPrefill` is set. Precision-sensitive
    // callers (the encoder / reranker) flip it off for their forward via
    // `ScopedExactF32`, keeping their F32 projections bit-exact regardless.
    bool                           _f32TcAllowed{true};
    // Route the M==1 blocked-NVFP4 vec (shexp / full-attn decode) through the
    // de-interleaved uint4-coalesced kernel. Opt-in MIMIRMIND_NVFP4_DEINT=1.
    bool                           _useDeintVec{false};
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

    // cuBLASLt dense BF16 matmul (opt-in via _useCublas). Computes
    // Y[M,N] (F32) = X[M,K] (F32->BF16) @ W[N,K] (BF16)^T on tensor cores.
    // Returns false if cuBLASLt is unavailable / errors, so the caller falls
    // back to the hand-written matmul_bf16_* kernels. State (handle, work-
    // space, X-staging buffer) lives in Impl.
    [[nodiscard]] bool cublasBf16Matmul(const void*  W,
                                        std::size_t  N,
                                        std::size_t  K,
                                        const float* X,
                                        std::size_t  M,
                                        float*       Y);

    // cuBLASLt per-tensor FP8 (E4M3) dense matmul. W is BF16 on entry; it is
    // quantised to E4M3 once and cached by pointer. Returns false (-> hand
    // kernel) if cuBLASLt cannot service the shape.
    [[nodiscard]] bool cublasFp8Matmul(const void*  W,
                                       std::size_t  N,
                                       std::size_t  K,
                                       const float* X,
                                       std::size_t  M,
                                       float*       Y);

    // M==1 blocked-NVFP4 vec via the de-interleaved uint4-coalesced kernel.
    // De-interleaves W once (cached by pointer) then launches the vec kernel.
    void nvblkDeintVec(const void* W, std::size_t N, std::size_t K,
                       const float* X, float* Y);
};

} // namespace mimirmind::compute::cuda