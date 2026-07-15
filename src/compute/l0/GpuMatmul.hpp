// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "compute/ComputeMatmul.hpp"
#include "core/gguf/GgufTypes.hpp"
#include "core/gpu/l0/CommandQueue.hpp"
#include "core/gpu/l0/GpuKernel.hpp"
#include "core/gpu/l0/GpuModule.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace mimirmind::core::l0 {
class L0Context;
class L0ComputeContext;
class UsmAllocator;
}
namespace mimirmind::core::config {
struct FeatureSettings;
}

namespace mimirmind::compute::l0 {

class GpuOps;

// M-bucket ladder + `AutotuneReport` shape live on `ComputeMatmul`
// so a future `HipGpuMatmul` shares them 1:1. Both are re-exported
// into the `compute::` namespace as `inline constexpr` / plain
// struct — every caller that spelled `compute::kAutotuneMBuckets` /
// `compute::AutotuneReport` keeps working unchanged.

/**
 * Drop-in replacement for compute::matmul that dispatches to GPU kernels
 * for weight types whose QuantType advertises a `gpuMatmulModule()`. For
 * unsupported types it falls back to the scalar CPU compute::matmul.
 *
 * Owns the SPIR-V modules + kernels. The command queue is shared with
 * the rest of the engine (passed in by reference) so element-wise GPU
 * ops can be appended into the same command list as matmuls. Not
 * thread-safe (the underlying ze_kernel_handle_t is mutated by
 * setArgumentValue). Construct once at startup, share across the engine.
 *
 * Holds a reference to GpuOps to route the DP4A Q8_0 path (M8.H.3)
 * through the shared x_quant_i8 kernel — no duplicated module load.
 */
class GpuMatmul : public ::mimirmind::compute::ComputeMatmul {
public:
    /// Takes `L0ComputeContext&` (Schicht 2 of the backend-neutralisation
    /// story). The concrete L0Context / UsmAllocator / CommandQueue refs
    /// are pulled from it at ctor time and stored as members — same
    /// pattern as `GpuOps`. `ops` is still passed separately because it
    /// holds compute-side state (kernel handles, self-test status) the
    /// context doesn't own.
    GpuMatmul(core::l0::L0ComputeContext& ctx,
              GpuOps&                     ops);
    ~GpuMatmul() override;

    GpuMatmul(const GpuMatmul&)            = delete;
    GpuMatmul& operator=(const GpuMatmul&) = delete;
    GpuMatmul(GpuMatmul&&)                 = delete;
    GpuMatmul& operator=(GpuMatmul&&)      = delete;

    /// True if this dispatcher will run `type` on the GPU.
    [[nodiscard]] bool supports(core::gguf::GgmlType type) const noexcept override;

    /// Per-QuantType micro-bench: for each type that has both a matvec
    /// and a GEMM kernel, time both paths on a representative prefill
    /// shape and pick the faster one. The decision is stored on the
    /// Entry and consulted by matmulAsync at every M > 1 dispatch.
    ///
    /// Idempotent — call once from `InferenceEngine::loadModel` after
    /// `_weights` is ready. The bench allocates a small temporary USM
    /// scratch through `allocator` and releases it before returning.
    /// `hiddenDim` is used as the (N=K) matmul dim; use the model's
    /// d_model. `mBatch` is the batch to time (16 mirrors a typical
    /// prefill chunk).
    ///
    /// Feature-flag overrides in `features` skip the actual bench:
    ///   features.gemm    == TriState::Disable → force matvec-loop everywhere
    ///   features.gemm    == TriState::Force   → force GEMM everywhere it exists
    ///   features.gemmMinM.has_value()         → pin the crossover threshold
    ///   features.dp4a    == TriState::Force   → pin DP4A path after parity
    ///   features.dp4a    == TriState::Disable → skip the DP4A parity/bench
    ///   features.gemmV2                       → route picked GEMM through v2 kernel
    /// If both `gemm` and `gemmMinM` are set, `gemmMinM` wins.
    void autotune(core::l0::UsmAllocator&          allocator,
                  std::size_t                     hiddenDim,
                  const core::config::FeatureSettings& features);

    /// Y [M, N] = X [M, K] * W [N, K]^T. Synchronous version (mirrors
    /// compute::matmul signature). For supported `type` the dispatch goes
    /// to the GPU kernel and we sync immediately. For unsupported types
    /// the call falls back to CPU.
    ///
    /// `scratch` (K floats) is only consumed on the CPU fallback path.
    void matmul(core::gguf::GgmlType type,
                const void*     W,
                std::size_t     N,
                std::size_t     K,
                const float*    X,
                std::size_t     M,
                float*          Y,
                float*          scratch) override;

    /**
     * Same as matmul() but doesn't sync. The Y buffer is NOT yet valid
     * when this returns; call sync() before the CPU reads from it. Use
     * to batch independent matmul calls into a single Level-Zero
     * submission and save the per-dispatch sync cost.
     *
     * For unsupported types we transparently flush any pending GPU work
     * first (to preserve ordering vs prior async appends), then run the
     * CPU fallback synchronously.
     */
    void matmulAsync(core::gguf::GgmlType type,
                     const void*     W,
                     std::size_t     N,
                     std::size_t     K,
                     const float*    X,
                     std::size_t     M,
                     float*          Y,
                     float*          scratch) override;

    /// M8.H.1 — DP4A matvec with pre-quantised int8 activation. The
    /// generic dispatcher; kernel slot lives on the Entry for `type`.
    ///   Y [M, N] = Xq [M, K] (int8) × Xscale [M] × W [N, K]^T (quantised).
    /// The activation is expected to already have been produced by
    /// GpuOps::xQuantI8Async (per-row symmetric int8, scale = amax/127).
    /// Async — call sync() before reading Y from the CPU.
    ///
    /// Available only when the DP4A kernel for `type` resolved at module
    /// load; query via dp4aAvailable(type) first. Throws if unavailable.
    void matmulDp4aAsync(core::gguf::GgmlType    type,
                         const std::int8_t* Xq,
                         const float*       Xscale,
                         const void*        W,
                         std::size_t        N,
                         std::size_t        K,
                         std::size_t        M,
                         float*             Y) override;

private:
    /// Internal dispatch for the DP4A path — quantises `X` into the
    /// persistent Xq/Xscale scratch via `_ops.xQuantI8Async`, then
    /// forwards to `matmulDp4aAsync`. Callers must first verify
    /// `M <= kDp4aMaxM && K <= kDp4aMaxK`.
    void dispatchDp4aFromFloat(core::gguf::GgmlType type,
                               const float*    X,
                               const void*     W,
                               std::size_t     N,
                               std::size_t     K,
                               std::size_t     M,
                               float*          Y);
public:

    /// True when at least one DP4A kernel loaded successfully. Query
    /// dp4aAvailable(type) for a per-type check.
    [[nodiscard]] bool dp4aAvailable() const noexcept override;

    /// True when the DP4A kernel for `type` loaded successfully. False
    /// when the SPV or the DP4A extension itself was unavailable on the
    /// target iGPU; callers must fall back to the plain matvec path.
    [[nodiscard]] bool dp4aAvailable(core::gguf::GgmlType type) const noexcept override;

    /**
     * M-MoE.Fused-Decode prototype — fused K-experts down projection for
     * MoE T=1 decode. Collapses K sequential down-matmul + scaledAdd
     * launches into one dispatch. Weight quant type is picked at call
     * time — one kernel per supported type is loaded eagerly at ctor.
     * Currently supported: Q6_K, Q8_0 (26B-A4B mixes them: gate_up=Q6_K,
     * down=Q8_0). Read-modify-write on `accum`; caller pre-zeros before
     * invoking.
     *
     *   accum[n] += sum_{k=0..kActive-1} kw[k] *
     *                 sum_{l=0..ffPer-1} dequant<T>(W[expIdx[k]], n, l)
     *                                   * gateAct[k, l]
     *
     * `gateAct`, `expIdx`, `kw` must live in USM the caller has already
     * populated (host-visible USM writes suffice on UMA-iGPU).
     * `expertBytes` is the byte stride between consecutive experts in W
     * — caller computes it from the quant type (Q6_K: `dModel *
     * (ffPer/256) * 210`; Q8_0: `dModel * (ffPer/32) * 34`).
     *
     * Async; call sync() before the CPU reads `accum`. Throws if the
     * kernel for `type` didn't load — query via
     * `moeDownFusedKAvailable(type)` first.
     */
    void moeDownFusedKAsync(core::gguf::GgmlType type,
                            const float*         gateAct,
                            const void*          W,
                            const std::int32_t*  expIdx,
                            const float*         kw,
                            float*               accum,
                            std::size_t          ffPer,
                            std::size_t          dModel,
                            std::size_t          kActive,
                            std::size_t          expertBytes) override;

    /// True when the fused-K kernel for `type` loaded at ctor.
    [[nodiscard]] bool moeDownFusedKAvailable(core::gguf::GgmlType type) const noexcept override {
        return _moeDownFusedK.find(type) != _moeDownFusedK.end();
    }

    /// True when at least one fused-K kernel loaded (any quant type).
    [[nodiscard]] bool moeDownFusedKAvailable() const noexcept override {
        return !_moeDownFusedK.empty();
    }

    /// Flush any pending appends (close + execute + sync + reset). Safe
    /// to call when there's no pending work — cheap no-op.
    void sync() override;

    /// Report the autotune decision + timings per QuantType. Rows are
    /// empty until autotune() has run. Used by ApiServer to expose the
    /// kernel-dispatch choice via /v1/system/status. The struct itself
    /// lives on `ComputeMatmul` (backend-neutral shape).
    [[nodiscard]] std::vector<AutotuneReport> autotuneReport() const override;

private:
    struct KernelSlot {
        std::unique_ptr<runtime::GpuModule> module;
        runtime::GpuKernel                  kernel;
    };

    struct Entry {
        KernelSlot                vec;      // M==1 hot path (matvec)
        std::optional<KernelSlot> gemm;     // optional M>1 batched path
        std::size_t               gemmMTile{1};

        // M8.K.1 experimental — Q8_0 only. Set when the v2 GEMM
        // (larger M-tile + sub-group scale broadcast) module loaded.
        // At dispatch time, `useGemmV2` (true only when
        // features.gemmV2 is true AND autotune picked GEMM)
        // routes to v2's kernel instead of v1's.
        std::optional<KernelSlot> gemmV2;
        std::size_t               gemmV2MTile{1};
        bool                      useGemmV2{false};
        std::array<double, kAutotuneBucketCount> gemmV2MsAtM{};

        // M8.J — smallest batch size M at which the timed GEMM bench
        // beat the timed matvec-loop bench with a 5 % margin.
        // matmulAsync dispatches GEMM iff `M >= gemmMinM`. The MAX
        // sentinel means "GEMM never wins, always take matvec".
        //   No GEMM kernel loaded    → MAX
        //   features.gemm=disable    → MAX
        //   features.gemm=force      → 2 (any M > 1 dispatches GEMM)
        //   features.gemmMinM = N    → N (debug override)
        //   Otherwise                → smallest bench-bucket-M where GEMM won,
        //                              or MAX if it never won at any bucket
        std::size_t               gemmMinM{std::numeric_limits<std::size_t>::max()};

        // M8.H.3 — DP4A kernel for this type. Loaded eagerly in the
        // ctor for any type with a `<name>_vec_dp4a.cl` module (Q8_0
        // since M8.H.1, Q4_K since M8.M). nullopt when the SPV or the
        // integer_dot_product extension is unavailable on the target
        // iGPU. Argument list is (Xq int8, Xscale fp32, W quantised,
        // Y fp32, K int32, N int32) — same across all DP4A kernels so
        // the dispatch code is type-agnostic.
        std::optional<KernelSlot> dp4a;

        // When true, matmulAsync routes the dispatch through the DP4A
        // path (x_quant_i8 + DP4A matvec) instead of vec/gemm. Set by
        // autotune() only when parity gate + timing bench both prefer
        // DP4A on this iGPU. Takes priority over the M-threshold GEMM
        // path.
        bool                      useDp4a{false};

        // Autotune telemetry — populated once by autotune(), consumed by
        // autotuneReport() for the /v1/system/status endpoint. Index
        // matches kAutotuneMBuckets in GpuMatmul.cpp; 0 = M=16, etc.
        // 0.0 means "not measured" (env pin, no gemm, parity fail).
        std::array<double, kAutotuneBucketCount> vecMsAtM{};
        std::array<double, kAutotuneBucketCount> gemmMsAtM{};
        double      lastDp4aMs{0.0};      // 0 unless DP4A available
        std::string autotuneSource;       // "bench" | "env_force_gemm" | ...
    };

    core::l0::L0Context&    _ctx;
    GpuOps&                _ops;
    core::l0::UsmAllocator& _alloc;
    runtime::CommandQueue& _queue;

    // One Entry per GgmlType that has a `gpuMatmulModule()` registered.
    // Populated at construction by iterating the QuantType registry.
    std::unordered_map<core::gguf::GgmlType, Entry> _entries;

    // DP4A kernel slots now live inside each Entry (Entry::dp4a) so
    // Q4_K, Q8_0, and any future int-dot-product quant share one code
    // path. The engine-side scratch (_dp4aXqUsm / _dp4aScaleUsm) is
    // sized once and reused across every DP4A dispatch regardless of
    // type — same int8 activation layout.

    // M8.H.3 — persistent Xq / Xscale scratch for the DP4A dispatch
    // path. Sized once at ctor for the worst-case shape we're willing
    // to serve without falling back to plain matvec. Reused across
    // every DP4A call; the engine serialises calls via engineMutex so
    // no aliasing is possible.
    //
    // 8192 × 2816 int8 = 23 MiB peak — trivially small vs the 24 GiB
    // Xe-LPG DRAM budget. If a request comes in with M > kDp4aMaxM or
    // K > kDp4aMaxK, matmulAsync logs once and falls back to vec/gemm.
    void*       _dp4aXqUsm{nullptr};
    void*       _dp4aScaleUsm{nullptr};
    std::size_t _dp4aXqBytes{0};
    std::size_t _dp4aScaleBytes{0};
    // One-shot warning latch — never log the fallback warning twice.
    mutable bool _dp4aScratchOverflowWarned{false};

    // M-MoE.Fused-Decode — one fused-K down-projection kernel per
    // supported expert weight quant type. Populated at ctor by trying to
    // load each variant; kernel-load failures leave the type absent from
    // the map and callers fall back to the sequential per-expert path via
    // `moeDownFusedKAvailable(type)`. Gemma 4 26B-A4B mixes types
    // (gate_up=Q6_K, ffn_down=Q8_0), so both are loaded eagerly.
    std::unordered_map<core::gguf::GgmlType, KernelSlot> _moeDownFusedK;

    // M5h: workgroup of 64 threads = 4 subgroups of 16, each subgroup
    // co-computes ONE output via sub_group_reduce_add. So a workgroup
    // emits 4 outputs and we need ceil(N/4) workgroups. Keep in sync
    // with the kernel macros `MATMUL_*_LOCAL` / `MATMUL_*_SG`.
    static constexpr std::uint32_t kLocalSize        = 64;
    static constexpr std::uint32_t kSubgroupSize     = 16;
    static constexpr std::uint32_t kOutputsPerGroup  = kLocalSize / kSubgroupSize;

    // Must match MATMUL_Q8_0_DP4A_LOCAL / _SG in
    // kernels/matmul_q8_0_vec_dp4a.cl. SG=16 matches Xe-LPG's preferred
    // subgroup width — the M8.H.1 SG=8 layout benched 30 % slower than
    // plain matvec at bench-time; SG=16 with 2 blocks per iteration
    // keeps all lanes busy and matches the geometry of matmul_q8_0_vec.
    // 4 outputs per workgroup.
    static constexpr std::uint32_t kDp4aLocalSize       = 64;
    static constexpr std::uint32_t kDp4aSubgroupSize    = 16;
    static constexpr std::uint32_t kDp4aOutputsPerGroup =
        kDp4aLocalSize / kDp4aSubgroupSize;

    // M8.K.1 v2 GEMM — must match MATMUL_Q8_0_GEMM_V2_M_TILE in
    // kernels/matmul_q8_0_gemm_v2.cl. The first attempt (M_TILE=16)
    // lost 2.35× to v1 on Xe-LPG (SLM overflow → occupancy collapse).
    // The revised v2 keeps M_TILE=8 and shrinks X_TILE_ELEMENTS to
    // 256 so SLM drops to 8 KiB/WG (v1: 32 KiB/WG) — 4× more resident
    // WGs on the Xe-LPG scheduler.
    static constexpr std::size_t kGemmV2MTile = 8;

    // Worst-case shape the internal DP4A scratch is sized for. Anything
    // beyond falls back to vec/gemm at dispatch time with a one-shot log.
    // 8192 = max context we serve. 2816 = Gemma 4 d_model with headroom
    // for future models up to that dim.
    static constexpr std::size_t   kDp4aMaxM           = 8192;
    static constexpr std::size_t   kDp4aMaxK           = 2816;
};

} // namespace mimirmind::compute::l0