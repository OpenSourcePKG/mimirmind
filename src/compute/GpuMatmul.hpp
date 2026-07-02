#pragma once

#include "model/GgufTypes.hpp"
#include "runtime/CommandQueue.hpp"
#include "runtime/GpuKernel.hpp"
#include "runtime/GpuModule.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace mimirmind::runtime {
class L0Context;
class UsmAllocator;
}

namespace mimirmind::compute {

class GpuOps;

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
class GpuMatmul {
public:
    GpuMatmul(runtime::L0Context&    ctx,
              GpuOps&                ops,
              runtime::UsmAllocator& alloc,
              runtime::CommandQueue& queue);
    ~GpuMatmul();

    GpuMatmul(const GpuMatmul&)            = delete;
    GpuMatmul& operator=(const GpuMatmul&) = delete;
    GpuMatmul(GpuMatmul&&)                 = delete;
    GpuMatmul& operator=(GpuMatmul&&)      = delete;

    /// True if this dispatcher will run `type` on the GPU.
    [[nodiscard]] bool supports(model::GgmlType type) const noexcept;

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
    /// Env-var overrides skip the actual bench:
    ///   MIMIRMIND_DISABLE_GEMM=1 → force matvec-loop for every type
    ///   MIMIRMIND_FORCE_GEMM=1   → force GEMM for every type that has one
    /// Both set → DISABLE wins (safer default).
    void autotune(runtime::UsmAllocator& allocator,
                  std::size_t            hiddenDim,
                  std::size_t            mBatch = 16);

    /// Y [M, N] = X [M, K] * W [N, K]^T. Synchronous version (mirrors
    /// compute::matmul signature). For supported `type` the dispatch goes
    /// to the GPU kernel and we sync immediately. For unsupported types
    /// the call falls back to CPU.
    ///
    /// `scratch` (K floats) is only consumed on the CPU fallback path.
    void matmul(model::GgmlType type,
                const void*     W,
                std::size_t     N,
                std::size_t     K,
                const float*    X,
                std::size_t     M,
                float*          Y,
                float*          scratch);

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
    void matmulAsync(model::GgmlType type,
                     const void*     W,
                     std::size_t     N,
                     std::size_t     K,
                     const float*    X,
                     std::size_t     M,
                     float*          Y,
                     float*          scratch);

    /// M8.H.1 — DP4A Q8_0 matvec with pre-quantised int8 activation.
    /// Y [M, N] = Xq [M, K] (int8) × Xscale [M] × W [N, K]^T (Q8_0).
    /// The activation is expected to already have been produced by
    /// GpuOps::xQuantI8Async (per-row symmetric int8, scale = amax/127).
    /// Async — call sync() before reading Y from the CPU.
    ///
    /// Available only when the DP4A extension resolved at module load;
    /// query via dp4aAvailable() first. Throws if unavailable.
    void matmulQ8_0Dp4aAsync(const std::int8_t* Xq,
                             const float*       Xscale,
                             const void*        W,
                             std::size_t        N,
                             std::size_t        K,
                             std::size_t        M,
                             float*             Y);

private:
    /// Internal dispatch for the DP4A Q8_0 path — quantises `X` into
    /// the persistent Xq/Xscale scratch via `_ops.xQuantI8Async`, then
    /// forwards to `matmulQ8_0Dp4aAsync`. Callers must first verify
    /// M <= kDp4aMaxM && K <= kDp4aMaxK.
    void dispatchQ8_0Dp4aFromFloat(const float* X,
                                   const void*  W,
                                   std::size_t  N,
                                   std::size_t  K,
                                   std::size_t  M,
                                   float*       Y);
public:

    /// True when matmul_q8_0_vec_dp4a loaded successfully. False when
    /// the SPV or the DP4A extension itself was unavailable on the
    /// target iGPU; callers must fall back to the plain matvec path.
    [[nodiscard]] bool dp4aAvailable() const noexcept {
        return _q8_0Dp4aSlot.has_value();
    }

    /// Flush any pending appends (close + execute + sync + reset). Safe
    /// to call when there's no pending work — cheap no-op.
    void sync();

    /// Report the autotune decision + timings per QuantType. Rows are
    /// empty until autotune() has run. Used by ApiServer to expose the
    /// kernel-dispatch choice via /v1/system/status.
    struct AutotuneReport {
        std::string name;                // e.g. "Q6_K"
        bool        gemmAvailable;       // GEMM kernel loaded for this type
        bool        gemmPicked;          // GEMM will be dispatched for at least some M (i.e., gemmMinM != MAX)
        // Legacy fields — the M=16 timing, kept so downstream consumers
        // don't break during M8.J rollout.
        double      vecMs;
        double      gemmMs;
        // M8.J — per-M-bucket timings + threshold. Buckets match
        // kAutotuneMBuckets in GpuMatmul.cpp (16, 64, 256).
        std::array<std::size_t, 3> mBuckets{};
        std::array<double, 3>      vecMsAtM{};
        std::array<double, 3>      gemmMsAtM{};
        std::size_t                gemmMinM{0};  // 0 = "not set" (rendered as null in JSON)
        // M8.H.3 — only meaningful for Q8_0 on iGPUs where the DP4A
        // module loaded. Zero elsewhere.
        bool        dp4aAvailable{false};
        bool        dp4aPicked{false};
        double      dp4aMs{0.0};
        // M8.K.1 — v2 GEMM prototype (M_TILE=16) telemetry. Q8_0 only.
        bool        gemmV2Available{false};
        bool        gemmV2Picked{false};
        std::array<double, 3> gemmV2MsAtM{};
        std::string source;              // "bench" | "env_force_gemm" | "env_disable_gemm" | "no_gemm" | "parity_fail" | "dp4a_parity_fail" | "env_force_dp4a" | "env_gemm_min_m"
    };
    [[nodiscard]] std::vector<AutotuneReport> autotuneReport() const;

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
        // MIMIRMIND_USE_GEMM_V2 is set AND autotune picked GEMM)
        // routes to v2's kernel instead of v1's.
        std::optional<KernelSlot> gemmV2;
        std::size_t               gemmV2MTile{1};
        bool                      useGemmV2{false};
        std::array<double, 3>     gemmV2MsAtM{};

        // M8.J — smallest batch size M at which the timed GEMM bench
        // beat the timed matvec-loop bench with a 5 % margin.
        // matmulAsync dispatches GEMM iff `M >= gemmMinM`. The MAX
        // sentinel means "GEMM never wins, always take matvec".
        //   No GEMM kernel loaded → MAX
        //   MIMIRMIND_DISABLE_GEMM → MAX
        //   MIMIRMIND_FORCE_GEMM   → 2 (any M > 1 dispatches GEMM)
        //   MIMIRMIND_GEMM_MIN_M=N → N (debug override)
        //   Otherwise              → smallest bench-bucket-M where GEMM won,
        //                             or MAX if it never won at any bucket
        std::size_t               gemmMinM{std::numeric_limits<std::size_t>::max()};

        // M8.H.3 — Q8_0 only. When true, matmulAsync routes the Q8_0
        // dispatch through the DP4A path (x_quant_i8 + DP4A matvec)
        // instead of vec/gemm. Set by autotune() only when parity gate
        // + timing bench both prefer DP4A on this iGPU. Takes priority
        // over the M-threshold GEMM path.
        bool                      useDp4a{false};

        // Autotune telemetry — populated once by autotune(), consumed by
        // autotuneReport() for the /v1/system/status endpoint. Index
        // matches kAutotuneMBuckets in GpuMatmul.cpp; 0 = M=16, etc.
        // 0.0 means "not measured" (env pin, no gemm, parity fail).
        std::array<double, 3> vecMsAtM{};
        std::array<double, 3> gemmMsAtM{};
        double      lastDp4aMs{0.0};      // 0 unless DP4A available
        std::string autotuneSource;       // "bench" | "env_force_gemm" | ...
    };

    runtime::L0Context&    _ctx;
    GpuOps&                _ops;
    runtime::UsmAllocator& _alloc;
    runtime::CommandQueue& _queue;

    // One Entry per GgmlType that has a `gpuMatmulModule()` registered.
    // Populated at construction by iterating the QuantType registry.
    std::unordered_map<model::GgmlType, Entry> _entries;

    // M8.H.1 — DP4A Q8_0 matvec kernel. Loaded eagerly in the ctor;
    // nullopt when the SPV or the DP4A extension itself is unavailable
    // on the current iGPU. Kept separate from `_entries` because it
    // takes a different argument list (Xq + Xscale) so the generic
    // matmulAsync dispatcher can't drive it directly — the autotune
    // integration (M8.H.3) selects it, and matmulAsync then routes
    // through _dp4aXqUsm / _dp4aScaleUsm.
    std::optional<KernelSlot> _q8_0Dp4aSlot;

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

} // namespace mimirmind::compute