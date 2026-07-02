#pragma once

#include "model/GgufTypes.hpp"
#include "runtime/CommandQueue.hpp"
#include "runtime/GpuKernel.hpp"
#include "runtime/GpuModule.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>

namespace mimirmind::runtime {
class L0Context;
class UsmAllocator;
}

namespace mimirmind::compute {

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
 */
class GpuMatmul {
public:
    GpuMatmul(runtime::L0Context& ctx, runtime::CommandQueue& queue);
    ~GpuMatmul() = default;

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
        bool        gemmPicked;          // autotune picked GEMM over matvec-loop
        double      vecMs;               // measured matvec-loop median ms (0 = not measured)
        double      gemmMs;              // measured GEMM median ms (0 = not measured)
        std::string source;              // "bench" | "env_force_gemm" | "env_disable_gemm" | "no_gemm" | "parity_fail"
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
        bool                      useGemm{false};  // set by autotune()

        // Autotune telemetry — populated once by autotune(), consumed by
        // autotuneReport() for the /v1/system/status endpoint.
        double      lastVecMs{0.0};
        double      lastGemmMs{0.0};
        std::string autotuneSource;   // "bench" | "env_force_gemm" | ...
    };

    runtime::L0Context&    _ctx;
    runtime::CommandQueue& _queue;

    // One Entry per GgmlType that has a `gpuMatmulModule()` registered.
    // Populated at construction by iterating the QuantType registry.
    std::unordered_map<model::GgmlType, Entry> _entries;

    // M8.H.1 — DP4A Q8_0 matvec kernel. Loaded eagerly in the ctor;
    // nullopt when the SPV or the DP4A extension itself is unavailable
    // on the current iGPU. Kept separate from `_entries` because it
    // takes a different argument list (Xq + Xscale) so the generic
    // matmulAsync dispatcher can't drive it — the autotune integration
    // (M8.H.3) is what will pick it against MATVEC / GEMM per QuantType.
    std::optional<KernelSlot> _q8_0Dp4aSlot;

    // M5h: workgroup of 64 threads = 4 subgroups of 16, each subgroup
    // co-computes ONE output via sub_group_reduce_add. So a workgroup
    // emits 4 outputs and we need ceil(N/4) workgroups. Keep in sync
    // with the kernel macros `MATMUL_*_LOCAL` / `MATMUL_*_SG`.
    static constexpr std::uint32_t kLocalSize        = 64;
    static constexpr std::uint32_t kSubgroupSize     = 16;
    static constexpr std::uint32_t kOutputsPerGroup  = kLocalSize / kSubgroupSize;

    // Must match MATMUL_Q8_0_DP4A_LOCAL / _SG in
    // kernels/matmul_q8_0_vec_dp4a.cl. SG=8 fits the 8 char4 chunks per
    // Q8_0 block with no idle lanes; 8 outputs per workgroup.
    static constexpr std::uint32_t kDp4aLocalSize       = 64;
    static constexpr std::uint32_t kDp4aSubgroupSize    = 8;
    static constexpr std::uint32_t kDp4aOutputsPerGroup =
        kDp4aLocalSize / kDp4aSubgroupSize;
};

} // namespace mimirmind::compute