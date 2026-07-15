// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "compute/ComputeMatmul.hpp"
#include "core/gguf/GgufTypes.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>

namespace mimirmind::core::hip {
class HipComputeContext;
}

namespace mimirmind::compute::hip {

class GpuOps;

/**
 * HIP/ROCm implementation of the backend-neutral `compute::ComputeMatmul`
 * interface. Parallel to the Level-Zero `compute::l0::GpuMatmul` — same
 * public surface (11 virtuals from `ComputeMatmul`), but every kernel
 * launch goes through `HipModule` + `HipKernel` + `HipStream` instead
 * of L0 modules / command queues.
 *
 * Same class name (`GpuMatmul`) as the L0 side; disambiguation happens
 * through the `compute::hip::` namespace vs `compute::l0::`, mirroring
 * the `src/core/gpu/{l0,hip}/` primitive layout and the `GpuOps` split.
 *
 * Skeleton stage (sub-F): ctor loads the 5 Q8_0 matmul kernel modules
 * (`matmul_q8_0_vec`, `_gemm`, `_gemm_v2`, `_vec_dp4a`,
 * `moe_down_fused_k_q8_0`) that already exist under `kernels_hip/`.
 * `supports()`, `dp4aAvailable()`, `moeDownFusedKAvailable()`, `sync()`
 * and `autotuneReport()` are all real. Every matmul-launch method
 * currently throws `std::runtime_error("compute::hip::GpuMatmul::<name>:
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
    /// Ctor takes `HipComputeContext&` (matches the l0 side's
    /// `L0ComputeContext&` pattern) plus a `GpuOps&` because the
    /// DP4A path shares `xQuantI8Async` with the elementwise kernels
    /// living on GpuOps.
    GpuMatmul(::mimirmind::core::hip::HipComputeContext& ctx, GpuOps& ops);
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

    void sync() override;

    [[nodiscard]] std::vector<::mimirmind::compute::AutotuneReport>
        autotuneReport() const override;

private:
    ::mimirmind::core::hip::HipComputeContext& _ctx;
    GpuOps&                                    _ops;

    struct Impl;
    std::unique_ptr<Impl>                      _pimpl;

    // Kernel-launch geometry constants (must stay in sync with the
    // MATMUL_Q8_0_* macros in the .hip source files).
    static constexpr std::uint32_t kLocalSize        = 64;
    static constexpr std::uint32_t kSubgroupSize     = 16;
    static constexpr std::uint32_t kOutputsPerGroup  = kLocalSize / kSubgroupSize;

    static constexpr std::uint32_t kDp4aLocalSize    = 64;
    static constexpr std::uint32_t kDp4aSubgroupSize = 16;
    static constexpr std::uint32_t kDp4aOutputsPerGroup =
        kDp4aLocalSize / kDp4aSubgroupSize;

    static constexpr std::size_t   kGemmV2MTile      = 8;

    // MoE fused-K down-projection: same 4-outputs-per-WG geometry as
    // the plain vec kernel by coincidence (MOE_DOWN_LOCAL=64, SG=16 in
    // moe_down_fused_k_q8_0.hip). Declared separately so a future
    // architecture-specific tuning of one doesn't drag the other.
    static constexpr std::uint32_t kMoeDownLocalSize      = 64;
    static constexpr std::uint32_t kMoeDownOutputsPerGroup = 4;
};

} // namespace mimirmind::compute::hip