// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "compute/ComputeMatmul.hpp"

namespace mimirmind::core::cpu {
class CpuContext;
}

namespace mimirmind::compute::cpu {

/**
 * CPU implementation of `compute::ComputeMatmul`. Every dispatch runs
 * through `compute::matmul` (see Matmul.hpp) on the calling thread.
 *
 * Skeleton stage (M-CPU.1): `matmul` / `matmulAsync` delegate to the
 * reference scalar-loop with double accumulation; `supports(type)` is
 * true for any QuantType registered in `compute::QuantTypeRegistry`
 * (F32, F16, BF16, Q4_K, Q5_K, Q5_0, Q6_K, Q8_0). DP4A and MoE
 * fused-K paths throw NotImplemented — those are backend-specific
 * kernel-fusion tricks with no CPU analogue that's worth writing
 * before we have a real perf target to hit.
 *
 * `sync` is a no-op — every method has already returned when the
 * caller sees it, so there's no pending work to flush. `autotuneReport`
 * returns an empty vector; nothing on CPU is worth benching.
 *
 * Takes `CpuContext&` for symmetry with the L0 / HIP GpuMatmul; the
 * reference itself is not read today.
 *
 * Not thread-safe. Construct once at startup.
 */
class GpuMatmul : public ::mimirmind::compute::ComputeMatmul {
public:
    explicit GpuMatmul(::mimirmind::core::cpu::CpuContext& ctx);
    ~GpuMatmul() override = default;

    GpuMatmul(const GpuMatmul&)            = delete;
    GpuMatmul& operator=(const GpuMatmul&) = delete;
    GpuMatmul(GpuMatmul&&)                 = delete;
    GpuMatmul& operator=(GpuMatmul&&)      = delete;

    [[nodiscard]] bool supports(::mimirmind::core::gguf::GgmlType type)
        const noexcept override;

    void matmul(::mimirmind::core::gguf::GgmlType type,
                const void*  W,
                std::size_t  N,
                std::size_t  K,
                const float* X,
                std::size_t  M,
                float*       Y,
                float*       scratch) override;

    void matmulAsync(::mimirmind::core::gguf::GgmlType type,
                     const void*  W,
                     std::size_t  N,
                     std::size_t  K,
                     const float* X,
                     std::size_t  M,
                     float*       Y,
                     float*       scratch) override;

    void matmulDp4aAsync(::mimirmind::core::gguf::GgmlType type,
                         const std::int8_t* Xq,
                         const float*       Xscale,
                         const void*        W,
                         std::size_t        N,
                         std::size_t        K,
                         std::size_t        M,
                         float*             Y) override;

    [[nodiscard]] bool dp4aAvailable() const noexcept override { return false; }
    [[nodiscard]] bool dp4aAvailable(::mimirmind::core::gguf::GgmlType /*type*/)
        const noexcept override { return false; }

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

    [[nodiscard]] bool moeDownFusedKAvailable() const noexcept override { return false; }
    [[nodiscard]] bool moeDownFusedKAvailable(::mimirmind::core::gguf::GgmlType /*type*/)
        const noexcept override { return false; }

    void sync() override {}

    [[nodiscard]] std::vector<::mimirmind::compute::AutotuneReport>
        autotuneReport() const override { return {}; }

private:
    ::mimirmind::core::cpu::CpuContext& _ctx;
};

} // namespace mimirmind::compute::cpu