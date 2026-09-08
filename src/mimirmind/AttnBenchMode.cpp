// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "mimirmind/AttnBenchMode.hpp"

#include "mimirmind/CliArgs.hpp"
#include "core/config/Config.hpp"
#include "core/gpu/cuda/CudaComputeContext.hpp"
#include "compute/cuda/GpuOps.hpp"
#include "compute/ComputeBuffer.hpp"
#include "runtime/KvCache.hpp"   // runtime::KvDtype
#include "core/log/Log.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

namespace mimirmind::cli {

int runAttnBench(const CliArgs& args, const ::mimirmind::core::config::Config& cfg) {
    (void)args;
    std::printf("\n[attnbench] prefill-attention microbench "
                "(no model load; synthetic contiguous Q/K/V)\n");

    core::cuda::CudaComputeContext ctx;   // auto-select device (GB10)
    compute::cuda::GpuOps ops(ctx,
                              cfg.features.flashPrefill,
                              cfg.features.flashPrefillGqaQ8,
                              cfg.features.flashPrefillGqaQ8Bq,
                              cfg.features.flashPrefillKTileQ8,
                              cfg.features.q8_0Reorder);

    // qwen3-coder-next full-attention layer shape.
    constexpr std::size_t nHeads   = 16;
    constexpr std::size_t nKvHeads  = 2;
    constexpr std::size_t headDim  = 256;
    const float scale = 1.0F / std::sqrt(static_cast<float>(headDim));

    std::printf("[attnbench] shape nHeads=%zu nKvHeads=%zu headDim=%zu kv=F32 "
                "| kernel selected by env (MIMIRMIND_ATTN_CUDNN / "
                "MIMIRMIND_ATTN_F32_MWTC)\n", nHeads, nKvHeads, headDim);

    const std::vector<std::size_t> Ts = {1024, 2048, 4096, 8192, 16384, 22714};
    for (std::size_t T : Ts) {
        const std::size_t qElems = T * nHeads * headDim;
        const std::size_t kvElems = T * nKvHeads * headDim;
        compute::ComputeBuffer q = ops.allocate(qElems * sizeof(float));
        compute::ComputeBuffer k = ops.allocate(kvElems * sizeof(float));
        compute::ComputeBuffer v = ops.allocate(kvElems * sizeof(float));
        compute::ComputeBuffer o = ops.allocate(qElems * sizeof(float));

        auto call = [&]() {
            // Public entry; for T_q==T_k>1 with positionOffset 0 it dispatches
            // to the prefill-flash path (cuDNN / MWTC / hand kernel per env).
            ops.attentionAsync(
                q.as<float>(), k.get(), v.get(), /*T_q=*/T, /*T_k=*/T,
                nHeads, nKvHeads, headDim, /*positionOffset=*/0, scale,
                o.as<float>(), /*slidingWindow=*/0, runtime::KvDtype::F32);
        };

        call();          // warmup (kernel autotune / smem opt-in resolves here)
        ops.flush();

        constexpr int N = 3;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < N; ++i) { call(); }
        ops.flush();
        const auto t1 = std::chrono::steady_clock::now();
        const double ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count() / N;
        std::printf("[attnbench]   T=%6zu  attentionPrefillFlash = %9.2f ms\n",
                    T, ms);
        std::fflush(stdout);
    }
    // Asymmetric sweep — the SERVING chunked-prefill pattern: a small query
    // chunk (T_q) attending a large cached KV range (T_kv = positionOffset+T_q,
    // bottom-right causal). If this stays fast where the symmetric sweep hit a
    // cliff, the fix is "keep T_q small (chunk) + cuDNN"; if it also cliffs, the
    // cliff is T_kv-driven and chunking cannot rescue it.
    std::printf("[attnbench] --- asymmetric (T_q=512, T_kv growing) — serving "
                "chunk pattern ---\n");
    constexpr std::size_t Tq = 512;
    for (std::size_t Tkv : {std::size_t{2048}, std::size_t{8192},
                            std::size_t{16384}, std::size_t{22714}}) {
        const std::size_t posOff = Tkv - Tq;
        compute::ComputeBuffer q = ops.allocate(Tq * nHeads * headDim * sizeof(float));
        compute::ComputeBuffer k = ops.allocate(Tkv * nKvHeads * headDim * sizeof(float));
        compute::ComputeBuffer v = ops.allocate(Tkv * nKvHeads * headDim * sizeof(float));
        compute::ComputeBuffer o = ops.allocate(Tq * nHeads * headDim * sizeof(float));
        auto call = [&]() {
            ops.attentionAsync(q.as<float>(), k.get(), v.get(), /*T_q=*/Tq,
                               /*T_k=*/Tkv, nHeads, nKvHeads, headDim,
                               /*positionOffset=*/posOff, scale, o.as<float>(),
                               /*slidingWindow=*/0, runtime::KvDtype::F32);
        };
        call();
        ops.flush();
        constexpr int N = 3;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < N; ++i) { call(); }
        ops.flush();
        const auto t1 = std::chrono::steady_clock::now();
        const double ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count() / N;
        std::printf("[attnbench]   T_q=512 T_kv=%6zu = %9.2f ms\n", Tkv, ms);
        std::fflush(stdout);
    }
    std::printf("[attnbench] done\n");
    return 0;
}

} // namespace mimirmind::cli
