// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "mimirmind/GdnBenchMode.hpp"

#include "mimirmind/CliArgs.hpp"
#include "core/config/Config.hpp"
#include "core/gpu/cuda/CudaComputeContext.hpp"
#include "compute/cuda/GpuOps.hpp"
#include "compute/ComputeBuffer.hpp"
#include "compute/ComputeOps.hpp"   // GdnBatchedShape

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

namespace mimirmind::cli {

int runGdnBench(const CliArgs& args, const ::mimirmind::core::config::Config& cfg) {
    (void)args;
    std::printf("\n[gdnbench] GatedDeltaNet chunk-forward (gdn.k2) microbench "
                "(no model load; synthetic q/k/v/gCum/beta/a0/state)\n");

    core::cuda::CudaComputeContext ctx;   // auto-select device (GB10)
    compute::cuda::GpuOps ops(ctx,
                              cfg.features.flashPrefill,
                              cfg.features.flashPrefillGqaQ8,
                              cfg.features.flashPrefillGqaQ8Bq,
                              cfg.features.flashPrefillKTileQ8,
                              cfg.features.q8_0Reorder);

    // Prod qwen3.6 GDN shape: 32 value heads, state/head dim 128, chunk 64.
    constexpr std::size_t nSeq = 1;
    constexpr std::size_t H    = 32;
    constexpr std::size_t S    = 128;
    constexpr std::size_t C    = 64;
    constexpr std::size_t kGdnChunkFwdWorkers = 96;   // ComputeOps.hpp cap

    std::printf("[gdnbench] shape nSeq=%zu H=%zu S=%zu C=%zu | TC path needs "
                "MIMIRMIND_GDN_CHUNK_TC=1 (S==128 && C==64)\n", nSeq, H, S, C);

    // Per-(chunk,head) GEMM flops (2 flops/MAC): step2 U/UQ = 2*C*S*S each,
    // step2.5 KQ = C*C*S, step4 D = C*S*C, step5 W@D = C*S*C, step6 = S*S*C.
    const double flopsPerChunkHead =
        2.0 * (2.0 * C * S * S + /*KQ*/ (double)C * C * S
               + /*D*/ (double)C * S * C + /*W@D*/ (double)C * S * C
               + /*step6*/ (double)S * S * C);

    const std::vector<std::size_t> Ts = {512, 1024, 2048, 4096};
    for (std::size_t T : Ts) {
        const std::size_t maxChunks = (T + C - 1) / C;
        compute::ComputeBuffer q     = ops.allocate(T * H * S * sizeof(float));
        compute::ComputeBuffer k     = ops.allocate(T * H * S * sizeof(float));
        compute::ComputeBuffer v     = ops.allocate(T * H * S * sizeof(float));
        compute::ComputeBuffer gCum  = ops.allocate(T * H * sizeof(float));
        compute::ComputeBuffer beta  = ops.allocate(T * H * sizeof(float));
        compute::ComputeBuffer a0    = ops.allocate(maxChunks * H * C * C * sizeof(float));
        compute::ComputeBuffer state = ops.allocate(nSeq * H * S * S * sizeof(float));
        compute::ComputeBuffer out   = ops.allocate(T * H * S * sizeof(float));
        compute::ComputeBuffer scr   = ops.allocate(kGdnChunkFwdWorkers * 7 * C * S * sizeof(float));

        compute::GdnBatchedShape shape;
        shape.nSeq = nSeq;
        shape.T    = T;
        shape.H    = H;
        shape.S    = S;   // seqT/seqOff nullptr => uniform layout

        auto call = [&]() {
            ops.deltanetChunkForwardBatchedAsync(
                q.as<float>(), k.as<float>(), v.as<float>(),
                gCum.as<float>(), beta.as<float>(), a0.as<float>(),
                state.as<float>(), out.as<float>(), scr.as<float>(),
                shape, C);
        };

        call();          // warmup
        ops.flush();

        constexpr int N = 5;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < N; ++i) { call(); }
        ops.flush();
        const auto t1 = std::chrono::steady_clock::now();
        const double ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count() / N;

        const double flops = flopsPerChunkHead * (double)H * (double)maxChunks;
        const double tflops = flops / (ms * 1.0e-3) / 1.0e12;
        // GB10 bf16 dense tensor-core peak ~ 250 TFLOP/s (order-of-magnitude ref).
        const double pctPeak = tflops / 250.0 * 100.0;
        std::printf("[gdnbench]   T=%5zu chunks=%3zu  %8.3f ms | "
                    "%7.2f TFLOP/s (bf16-TC) = ~%5.1f%% of ~250TF peak\n",
                    T, maxChunks, ms, tflops, pctPeak);
        std::fflush(stdout);
    }
    std::printf("[gdnbench] interpretation: <<10%% of peak => load/latency-bound "
                "(cp.async/CuTe smem-pipeline is the lever); >>40%% => compute-bound "
                "(restructure won't help).\n");

    // 5.18.22 — nSeq (concurrency) sweep at fixed T. G = min(96, nSeq*H); at
    // nSeq=1 the grid is 32 blocks (<48 SMs, 0.67 waves per ncu). This shows
    // whether concurrent serving (nSeq>1) already fills the machine => the
    // single-user column-split (v5) only matters if RAG runs nSeq==1.
    std::printf("[gdnbench] --- nSeq (concurrency) sweep at T=2048 "
                "(G = min(96, nSeq*%zu); 48 SMs) ---\n", H);
    constexpr std::size_t Tf = 2048;
    const std::size_t maxChunksF = (Tf + C - 1) / C;
    for (std::size_t nS : {std::size_t{1}, std::size_t{2}, std::size_t{3},
                           std::size_t{4}, std::size_t{8}}) {
        compute::ComputeBuffer q     = ops.allocate(nS * Tf * H * S * sizeof(float));
        compute::ComputeBuffer k     = ops.allocate(nS * Tf * H * S * sizeof(float));
        compute::ComputeBuffer v     = ops.allocate(nS * Tf * H * S * sizeof(float));
        compute::ComputeBuffer gCum  = ops.allocate(nS * Tf * H * sizeof(float));
        compute::ComputeBuffer beta  = ops.allocate(nS * Tf * H * sizeof(float));
        compute::ComputeBuffer a0    = ops.allocate(nS * maxChunksF * H * C * C * sizeof(float));
        compute::ComputeBuffer state = ops.allocate(nS * H * S * S * sizeof(float));
        compute::ComputeBuffer out   = ops.allocate(nS * Tf * H * S * sizeof(float));
        compute::ComputeBuffer scr   = ops.allocate(kGdnChunkFwdWorkers * 7 * C * S * sizeof(float));
        compute::GdnBatchedShape shape;
        shape.nSeq = nS; shape.T = Tf; shape.H = H; shape.S = S;
        auto call = [&]() {
            ops.deltanetChunkForwardBatchedAsync(
                q.as<float>(), k.as<float>(), v.as<float>(),
                gCum.as<float>(), beta.as<float>(), a0.as<float>(),
                state.as<float>(), out.as<float>(), scr.as<float>(), shape, C);
        };
        call(); ops.flush();
        constexpr int N = 5;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < N; ++i) { call(); }
        ops.flush();
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / N;
        const std::size_t items = nS * H;
        const std::size_t G = items < 96 ? items : 96;   // kGdnChunkFwdWorkers
        const double flops = flopsPerChunkHead * (double)H * (double)maxChunksF * (double)nS;
        const double tflops = flops / (ms * 1.0e-3) / 1.0e12;
        std::printf("[gdnbench]   nSeq=%zu  G=%3zu blocks (%.2f waves/48SM)  "
                    "%8.3f ms | %7.2f TFLOP/s = ~%5.1f%% peak\n",
                    nS, G, (double)G / 48.0, ms, tflops, tflops / 250.0 * 100.0);
        std::fflush(stdout);
    }
    std::printf("[gdnbench] done\n");
    return 0;
}

} // namespace mimirmind::cli
