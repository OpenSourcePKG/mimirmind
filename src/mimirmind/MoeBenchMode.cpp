// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "mimirmind/MoeBenchMode.hpp"

#include "mimirmind/CliArgs.hpp"
#include "core/config/Config.hpp"
#include "core/gpu/cuda/CudaComputeContext.hpp"
#include "compute/cuda/GpuOps.hpp"
#include "compute/ComputeBuffer.hpp"
#include "compute/ComputeOps.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace mimirmind::cli {

// 5.28.2 decode MoE-GEMM microbench. NVFP4_BLK geom = {32 elems, 20 bytes/super}
// (moeBlockGeom, Qwen3_5MoeBackend.cpp). Representative qwen3.6-35B-A3B decode
// dims; the bandwidth-vs-peak conclusion is robust to the exact n_ff.
int runMoeBench(const CliArgs& args, const ::mimirmind::core::config::Config& cfg) {
    (void)args;
    std::printf("\n[moebench] decode MoE-GEMM microbench (5.28.2 de-risk; no model "
                "load; synthetic NVFP4_BLK banks + low-M routing)\n");

    core::cuda::CudaComputeContext ctx;   // auto-select device (GB10)
    compute::cuda::GpuOps ops(ctx,
                              cfg.features.flashPrefill,
                              cfg.features.flashPrefillGqaQ8,
                              cfg.features.flashPrefillGqaQ8Bq,
                              cfg.features.flashPrefillKTileQ8,
                              cfg.features.q8_0Reorder);

    // Representative decode dims (override via env for other models).
    std::size_t d_model  = 2048;
    std::size_t n_ff_exp = 768;
    std::size_t nExperts = 256;
    std::size_t topK     = 8;
    if (const char* e = std::getenv("MOEBENCH_DMODEL"))  d_model  = std::strtoul(e, nullptr, 10);
    if (const char* e = std::getenv("MOEBENCH_NFF"))     n_ff_exp = std::strtoul(e, nullptr, 10);
    if (const char* e = std::getenv("MOEBENCH_NEXP"))    nExperts = std::strtoul(e, nullptr, 10);
    if (const char* e = std::getenv("MOEBENCH_TOPK"))    topK     = std::strtoul(e, nullptr, 10);

    constexpr std::size_t GE = 32, GB = 20;   // NVFP4_BLK super-block geom
    const std::size_t perExpGU = n_ff_exp * ((d_model  / GE) * GB);  // gate/up bank/expert
    const std::size_t perExpDN = d_model  * ((n_ff_exp / GE) * GB);  // down bank/expert
    const std::size_t tileM    = 4;           // m4reg decode schedule (smallM)

    std::printf("[moebench] dims d_model=%zu n_ff_exp=%zu nExperts=%zu topK=%zu "
                "tileM=%zu | bank gate/up=%.1f MiB down=%.1f MiB (all experts)\n",
                d_model, n_ff_exp, nExperts, topK, tileM,
                (double)nExperts * perExpGU / (1024.0 * 1024.0),
                (double)nExperts * perExpDN / (1024.0 * 1024.0));
    std::printf("[moebench] peak ref: 273 GB/s (GB10 LPDDR5x). achieved%% near 100 "
                "=> BANDWIDTH-bound (swap_ab/masked cannot help, NO-GO); <<100 => "
                "compute/latency headroom (swap_ab worth building, GO)\n");

    // Persistent banks (allocated once, deint cache keyed on ptr -> warmup pays it).
    compute::ComputeBuffer gateBank = ops.allocate(nExperts * perExpGU);
    compute::ComputeBuffer downBank = ops.allocate(nExperts * perExpDN);

    // Two regimes: (A) single-user low-M (few active experts -> grid underfill),
    // (B) SERVING conc (ALL 256 experts active, rows/expert = conc*topK/nExperts)
    // — the regime where MoE-GEMM is the 47% decode term. Encode as (label, R,
    // forceAllExperts).
    struct Shape { const char* tag; std::size_t R; bool allExp; };
    std::vector<Shape> shapes;
    for (std::size_t M : {std::size_t{1}, std::size_t{2}, std::size_t{4}, std::size_t{8}})
        shapes.push_back({"single", M * topK, false});
    // serving: conc such that R = rowsPerExpert * nExperts (all experts active).
    for (std::size_t rpe : {std::size_t{1}, std::size_t{2}, std::size_t{4}, std::size_t{8}})
        shapes.push_back({"serve", rpe * nExperts, true});
    // ncu mode: profile ONE representative serve shape (conc64: 2 rows/expert, all
    // 256 experts) with few launches, so ncu reads the deint kernel's DRAM %.
    const bool ncuMode = std::getenv("MOEBENCH_NCU") != nullptr;
    if (ncuMode) shapes = {{"serve", 2 * nExperts, true}};
    const int nIter = ncuMode ? 2 : 20;

    for (const Shape& sh : shapes) {
        const std::size_t R = sh.R;
        const std::size_t activeExp = sh.allExp ? nExperts
                                                : (R < nExperts ? R : nExperts);
        const std::size_t maxTiles = (R + tileM - 1) / tileM + nExperts;

        // Host routing histogram -> expOffset prefix sum. Distribute R rows over
        // `activeExp` experts as evenly as possible (round-robin).
        std::vector<std::int32_t> expOffH(nExperts + 1, 0);
        const std::size_t base = activeExp ? R / activeExp : 0;
        const std::size_t rem  = activeExp ? R % activeExp : 0;
        for (std::size_t e = 0; e < nExperts; ++e) {
            std::size_t cnt = 0;
            if (e < activeExp) cnt = base + (e < rem ? 1 : 0);
            expOffH[e + 1] = expOffH[e] + static_cast<std::int32_t>(cnt);
        }
        compute::ComputeBuffer expOff = ops.allocate((nExperts + 1) * sizeof(std::int32_t));
        ops.uploadHostBytes(expOff.get(), expOffH.data(),
                            (nExperts + 1) * sizeof(std::int32_t));

        compute::ComputeBuffer tileExpert = ops.allocate(maxTiles * sizeof(std::int32_t));
        compute::ComputeBuffer tileRow0   = ops.allocate(maxTiles * sizeof(std::int32_t));
        compute::ComputeBuffer tileRows   = ops.allocate(maxTiles * sizeof(std::int32_t));
        compute::ComputeBuffer tileCount  = ops.allocate(sizeof(std::int32_t));
        ops.moeGroupTilesAsync(expOff.as<std::int32_t>(), tileExpert.as<std::int32_t>(),
                               tileRow0.as<std::int32_t>(), tileRows.as<std::int32_t>(),
                               tileCount.as<std::int32_t>(), nExperts, maxTiles, tileM);

        // I/O buffers. gate: x[R,d_model] -> y[R,n_ff]. down: x[R,n_ff] -> y[R,d_model].
        compute::ComputeBuffer xGate = ops.allocate(R * d_model * sizeof(float));
        compute::ComputeBuffer yGate = ops.allocate(R * n_ff_exp * sizeof(float));
        compute::ComputeBuffer xDown = ops.allocate(R * n_ff_exp * sizeof(float));
        compute::ComputeBuffer yDown = ops.allocate(R * d_model * sizeof(float));

        auto gate = [&]() {
            ops.moeGroupedGemmNvfp4DeintAsync(
                xGate.as<float>(),
                static_cast<const unsigned char*>(gateBank.get()),
                yGate.as<float>(), tileExpert.as<std::int32_t>(),
                tileRow0.as<std::int32_t>(), tileRows.as<std::int32_t>(),
                d_model, n_ff_exp, nExperts, maxTiles, /*decodeSmallM=*/true);
        };
        auto down = [&]() {
            ops.moeGroupedGemmNvfp4DeintAsync(
                xDown.as<float>(),
                static_cast<const unsigned char*>(downBank.get()),
                yDown.as<float>(), tileExpert.as<std::int32_t>(),
                tileRow0.as<std::int32_t>(), tileRows.as<std::int32_t>(),
                n_ff_exp, d_model, nExperts, maxTiles, /*decodeSmallM=*/true);
        };

        gate(); down(); ops.flush();   // warmup (pays the one-time deint of each bank)

        const int N = nIter;
        auto timeIt = [&](auto&& fn) {
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < N; ++i) fn();
            ops.flush();
            const auto t1 = std::chrono::steady_clock::now();
            return std::chrono::duration<double, std::milli>(t1 - t0).count() / N;
        };
        const double msG = timeIt(gate);
        const double msD = timeIt(down);

        // Active-expert weight bytes read (only routed experts' banks touched).
        const double bytesG = (double)activeExp * perExpGU;
        const double bytesD = (double)activeExp * perExpDN;
        const double gbsG = bytesG / (msG * 1.0e-3) / 1.0e9;
        const double gbsD = bytesD / (msD * 1.0e-3) / 1.0e9;
        std::printf("[moebench] %-6s R=%4zu act=%3zu rpe=%zu | gate %7.4f ms %6.1f "
                    "GB/s (~%4.1f%%) | down %7.4f ms %6.1f GB/s (~%4.1f%%)\n",
                    sh.tag, R, activeExp, (activeExp ? R / activeExp : 0),
                    msG, gbsG, gbsG / 273.0 * 100.0,
                    msD, gbsD, gbsD / 273.0 * 100.0);
        std::fflush(stdout);
    }
    std::printf("[moebench] done. GO/NO-GO: near-peak achieved%% at low M confirms "
                "bandwidth-bound => swap_ab NO-GO (5.28.2 spec).\n");
    return 0;
}

} // namespace mimirmind::cli
