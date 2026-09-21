// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

namespace mimirmind::core::config { struct Config; }
namespace mimirmind::cli { struct CliArgs; }

namespace mimirmind::cli {

/**
 * `mimirmind moebench` — disposable single-kernel decode MoE-GEMM microbench
 * (5.28.2 de-risk gate). Boots ONLY a CUDA compute context + GpuOps (no model
 * weights, no serving), allocates synthetic NVFP4_BLK expert banks + a
 * representative low-M decode routing (M tokens x top-8, ~1 row/expert), and
 * times the prod device-driven deint decode path
 * (`moeGroupedGemmNvfp4DeintAsync`) across M in {1,2,4,8} for gate (N=n_ff) and
 * down (N=d_model). Reports ms + achieved WEIGHT-READ bandwidth vs the 273 GB/s
 * GB10 ceiling — the objective question the swap_ab/masked kernel hinges on:
 * if the decode MoE GEMM already runs at ~bandwidth peak it is bandwidth-bound
 * (swap_ab, a tile-fill/compute fix, cannot help -> NO-GO); if it runs well
 * below peak there is compute/latency headroom (swap_ab/masked worth building
 * -> GO). Not a production path; ncu-safe single-kernel harness.
 */
[[nodiscard]] int runMoeBench(const CliArgs& args,
                              const ::mimirmind::core::config::Config& cfg);

} // namespace mimirmind::cli
