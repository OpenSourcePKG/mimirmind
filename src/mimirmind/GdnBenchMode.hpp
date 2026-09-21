// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

namespace mimirmind::core::config { struct Config; }
namespace mimirmind::cli { struct CliArgs; }

namespace mimirmind::cli {

/**
 * `mimirmind gdnbench` — disposable single-kernel GatedDeltaNet chunk-forward
 * (gdn.k2) microbenchmark. Boots ONLY a CUDA compute context + GpuOps (no
 * model weights, no serving), allocates synthetic q/k/v/gCum/beta/a0/state at
 * the prod qwen3.6 GDN shape (H=32 value heads, S=128 state dim, C=64 chunk),
 * and times `deltanetChunkForwardBatchedAsync` across a token sweep. The TC
 * path (deltanet_chunk_forward_batched_tc) engages with MIMIRMIND_GDN_CHUNK_TC=1.
 * Reports ms + achieved bf16-TC TFLOP/s so the objective question — is gdn.k2
 * load/latency-bound (cp.async/CuTe helps) or compute-bound (it does not) —
 * is answered before any restructure. Not a production path; safe standalone.
 */
[[nodiscard]] int runGdnBench(const CliArgs& args,
                              const ::mimirmind::core::config::Config& cfg);

} // namespace mimirmind::cli
