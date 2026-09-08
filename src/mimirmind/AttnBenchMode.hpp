// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

namespace mimirmind::core::config { struct Config; }
namespace mimirmind::cli { struct CliArgs; }

namespace mimirmind::cli {

/**
 * `mimirmind attnbench` — disposable single-kernel prefill-attention
 * microbenchmark. Boots ONLY a CUDA compute context + GpuOps (no model
 * weights, no serving), allocates synthetic contiguous Q/K/V at the
 * qwen3-coder-next full-attention shape (nHeads=16, nKvHeads=2,
 * headDim=256), and times `attentionPrefillFlashAsync` across a sweep of
 * prompt lengths T. The concrete kernel it dispatches to (cuDNN SDPA vs
 * multi-warp TF32 tensor-core vs the default hand kernel) is selected by
 * the usual GpuOps env flags (MIMIRMIND_ATTN_CUDNN / MIMIRMIND_ATTN_F32_MWTC),
 * so running the same binary under different env isolates which head_dim=256
 * prefill attention path is actually fast on this GB10 — the question that
 * gates wiring a fast kernel into the serving paged prefill.
 *
 * Not a production path; safe to run alongside nothing else on the GPU.
 */
[[nodiscard]] int runAttnBench(const CliArgs& args,
                               const ::mimirmind::core::config::Config& cfg);

} // namespace mimirmind::cli
