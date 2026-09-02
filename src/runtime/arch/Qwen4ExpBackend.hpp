// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "runtime/arch/Qwen3_5MoeBackend.hpp"

namespace mimirmind::runtime::arch {

/**
 * Qwen4-Exp (`qwen4_exp`) — Qwen3.8-Flash-Next (blazux/qwen3.8-Flash-DGX).
 *
 * Architecturally the SAME hybrid backbone as Qwen3_5MoeBackend — GatedDeltaNet
 * linear attention + periodic full attention (interval 4), routed MoE (512
 * experts, top-10) + gated shared expert, and an MTP head — so it derives from
 * Qwen3_5MoeBackend and reuses the whole routed-expert + serving + MTP path.
 *
 * Over that backbone the model adds two components (roadmap 5.27):
 *   - Hyper-Connections: learned multi-stream residual mixing (per-layer
 *     attn/mlp hyper_connection + a top-level mixer), replacing the plain
 *     residual add. Wired in I-3.
 *   - PLE per-layer n-gram embeddings: a ~48 GiB FP8 sparse lookup (~16 rows /
 *     token) fed into designated layers, served off-VRAM via host mmap. I-4.
 *
 * I-1 (this commit) is the SEAM only: the ctor forwards to Qwen3_5MoeBackend
 * and nothing else is overridden, so a qwen4_exp checkpoint loads + runs the
 * MoE backbone unchanged (Hyper-Connections + PLE not yet active). This lets
 * the factory select the arch and the config/loader path light up before the
 * two new components land.
 */
class Qwen4ExpBackend : public Qwen3_5MoeBackend {
public:
    Qwen4ExpBackend(const model::LlmConfig&       config,
                    const core::gguf::WeightsMap& weights,
                    const model::FusedQkvWeights* fusedQkv,
                    compute::ComputeOps&          ops,
                    compute::ComputeMatmul&       gmm,
                    runtime::OpProfiler&          opProfiler,
                    bool                          moeGroupEnabled     = true,
                    bool                          moeFusedDownEnabled = false);
};

} // namespace mimirmind::runtime::arch
