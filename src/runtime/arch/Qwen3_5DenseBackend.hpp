// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "runtime/arch/Qwen3_5Backend.hpp"

namespace mimirmind::runtime::arch {

/**
 * Qwen3.5 dense (`qwen3_5` text tower, no experts) — the dense-MLP FFN variant
 * of the shared Qwen3_5Backend architecture (GatedDeltaNet linear attention +
 * periodic full attention). First consumer: Qwen3.8-27B (dense hybrid; every
 * layer runs a plain SwiGLU MLP over `feedForwardLength` instead of routed
 * experts). Selected by the arch factory when `config.expertCount == 0`.
 *
 * Single-session generate (runBlock) only for now; the serving-class batched
 * decode / MTP / DFlash-verify paths live on the Qwen3_5MoeBackend sibling and
 * are not yet wired for the dense variant.
 */
class Qwen3_5DenseBackend final : public Qwen3_5Backend {
public:
    Qwen3_5DenseBackend(const model::LlmConfig&       config,
                        const core::gguf::WeightsMap& weights,
                        const model::FusedQkvWeights* fusedQkv,
                        compute::ComputeOps&          ops,
                        compute::ComputeMatmul&       gmm,
                        runtime::OpProfiler&          opProfiler)
        : Qwen3_5Backend(config, weights, fusedQkv, ops, gmm, opProfiler) {}

protected:
    /// FFN seam: plain SwiGLU over the full feedForwardLength (no router /
    /// experts / shared-expert). Writes the result into s.moeAccumBuf.
    void runFfn(std::size_t   blockIdx,
                const float*  ffnInput,
                std::size_t   T,
                BlockBuffers& s) override;
};

} // namespace mimirmind::runtime::arch
