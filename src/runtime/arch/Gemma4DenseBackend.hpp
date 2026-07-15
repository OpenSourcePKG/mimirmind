// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "runtime/arch/GemmaBaseBackend.hpp"

namespace mimirmind::runtime::arch {

/**
 * Gemma 4 dense variant — targets the classic 4B / 12B / E4B family.
 * Uses a single GELU-SwiGLU FFN post-normed via `post_ffw_norm` — no
 * `_1` / `_2` split norms, no router, no expert bank. Everything else
 * (attention block, layer output scale, RoPE, SWA-vs-full split, Q/K/V
 * norms) is inherited from `GemmaBaseBackend` unchanged.
 *
 * Tensor requires beyond the shared attention set:
 *   - ffn_norm, ffn_gate, ffn_up, ffn_down
 *   - post_ffw_norm
 *   - layer_output_scale
 */
class Gemma4DenseBackend final : public GemmaBaseBackend {
public:
    Gemma4DenseBackend(const model::LlmConfig&        config,
                       const core::gguf::WeightsMap&       weights,
                       const model::FusedQkvWeights*  fusedQkv,
                       compute::l0::GpuOps&               ops,
                       compute::l0::GpuMatmul&            gmm,
                       runtime::OpProfiler&           opProfiler);

    void runBlock(std::size_t   blockIdx,
                  float*        x,
                  std::size_t   T,
                  KvCache&      cache,
                  BlockBuffers& s,
                  bool          traceBlock0) override;
};

} // namespace mimirmind::runtime::arch