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
                       compute::ComputeOps&               ops,
                       compute::ComputeMatmul&            gmm,
                       runtime::OpProfiler&           opProfiler);

    void runBlock(std::size_t   blockIdx,
                  float*        x,
                  std::size_t   T,
                  KvCache&      cache,
                  BlockBuffers& s,
                  bool          traceBlock0) override;

    /// M9.1 — synchronized batched decode of `nSeq` lock-step sequences
    /// (T=1 each) over per-sequence caches. The dense FFN gate/up/down run
    /// as matmuls at M=nSeq, so each dense weight read amortizes across the
    /// whole batch — the amortization the sparse-MoE path cannot get
    /// (different tokens route to different experts). F32 KV only.
    void runBlockBatched(std::size_t                blockIdx,
                         float*                     x,
                         std::size_t                nSeq,
                         std::span<KvCache* const>  caches,
                         BlockBuffers&              s,
                         bool                       diag) override;

    [[nodiscard]] bool supportsBatchedDecode() const noexcept override {
        return true;
    }

private:
    /// Dense SwiGLU/GELU FFN tail at `T` rows: fused attn-residual +
    /// ffn_norm → gate/up → gelu·mul → down → post_ffw_norm → residual →
    /// layer_output_scale. Shared by runBlock (T=1 / prefill) and
    /// runBlockBatched (T=nSeq). All ops are M=T so the matmuls amortize.
    void runFfnDenseSection(std::size_t   blockIdx,
                            float*        x,
                            std::size_t   T,
                            BlockBuffers& s,
                            bool          diag);
};

} // namespace mimirmind::runtime::arch