// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "runtime/arch/ArchBackend.hpp"

namespace mimirmind::compute {
class ComputeMatmul;
class ComputeOps;
} // namespace mimirmind::compute

namespace mimirmind::runtime {
class OpProfiler;
} // namespace mimirmind::runtime

namespace mimirmind::runtime::arch {

/**
 * Kolibri-1 (Aleph-Alpha `kolibri1`, `Kolibri1ForCausalLM`) — a PURE-ATTENTION
 * MoE: 50 layers of GQA (48 Q / 4 KV heads, head_dim 128) in a 4:1
 * sliding(513):full pattern, per-head QK-RMSNorm, a 384-expert / top-6 SIGMOID
 * router with a per-layer `moe.router.expert_bias` and one always-on shared
 * expert. FP8 128×128 block weights (attn + experts), bf16 everywhere else
 * (embeds / lm_head / norms / router / expert_bias), lm_head computed in f32.
 *
 * I-1 (roadmap 5.34.2) is a SEAM ONLY: the config parses into LlmConfig and the
 * arch factory selects this backend, but the forward is not yet implemented.
 * The reuse/diff map (from the I-0 recon) is:
 *   - attention: reuse Gemma-4 sliding-window attention (SWA 513) + Qwen3
 *     per-head QK-RMSNorm; full-attention layers are NoPE (RoPE only where
 *     slidingWindowPattern[l] == true) — the one real attention diff (I-4,
 *     roadmap 5.34.5);
 *   - MoE: reuse grouped-MoE + shared-expert + FP8 decode-GEMV; the router diff
 *     is a SIGMOID gate (not softmax), top-6 NOT renormalised, with expert_bias
 *     added to the logits for selection, and an always-on shared expert (I-5,
 *     roadmap 5.34.6);
 *   - 4 norms per layer (input / post_attention / post_attn / post_ffn) vs
 *     qwen3's 2 — the sandwich/peri-LN layout to wire in the forward;
 *   - FP8 128×128 block loader (compressed-tensors `weight_scale_inv`) — the
 *     weight-materialise gap (I-3, roadmap 5.34.4);
 *   - tokenizer: GPT-2/Qwen byte-level BPE, reuses `Tokenizer::loadFromHfJson`
 *     (I-2, roadmap 5.34.3).
 *
 * Deriving directly from ArchBackend (not Qwen3_5Backend) keeps the seam small
 * and free of the GatedDeltaNet machinery Kolibri does not use; the shared
 * attention / grouped-MoE helpers are pulled in when the forward lands.
 */
class Kolibri1Backend : public ArchBackend {
public:
    Kolibri1Backend(const model::LlmConfig&       config,
                    const core::gguf::WeightsMap& weights,
                    const model::FusedQkvWeights* fusedQkv,
                    compute::ComputeOps&          ops,
                    compute::ComputeMatmul&       gmm,
                    runtime::OpProfiler&          opProfiler,
                    bool                          moeGroupEnabled     = true,
                    bool                          moeFusedDownEnabled = false);

    void runBlock(std::size_t   blockIdx,
                  float*        x,
                  std::size_t   T,
                  KvCache&      cache,
                  BlockBuffers& buffers,
                  bool          traceBlock0) override;

    [[nodiscard]] std::vector<std::size_t> kvDimPerLayer() const override;

    [[nodiscard]] std::pair<std::size_t, std::size_t>
        maxQKVDims() const override;

    [[nodiscard]] const char* name() const noexcept override {
        return "kolibri1";
    }

protected:
    [[nodiscard]] ArchCapabilities computeCapabilities() const override;

private:
    // _config is read now (kvDimPerLayer / maxQKVDims); the rest are held for
    // the forward that lands in I-3+ (FP8 materialise / attention / MoE) and are
    // [[maybe_unused]] until then so the I-1 seam compiles warning-clean.
    const model::LlmConfig&                      _config;
    [[maybe_unused]] const core::gguf::WeightsMap& _weights;
    [[maybe_unused]] const model::FusedQkvWeights* _fusedQkv;
    [[maybe_unused]] compute::ComputeOps&          _ops;
    [[maybe_unused]] compute::ComputeMatmul&       _gmm;
    [[maybe_unused]] runtime::OpProfiler&          _opProfiler;
    [[maybe_unused]] bool                          _moeGroupEnabled;
    [[maybe_unused]] bool                          _moeFusedDownEnabled;
};

} // namespace mimirmind::runtime::arch
