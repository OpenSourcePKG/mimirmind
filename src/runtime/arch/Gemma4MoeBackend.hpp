#pragma once

#include "runtime/arch/GemmaBaseBackend.hpp"

namespace mimirmind::runtime::arch {

/**
 * Gemma 4 MoE variant — the 26B-A4B-it target (30 blocks, 128 experts,
 * top-8). Adds the hybrid FFN choreography on top of `GemmaBaseBackend`'s
 * shared attention pipeline: dense Path A + MoE Path B + combine +
 * `post_ffw_norm`, then residual + `layer_output_scale`.
 *
 * Tensor requires beyond the shared attention set:
 *   - post_ffw_norm_1, pre_ffw_norm_2, post_ffw_norm_2, post_ffw_norm
 *   - ffn_gate, ffn_up, ffn_down, ffn_norm (Path A dense weights)
 *   - ffn_gate_inp.scale, ffn_gate_inp.weight (router)
 *   - ffn_gate_up_exps.weight, ffn_down_exps.weight, ffn_down_exps.scale
 *   - layer_output_scale
 */
class Gemma4MoeBackend final : public GemmaBaseBackend {
public:
    Gemma4MoeBackend(const model::LlmConfig&        config,
                     const model::WeightsMap&       weights,
                     const model::FusedQkvWeights*  fusedQkv,
                     compute::GpuOps&               ops,
                     compute::GpuMatmul&            gmm,
                     runtime::OpProfiler&           opProfiler);

    void runBlock(std::size_t   blockIdx,
                  float*        x,
                  std::size_t   T,
                  KvCache&      cache,
                  BlockBuffers& s,
                  bool          traceBlock0) override;
};

} // namespace mimirmind::runtime::arch