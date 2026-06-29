#pragma once

#include "runtime/arch/ArchBackend.hpp"

#include <cstddef>
#include <vector>

namespace mimirmind::compute {
class GpuMatmul;
class GpuOps;
} // namespace mimirmind::compute

namespace mimirmind::model {
class WeightsMap;
struct LlmConfig;
} // namespace mimirmind::model

namespace mimirmind::runtime::arch {

/**
 * Gemma 4 26B-A4B-it block.
 *
 * Key differences vs Qwen2:
 *   - Q-K-norm per head + bare V-norm before KV cache
 *   - f_attention_scale = 1.0 (Q pre-scaled by sqrt(head_dim) so the
 *     internal 1/sqrt(headDim) in multiHeadAttention cancels)
 *   - hybrid FFN: dense path A (GELU SwiGLU) + MoE path B (128 experts,
 *     top-8 with renormalised weights + per-expert down_exps.scale)
 *   - multi-norm choreography: attn_post_norm, post_ffw_norm_1,
 *     pre_ffw_norm_2, post_ffw_norm_2, post_ffw_norm
 *   - layer_output_scale F32 scalar per block
 *   - KV sharing: some blocks omit attn_k/v.weight and reuse an earlier
 *     block's KV slot (pattern resolved at construction)
 */
class Gemma4Backend final : public ArchBackend {
public:
    Gemma4Backend(const model::LlmConfig&   config,
                  const model::WeightsMap&  weights,
                  compute::GpuOps&          ops,
                  compute::GpuMatmul&       gmm);

    void runBlock(std::size_t   blockIdx,
                  float*        x,
                  std::size_t   T,
                  KvCache&      cache,
                  BlockBuffers& buffers,
                  bool          traceBlock0) override;

    [[nodiscard]] bool        scalesEmbedding() const noexcept override { return true; }
    [[nodiscard]] const char* name()            const noexcept override { return "gemma4"; }

private:
    /// Walks the weights map once at construction and records, per block,
    /// either its own index (if it owns attn_k/v.weight) or the index of
    /// the most recent block that does. Throws if block 0 is shared
    /// (model would be malformed).
    void buildKvSharePattern();

    const model::LlmConfig&   _config;
    const model::WeightsMap&  _weights;
    compute::GpuOps&          _ops;
    compute::GpuMatmul&       _gmm;

    /// _kvSource[b] = b if block b carries its own attn_k/v.weight;
    /// otherwise the most recent earlier block that does.
    std::vector<std::size_t>  _kvSource;
};

} // namespace mimirmind::runtime::arch