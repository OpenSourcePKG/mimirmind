#pragma once

#include "runtime/arch/ArchBackend.hpp"

#include <cstddef>
#include <utility>
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
 * Architectural quirks (vs Qwen2):
 *   - Per-layer head_dim and kv_heads — SWA layers use head_dim_swa /
 *     kv_heads_swa, full-attention layers use head_dim_full / kv_heads_full.
 *     For 26B-A4B that's (256 × 8) vs (512 × 2).
 *   - Q-K-norm per head; V is bare-RMS-normalised (no learned weight).
 *   - Alternative attention: full-attention layers may omit attn_v.weight
 *     entirely. When that happens V = raw K projection (pre-K-norm,
 *     pre-RoPE). This is the "use_alternative_attention" flag in the
 *     upstream HF/llama.cpp Gemma 4 reference code.
 *   - f_attention_scale = 1.0 (we pre-multiply Q by sqrt(head_dim) so
 *     the internal 1/sqrt(headDim) in multiHeadAttention cancels).
 *   - Per-layer RoPE base (SWA vs full) and `freq_factors` (proportional
 *     RoPE) only on full layers.
 *   - Hybrid FFN: dense path A (GELU SwiGLU) + MoE path B (128 experts,
 *     top-8 with renormalised weights + per-expert ffn_down_exps.scale).
 *   - Multi-norm choreography: attn_norm, attn_post_norm, ffn_norm,
 *     post_ffw_norm_1, pre_ffw_norm_2, post_ffw_norm_2, post_ffw_norm.
 *   - layer_output_scale F32 scalar per block (final per-layer multiply).
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

    [[nodiscard]] std::vector<std::size_t>
        kvDimPerLayer() const override;
    [[nodiscard]] std::pair<std::size_t, std::size_t>
        maxQKVDims() const override;

private:
    /// Per-layer config snapshot, resolved once at construction.
    struct LayerInfo {
        bool        isSwa;            // SWA vs full-attention layer
        bool        altAttention;     // V derived from raw K (no attn_v.weight)
        std::size_t headDim;          // head_dim for this layer
        std::size_t nHeads;           // Q head count (always config.headCount)
        std::size_t nKvHeads;         // KV head count for this layer
        std::size_t qDim;             // nHeads * headDim
        std::size_t kvDim;            // nKvHeads * headDim
        float       ropeBase;         // SWA vs full rope base
    };

    /// Inspect WeightsMap + config to fill per-layer info.
    void buildLayerInfos();

    const model::LlmConfig&   _config;
    const model::WeightsMap&  _weights;
    compute::GpuOps&          _ops;
    compute::GpuMatmul&       _gmm;

    std::vector<LayerInfo>    _layers;

    /// USM pointer to the global `rope_freqs.weight` (F32 [head_dim/2])
    /// used as `freq_factors` for full-attention layers only.
    const float*              _ropeFreqsForFullAttn{nullptr};
};

} // namespace mimirmind::runtime::arch