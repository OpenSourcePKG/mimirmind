// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "compute/ComputeBuffer.hpp"
#include "core/gguf/GgufTypes.hpp"

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace mimirmind::compute {
class ComputeOps;
} // namespace mimirmind::compute

namespace mimirmind::runtime::encoder::laya {

/**
 * Weights of the Laya typed-decision head that sits on the ModernBERT backbone
 * (LayaEncoderModel/Runner). Loaded from the same model.safetensors (the
 * non-`encoder.` tensors). See the verified spec in the Synaipse note
 * "Laya decision-head mechanism (reverse-eng for 8.27.3)".
 *
 * Pipeline the head implements (forward lives in LayaDecisionRunner, next step):
 *   h = backbone(...)                       [T, d]
 *   h += type_emb[qtype]                     (broadcast over tokens)
 *   h = head.layers[0..nHeadLayers-1](h)     std. pre-norm TransformerEncoderLayer
 *   logits = scorer(gather(h, markers))      per option
 *   act = softmax(act_head([h[0] ⊕ feats]))  confidence
 *
 * All F16 in the checkpoint; this loader expands to F32 (linear weights as
 * `matmulType`, norms/biases/embeddings F32).
 */
struct LayaHeadLayerWeights {
    const float* inProjW{};   // self_attn.in_proj_weight  [3*d, d] (fused Q|K|V)
    const float* inProjB{};   // self_attn.in_proj_bias    [3*d]
    const float* outProjW{};  // self_attn.out_proj.weight [d, d]
    const float* outProjB{};  // self_attn.out_proj.bias   [d]
    const float* norm1W{};    const float* norm1B{};   // pre-attention LayerNorm
    const float* norm2W{};    const float* norm2B{};   // pre-MLP LayerNorm
    const float* lin1W{};     const float* lin1B{};    // [4*d, d], [4*d]  (relu FFN up)
    const float* lin2W{};     const float* lin2B{};    // [d, 4*d], [d]    (FFN down)
};

struct LayaDecisionHeadConfig {
    std::size_t hidden{0};
    std::size_t heads{0};      // d // 64
    std::size_t headDim{64};
    std::size_t ffn{0};        // 4 * d
    std::size_t nHeadLayers{0};
    std::size_t nAct{2};
};

class LayaDecisionHead {
public:
    LayaDecisionHead() = default;

    LayaDecisionHead(const LayaDecisionHead&)            = delete;
    LayaDecisionHead& operator=(const LayaDecisionHead&) = delete;
    LayaDecisionHead(LayaDecisionHead&&) noexcept            = default;
    LayaDecisionHead& operator=(LayaDecisionHead&&) noexcept = default;

    /// Load the head tensors from `dir`'s model.safetensors. `hidden` is the
    /// encoder hidden size (the head has no config.json of its own); heads =
    /// hidden/64, ffn = 4*hidden. `nHeadLayers` = rl_agent_config head_layers
    /// (2 for the shipped checkpoints). Throws on a missing tensor.
    void load(std::string_view dir, compute::ComputeOps& ops, std::size_t hidden,
              std::size_t nHeadLayers = 2,
              core::gguf::GgmlType matmulType = core::gguf::GgmlType::F32);

    [[nodiscard]] const LayaDecisionHeadConfig& config() const noexcept { return _config; }
    [[nodiscard]] core::gguf::GgmlType matmulType() const noexcept { return _matmulType; }
    [[nodiscard]] std::size_t weightBytes() const noexcept { return _weightBytes; }

    [[nodiscard]] const float* typeEmb() const noexcept { return _typeEmb; }          // [3, d]
    [[nodiscard]] const float* temperature() const noexcept { return _temperature; }  // [3]
    [[nodiscard]] const LayaHeadLayerWeights& layer(std::size_t i) const { return _layers.at(i); }

    // scorer = LayerNorm(d) -> Linear(d,d) -> GELU(erf) -> Linear(d,1)
    [[nodiscard]] const float* scorerLnW() const noexcept { return _scorerLnW; }
    [[nodiscard]] const float* scorerLnB() const noexcept { return _scorerLnB; }
    [[nodiscard]] const float* scorerW1() const noexcept { return _scorerW1; }   // [d, d]
    [[nodiscard]] const float* scorerB1() const noexcept { return _scorerB1; }   // [d]
    [[nodiscard]] const float* scorerW3() const noexcept { return _scorerW3; }   // [1, d]
    [[nodiscard]] const float* scorerB3() const noexcept { return _scorerB3; }   // [1]

    // act_head = Linear(d+4, 256) -> GELU(erf) -> Linear(256, nAct)
    [[nodiscard]] const float* actW0() const noexcept { return _actW0; }   // [256, d+4]
    [[nodiscard]] const float* actB0() const noexcept { return _actB0; }   // [256]
    [[nodiscard]] const float* actW2() const noexcept { return _actW2; }   // [nAct, 256]
    [[nodiscard]] const float* actB2() const noexcept { return _actB2; }   // [nAct]

private:
    LayaDecisionHeadConfig _config{};
    core::gguf::GgmlType   _matmulType{core::gguf::GgmlType::F32};
    std::vector<compute::ComputeBuffer> _owned;
    std::size_t _weightBytes{0};
    std::vector<LayaHeadLayerWeights> _layers;

    const float* _typeEmb{};
    const float* _temperature{};
    const float* _scorerLnW{}; const float* _scorerLnB{};
    const float* _scorerW1{};  const float* _scorerB1{};
    const float* _scorerW3{};  const float* _scorerB3{};
    const float* _actW0{};     const float* _actB0{};
    const float* _actW2{};     const float* _actB2{};
};

} // namespace mimirmind::runtime::encoder::laya
