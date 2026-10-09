// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace mimirmind::compute {
class ComputeOps;
class ComputeMatmul;
} // namespace mimirmind::compute

namespace mimirmind::runtime::encoder::laya {

class LayaEncoderModel;
class LayaEncoderRunner;
class LayaDecisionHead;

/**
 * The Laya typed-decision forward: runs the ModernBERT backbone, then the head
 * (type_emb add + 2 standard pre-norm TransformerEncoderLayers + marker-gather
 * scorer + act_head) to produce per-option logits and the act/confidence head.
 * Sequential option layout (parallel_options=false), B=1. See the Synaipse spec
 * note "Laya decision-head mechanism (reverse-eng for 8.27.3)".
 */
class LayaDecisionRunner {
public:
    LayaDecisionRunner(const LayaEncoderModel& model, LayaEncoderRunner& encoder,
                       const LayaDecisionHead& head, compute::ComputeOps& ops,
                       compute::ComputeMatmul& matmul);

    struct Result {
        std::vector<float> logits;   // per-option raw scorer logits [k]
        std::vector<float> actLogits;// act head [2]
        float actProbability{0.0F};  // softmax(actLogits)[0]
    };

    /// Run one (tokenized state + option markers) through backbone + head.
    /// `inputIds` = the full rendered sequence; `markerPos` = absolute token
    /// index of each option's marker; `qtype` = 0 choice / 1 score / 2 noul.
    [[nodiscard]] Result decide(std::span<const std::int32_t> inputIds,
                                std::span<const std::int32_t> markerPos,
                                std::int32_t qtype);

private:
    const LayaEncoderModel&  _m;
    LayaEncoderRunner&       _enc;
    const LayaDecisionHead&  _head;
    compute::ComputeOps&     _ops;
    compute::ComputeMatmul&  _mm;
};

} // namespace mimirmind::runtime::encoder::laya
