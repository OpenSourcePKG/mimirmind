// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "hugin/train/DenseBlock.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace mimirmind::hugin::train {

/**
 * Whole dense decoder (llama / qwen2) for adapter training (roadmap 7.4.3):
 * token embedding -> DenseBlocks -> final RMSNorm -> lm_head, all frozen; only
 * the blocks' LoRA adapters learn.
 *
 * lossAndGrad() runs the forward once keeping only each block's input
 * (checkpointing), computes the lm_head and cross-entropy ONLY on rows with a
 * target (prompt rows are ignored in SFT, so their logits are never built),
 * and back-propagates down to the lowest block that carries an adapter
 * (nothing below it can receive a gradient). Gradients accumulate in the
 * adapters until the caller steps the optimizer and zeroes them.
 */
class DenseModel {
public:
    struct Head {
        std::size_t        vocab{0};
        std::vector<float> embed;      ///< [vocab, dModel]
        std::vector<float> finalNorm;  ///< [dModel]
        std::vector<float> lmHead;     ///< [vocab, dModel]; empty = tied to embed
        float              eps{1e-5F};
    };

    DenseModel(std::vector<DenseBlock> blocks, Head head);

    [[nodiscard]] std::size_t       blockCount() const noexcept { return _blocks.size(); }
    [[nodiscard]] DenseBlock&       block(std::size_t i) { return _blocks.at(i); }
    [[nodiscard]] const DenseBlock& block(std::size_t i) const { return _blocks.at(i); }
    [[nodiscard]] std::size_t       dModel() const noexcept { return _d; }
    [[nodiscard]] std::size_t       vocab() const noexcept { return _head.vocab; }

    /// Logits [rows.size(), vocab] for the given positions of `ids` (all rows
    /// when `rows` is empty).
    [[nodiscard]] std::vector<float> logits(std::span<const std::int32_t> ids,
                                            std::span<const std::size_t> rows = {}) const;

    /// Mean CE over rows whose target is >= 0 (targets[t] = next token for
    /// position t); LoRA gradients accumulated. Returns 0 when nothing counts.
    double lossAndGrad(std::span<const std::int32_t> ids, std::span<const std::int32_t> targets);

    void zeroGrad();

private:
    void embed(std::span<const std::int32_t> ids, float* x) const;
    [[nodiscard]] const std::vector<float>& lmHead() const {
        return _head.lmHead.empty() ? _head.embed : _head.lmHead;
    }

    std::vector<DenseBlock> _blocks;
    Head                    _head;
    std::size_t             _d{0};
};

} // namespace mimirmind::hugin::train
