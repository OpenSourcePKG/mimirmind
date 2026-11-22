// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "runtime/InferenceEngine.hpp"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace mimirmind::hugin {

/**
 * The JSON reports of Hugin's inspection tools, computed from what one
 * forward captured (residual rows, stage taps) — no engine, no device. The
 * single-purpose tools (activations, attention_pattern, expert_usage) and the
 * all-in-one `trace` build their answers here, so one prompt gives the same
 * numbers whichever tool asked.
 */
class ForwardReport {
public:
    using Stage = runtime::InferenceEngine::ForwardProbe::Stage;
    /// Token id -> piece (the model's tokenizer).
    using Piece = std::function<std::string(std::int32_t)>;

    /// Captured attention stages of one block.
    struct AttentionStages {
        const Stage* q{nullptr};     // post-RoPE Q   [N x H*hd]
        const Stage* k{nullptr};     // post-RoPE K   [>=N x Hkv*hd]
        const Stage* v{nullptr};     // V             [>=N x Hkv*hd]
        const Stage* ctx{nullptr};   // kernel output [N x H*hd]
        double       scale{1.0};
        std::size_t  window{0};      // sliding window; 0 = full
        std::size_t  headCount{0};
    };

    struct AttentionOptions {
        std::size_t              position{0};
        std::vector<std::size_t> heads;        // empty = all
        std::size_t              topK{5};
        bool                     full{false};  // include each head's weight row
        std::optional<std::pair<std::size_t, std::size_t>> span;   // [start, end)
    };

    /**
     * attention_pattern of one block: per-head softmax(q·k·scale) over the
     * causal (and windowed) keys, entropy, BOS / self weight, top-k, and the
     * consistency of weights·V against the kernel's own output. Throws on
     * shapes that do not fit `ids` / the head count.
     */
    [[nodiscard]] static nlohmann::json attention(const AttentionStages& s,
                                                  std::span<const std::int32_t> ids,
                                                  const Piece& piece, std::uint32_t block,
                                                  const AttentionOptions& o);

    /**
     * activations entry of one block at one position: L2, mean, std, abs-max
     * over finite values, non-finite count, strongest dims; with `prev` also
     * the update L2 and the cosine to the previous block; `values` adds the
     * raw vector.
     */
    [[nodiscard]] static nlohmann::json residual(std::uint32_t block, std::span<const float> x,
                                                 std::optional<std::span<const float>> prev,
                                                 std::size_t topDims, bool values);

    /**
     * expert_usage entry of one block from the router's captured top-k ids
     * (and weights, if captured): distinct experts, load entropy, top experts
     * by count and weight, the picks at `position`. Adds the counts to
     * `totalCount` (size = experts).
     */
    [[nodiscard]] static nlohmann::json expertBlock(std::uint32_t block, const Stage& idx,
                                                    const Stage* weight, std::size_t experts,
                                                    std::size_t topN,
                                                    std::optional<std::size_t> position,
                                                    std::vector<double>& totalCount);

    /// Summary over all blocks' `totalCount`: top experts overall and the never-used count.
    [[nodiscard]] static nlohmann::json expertSummary(const std::vector<double>& totalCount,
                                                      std::size_t topN);
};

} // namespace mimirmind::hugin
