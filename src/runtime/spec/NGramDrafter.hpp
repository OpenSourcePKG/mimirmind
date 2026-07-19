// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "runtime/spec/Drafter.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace mimirmind::runtime {

/**
 * Prompt-Lookup Decoding (PLD) drafter — proposes draft tokens by
 * searching the committed context for a match of its own tail.
 *
 * Algorithm per round:
 *   1. Take the last `k` tokens of `context` as needle, iterating
 *      `k = maxK, maxK-1, ..., minK`.
 *   2. Scan `context[0 .. needleStart-1]` for a match; prefer the
 *      most-recent match (backward scan) to bias towards local
 *      repetition structure that RAG-style prompts exhibit.
 *   3. On the first useful match at any `k`, return up to N tokens
 *      starting at `match+k`, truncated so no proposed token overlaps
 *      the needle itself.
 *   4. If no useful match at any `k`, return an empty batch —
 *      `SpeculativeDecoder` degrades to a verify-only round.
 *
 * Zero cost per round: no model forward, no KV, no GPU dispatch. The
 * scan is at most O(len * (maxK - minK + 1) * k) with tiny constants —
 * negligible next to a target forward.
 */
class NGramDrafter final : public Drafter {
public:
    struct Config {
        /// Upper bound on the needle length. Larger = more specific
        /// matches = fewer hits but higher accept-rate on hits.
        std::size_t maxK{3};
        /// Lower bound. Below this the match is too generic to be
        /// useful. Must be >= 1.
        std::size_t minK{2};
    };

    explicit NGramDrafter(Config cfg);

    NGramDrafter(const NGramDrafter&)            = delete;
    NGramDrafter& operator=(const NGramDrafter&) = delete;
    NGramDrafter(NGramDrafter&&)                 = delete;
    NGramDrafter& operator=(NGramDrafter&&)      = delete;

    compute::SpeculativeBatch propose(
        std::span<const std::int32_t>  context,
        std::size_t                    N,
        const compute::SamplingParams& sampling) override;

    [[nodiscard]] std::string_view kind() const noexcept override { return "ngram"; }

    [[nodiscard]] const Config& config() const noexcept { return _cfg; }

    /// Pure-function core exposed for unit tests. Returns the follow-up
    /// tokens the drafter would propose (or empty on no match). The
    /// wrapping `SpeculativeBatch` in `propose()` fills `probs` with
    /// 1.0F per token and leaves `hitStop=false` — PLD has no notion
    /// of an EOS short-circuit; the target's verify pass catches a
    /// stop id like any other token.
    [[nodiscard]] static std::vector<std::int32_t> lookup(
        std::span<const std::int32_t> context,
        std::size_t                   N,
        std::size_t                   maxK,
        std::size_t                   minK);

private:
    Config _cfg;
};

} // namespace mimirmind::runtime