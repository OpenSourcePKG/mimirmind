// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "compute/Sampling.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace mimirmind::runtime {
class InferenceEngine;
}

namespace mimirmind::compute {

/**
 * One speculation round's output: N token ids that the draft model
 * committed to, plus a per-position draft-probability slot. `probs[i]`
 * is `p_draft(tokens[i] | prefix + tokens[0..i-1])` when the accept
 * logic in M9.11.4 populates it; M9.11.2 leaves them at 1.0 as a
 * placeholder for greedy draft sampling.
 *
 * `hitStop` is true when the draft hit its EOS / stop-id before
 * generating N tokens. The accept-loop still runs over `tokens.size()`
 * — the target verify batch has to see the same short sequence.
 */
struct SpeculativeBatch {
    std::vector<std::int32_t> tokens;
    std::vector<float>        probs;
    bool                      hitStop{false};
};

/**
 * Runs the draft model N times to produce a batch of speculative tokens
 * that a target-model verify pass (M9.11.3) will later accept or reject
 * via modified rejection sampling (M9.11.4).
 *
 * M9.11.2 scope: token-only. The class is stateless in this iteration;
 * the RNG that M9.11.4 needs for real rejection sampling will move
 * here when that step lands so callers keep the same object across
 * both.
 */
class SpeculativeSampler {
public:
    SpeculativeSampler() = default;

    /**
     * Advance `draft` by up to `N` tokens starting from `promptSoFar`.
     * The draft engine's KV cache is left at position
     * `promptSoFar.size() + returned.tokens.size()`; caller owns any
     * rollback after accept/reject (M9.11.5).
     *
     * `promptSoFar` includes every token the target has already
     * committed — the draft engine's own prefix cache handles the
     * longest-common-prefix skip internally.
     *
     * Returns fewer than N tokens when the draft's decode loop stopped
     * on EOS / a chat-template stop id — `hitStop` is set in that case.
     */
    SpeculativeBatch speculate(runtime::InferenceEngine&      draft,
                               std::span<const std::int32_t>  promptSoFar,
                               std::size_t                    N,
                               const SamplingParams&          sampling = {});
};

} // namespace mimirmind::compute