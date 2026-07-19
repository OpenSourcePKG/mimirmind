// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "compute/Sampling.hpp"
#include "compute/SpeculativeSampler.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace mimirmind::runtime {

/**
 * Source of speculative draft tokens for `SpeculativeDecoder`.
 *
 * A Drafter observes the target's committed context and proposes the
 * next up-to-N tokens per verify round. Implementations range from
 * full-model draft engines (`ModelDrafter`) to zero-cost prompt-lookup
 * (`NGramDrafter`, aka PLD).
 *
 * The decoder passes the full committed context on every round so
 * implementations can either use the tail as a warm-cache lookup key
 * (model draft) or scan the whole thing for a prefix match (n-gram).
 *
 * Returning fewer than N tokens is legal — the verify batch shrinks
 * accordingly. Returning an empty batch is legal — the decoder still
 * runs a verify-only round that produces a single recovery token, so
 * degrading to plain decode is graceful.
 *
 * Not thread-safe.
 */
class Drafter {
public:
    virtual ~Drafter() = default;

    /// Propose up to `N` draft tokens continuing from `context`.
    /// `context` contains every token the target has already committed
    /// (prompt + generated so far). `sampling` is the caller's sampler
    /// setting — model drafters forward it to their own sampler; token-
    /// free drafters ignore it.
    virtual compute::SpeculativeBatch propose(
        std::span<const std::int32_t>  context,
        std::size_t                    N,
        const compute::SamplingParams& sampling) = 0;

    /// Called once by `SpeculativeDecoder` before the first speculation
    /// round so a model-backed drafter can prefill its own KV to match
    /// the target's post-prompt state. Zero-cost drafters ignore it.
    virtual void warmPrefix(std::span<const std::int32_t> /*promptPlusFirstToken*/) {}

    /// Human-readable label for logs / status ("model", "ngram", ...).
    [[nodiscard]] virtual std::string_view kind() const noexcept = 0;
};

} // namespace mimirmind::runtime