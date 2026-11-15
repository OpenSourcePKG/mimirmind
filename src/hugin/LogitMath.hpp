// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace mimirmind::hugin {

/**
 * Host-side math over one logits row, shared by Hugin's forward tools
 * (roadmap 7.2 Inc-2). Pure functions; no engine dependency so they are
 * unit-testable without a model.
 */
class LogitMath {
public:
    /// Gemma-style final soft-cap (cap <= 0 = identity), applied in place.
    static void softcap(std::vector<float>& logits, float cap) noexcept;

    /// Numerically stable log-softmax (double precision).
    [[nodiscard]] static std::vector<double> logSoftmax(std::span<const float> logits);

    /// Indices of the k largest values, descending.
    [[nodiscard]] static std::vector<std::int32_t> topK(std::span<const double> v, std::size_t k);

    /// Rank (0 = argmax) of `index` in `v`.
    [[nodiscard]] static std::size_t rankOf(std::span<const double> v, std::size_t index);

    /// Shannon entropy (nats) of the distribution given by log-probs.
    [[nodiscard]] static double entropy(std::span<const double> logp) noexcept;

    /// KL(p || q) in nats, both given as log-probs of equal size.
    [[nodiscard]] static double kl(std::span<const double> logp, std::span<const double> logq) noexcept;
};

} // namespace mimirmind::hugin
