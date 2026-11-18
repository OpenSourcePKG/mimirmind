// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace mimirmind::hugin {

/**
 * How well a direction separates two classes (roadmap 7.12 guardrails).
 * Scores are projections / cosines, larger = more "positive".
 */
struct SeparationReport {
    double      auc{0.5};               ///< P(score_pos > score_neg), ties count 1/2
    double      accuracy{0.0};          ///< at the given threshold
    double      negativeFireRate{0.0};  ///< negatives above the threshold
    std::size_t positives{0};
    std::size_t negatives{0};
    /// Prompts behind the scores when one prompt yields several rows
    /// (0 = one score per prompt, i.e. positives / negatives).
    std::size_t positivePrompts{0};
    std::size_t negativePrompts{0};
};

/// Distribution of a separation metric under shuffled class labels.
struct NullStats {
    double      mean{0.5};
    double      p95{0.5};
    double      max{0.5};
    std::size_t permutations{0};
};

/**
 * Host-side separation statistics for steering / condition directions.
 * Pure functions, no engine dependency, unit-testable without a model.
 */
class SeparationMath {
public:
    /// Mann-Whitney AUC; 0.5 when a class is empty.
    [[nodiscard]] static double auc(std::span<const double> pos, std::span<const double> neg);

    /// AUC, accuracy, negative fire rate at `threshold` (score > threshold = positive).
    [[nodiscard]] static SeparationReport report(std::span<const double> pos,
                                                 std::span<const double> neg, double threshold);

    /// In-sample AUC of the mean-difference direction (rows [n, d] row-major,
    /// first `positives` rows are the positive class). Projections onto the
    /// unit direction mean(pos) - mean(neg).
    [[nodiscard]] static double meanDifferenceAuc(std::span<const float> rows, std::size_t d,
                                                  std::span<const std::uint8_t> isPositive);

    /// meanDifferenceAuc under `permutations` random relabellings with the same
    /// class sizes. With d >> n even random labels separate in-sample, so the
    /// real AUC only means something above this null.
    [[nodiscard]] static NullStats meanDifferenceNull(std::span<const float> rows, std::size_t d,
                                                      std::size_t positives,
                                                      std::size_t permutations,
                                                      std::uint64_t seed);

    /// Deterministic shuffle of [0, n) (splitmix64 Fisher-Yates).
    [[nodiscard]] static std::vector<std::size_t> permutation(std::size_t n, std::uint64_t seed);

    /// Mean / 95th percentile / max of `values` (empty -> zeros, permutations 0).
    [[nodiscard]] static NullStats summarize(std::vector<double> values);
};

} // namespace mimirmind::hugin
