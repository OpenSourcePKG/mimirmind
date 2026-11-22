// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace mimirmind::hugin {

/// Per-expert statistics of one MoE block over a calibration set.
struct ExpertStats {
    std::size_t expert{0};
    std::size_t count{0};        ///< tokens that routed to this expert
    double      weightSum{0.0};  ///< sum of the routing weight g_j(x)
    double      normSum{0.0};    ///< sum of ||f_j(x)||_2
    double      reap{0.0};       ///< mean over routed tokens of g_j(x) * ||f_j(x)||_2
};

/**
 * REAP expert saliency (roadmap 7.9; Cerebras REAP, arXiv 2510.13999).
 *
 * For one MoE block, S_j = mean over the tokens x routed to expert j of
 * g_j(x) * ||f_j(x)||_2, where g is the routing weight actually used and f_j
 * the expert's output before weighting. An expert no token routed to scores 0.
 * Pruning removes the lowest S_j; frequency (token count) is kept alongside
 * as the baseline the paper compares against. Shared experts are not routed
 * and never appear here.
 *
 * Pure host math, no engine dependency, unit-testable without a model.
 */
class ExpertSaliency {
public:
    explicit ExpertSaliency(std::size_t experts);

    /// Add one forward's routing of this block: `idx`, `weight` and `norm`
    /// are [tokens, topK] row-major (the moe_topk_idx / moe_topk_w /
    /// moe_expert_norm stages). Out-of-range ids throw.
    void add(std::span<const float> idx, std::span<const float> weight,
             std::span<const float> norm, std::size_t topK);

    [[nodiscard]] std::size_t experts() const noexcept { return _stats.size(); }
    [[nodiscard]] std::size_t tokens() const noexcept { return _tokens; }

    /// Stats with `reap` filled, indexed by expert id.
    [[nodiscard]] std::vector<ExpertStats> stats() const;

    /// The `n` experts with the lowest REAP score (ties: fewer tokens first,
    /// then lower id), ascending by expert id.
    [[nodiscard]] std::vector<std::size_t> pruneByReap(std::size_t n) const;
    /// The `n` experts with the fewest routed tokens (ties: lower routing
    /// weight sum, then lower id), ascending by expert id.
    [[nodiscard]] std::vector<std::size_t> pruneByFrequency(std::size_t n) const;

    /// |a ∩ b| / |a ∪ b| of two id sets (1 when both are empty).
    [[nodiscard]] static double jaccard(std::span<const std::size_t> a, std::span<const std::size_t> b);

private:
    std::vector<ExpertStats> _stats;
    std::size_t              _tokens{0};
};

} // namespace mimirmind::hugin
