// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstddef>
#include <cstdint>

namespace mimirmind::hugin::train {

/**
 * Forward KL(p || q) of a frozen reference distribution p against the model's
 * softmax q = softmax(z), per row (7.4.5: the base-model anchor of LoRA-SFT).
 *
 * p is stored sparsely: its top-K tokens with their probabilities plus one
 * tail bucket holding the remaining mass. The KL is taken over that partition
 * (K tokens + the tail as one event), which is exact for the bucketed
 * distributions and makes the cache K floats per row instead of the vocab:
 *
 *   KL = sum_k p_k log(p_k / q_k) + p_tail log(p_tail / q_tail)
 *   dKL/dz_j = q_j - p~_j,   p~_j = p_j on the top-K, p_tail * q_j / q_tail off it
 *
 * (p~ spreads the tail mass like the model does, so the gradient never pulls
 * single tail tokens towards the reference.)
 */
class KlDivGrad {
public:
    /// Top-K of softmax(logits[V]) into ids[K] / p[K] (descending) and the
    /// tail mass 1 - sum(p).
    static void topK(const float* logits, std::size_t V, std::size_t K, std::int32_t* ids, float* p,
                     float& tail);

    /// KL of one row; when dLogits is non-null, weight * dKL/dz is ADDED to it.
    static double rowLossAndGrad(const float* logits, std::size_t V, std::size_t K, const std::int32_t* ids,
                                 const float* p, float tail, float weight, float* dLogits);
};

} // namespace mimirmind::hugin::train
