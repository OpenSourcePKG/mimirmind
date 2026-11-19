// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstddef>

namespace mimirmind::hugin::train {

/**
 * Sigmoid-gated MoE combine for ONE token over its already selected experts
 * (Kolibri-1: sigmoid top-k, weights not renormalised; the selection bias
 * only picks experts and carries no gradient) and its backward (7.4.1).
 *
 *   out = scale * sum_e sigmoid(z_e) * x_e     z [k], x [k, D], out [D]
 *
 * The selection itself is frozen (non-differentiable); the gradient flows
 * into the selected router logits and the selected expert outputs.
 */
class MoeCombineGrad {
public:
    static void forward(const float* logits, const float* expertOut, std::size_t k,
                        std::size_t D, float scale, float* out);
    /// dLogits [k] and dExpertOut [k, D] written.
    static void backward(const float* dOut, const float* logits, const float* expertOut,
                         std::size_t k, std::size_t D, float scale,
                         float* dLogits, float* dExpertOut);
};

} // namespace mimirmind::hugin::train
