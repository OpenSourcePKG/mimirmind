// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstddef>

namespace mimirmind::hugin::train {

/// Low-rank adapter on a frozen linear: y += scale * (x A^T) B^T.
struct LoraParams {
    const float* a{nullptr};    ///< [rank, K]
    const float* b{nullptr};    ///< [N, rank]
    std::size_t  rank{0};
    float        scale{1.0F};
};

/**
 * Frozen linear y = x W^T (+ LoRA) and its backward (roadmap 7.4.1).
 *
 * Shapes (row-major F32): x [M, K], W [N, K] (engine layout: one row per
 * output feature), y [M, N]. The base weight never receives a gradient; only
 * the LoRA factors do. Inc-1 takes F32 weights; quantised weights are
 * dequantised row by row in 7.4.2.
 *
 * Gradient convention for all train ops: input gradients are WRITTEN,
 * parameter gradients are ACCUMULATED (+=), so micro-batches can add up.
 */
class LinearGrad {
public:
    static void forward(const float* x, std::size_t M, std::size_t K,
                        const float* w, std::size_t N, const LoraParams* lora, float* y);

    /// dx [M, K] written; dA [rank, K] and dB [N, rank] accumulated when non-null.
    static void backward(const float* dy, const float* x, std::size_t M, std::size_t K,
                         const float* w, std::size_t N, const LoraParams* lora,
                         float* dx, float* dA, float* dB);
};

} // namespace mimirmind::hugin::train
