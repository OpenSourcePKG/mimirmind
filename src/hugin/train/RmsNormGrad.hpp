// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstddef>

namespace mimirmind::hugin::train {

/**
 * Plain RMSNorm y = x * w / sqrt(mean(x^2) + eps) per row (not the Gemma
 * (1 + w) variant) and its backward (roadmap 7.4.1). Forward is the engine's
 * compute::rmsNorm. Per-head QK-norm is the same op with M = T * nHeads rows
 * of K = headDim and the shared head weight.
 */
class RmsNormGrad {
public:
    static void forward(const float* x, std::size_t M, std::size_t K,
                        const float* w, float eps, float* y);

    /// dx [M, K] written; dw [K] accumulated when non-null.
    static void backward(const float* dy, const float* x, std::size_t M, std::size_t K,
                         const float* w, float eps, float* dx, float* dw);
};

} // namespace mimirmind::hugin::train
