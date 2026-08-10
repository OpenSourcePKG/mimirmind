// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "compute/Activations.hpp"

#include <cmath>

namespace mimirmind::compute {

void geluErfInPlace(float* x, std::size_t n) noexcept {
    // 1/sqrt(2). erf-GELU (BERT/RoBERTa/XLM-R), matches PyTorch/HF nn.GELU.
    constexpr float kInvSqrt2 = 0.70710678118654752440F;
    for (std::size_t i = 0; i < n; ++i) {
        const float v = x[i];
        x[i] = 0.5F * v * (1.0F + std::erf(v * kInvSqrt2));
    }
}

void tanhInPlace(float* x, std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = std::tanh(x[i]);
    }
}

void siluInPlace(float* x, std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i) {
        const float v = x[i];
        x[i] = v / (1.0F + std::exp(-v));
    }
}

void mulInPlace(float* a, const float* b, std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i) {
        a[i] *= b[i];
    }
}

} // namespace mimirmind::compute