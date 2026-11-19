// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "hugin/train/SwiGluGrad.hpp"

#include <cmath>

namespace mimirmind::hugin::train {

void SwiGluGrad::forward(const float* gate, const float* up, std::size_t n, float* y) {
    for (std::size_t i = 0; i < n; ++i) {
        const float g = gate[i];
        y[i] = g / (1.0F + std::exp(-g)) * up[i];
    }
}

void SwiGluGrad::backward(const float* dy, const float* gate, const float* up, std::size_t n,
                          float* dGate, float* dUp) {
    for (std::size_t i = 0; i < n; ++i) {
        const float g   = gate[i];
        const float sig = 1.0F / (1.0F + std::exp(-g));
        const float silu = g * sig;
        dUp[i]   = dy[i] * silu;
        dGate[i] = dy[i] * up[i] * sig * (1.0F + g * (1.0F - sig));
    }
}

} // namespace mimirmind::hugin::train
