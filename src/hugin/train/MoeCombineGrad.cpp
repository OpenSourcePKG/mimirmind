// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "hugin/train/MoeCombineGrad.hpp"

#include <algorithm>
#include <cmath>


namespace {
/// Explicit float -> double widening (the build runs with -Wdouble-promotion).
constexpr double f64(float x) { return x; }
} // namespace

namespace mimirmind::hugin::train {

void MoeCombineGrad::forward(const float* logits, const float* expertOut, std::size_t k,
                             std::size_t D, float scale, float* out) {
    std::fill(out, out + D, 0.0F);
    for (std::size_t e = 0; e < k; ++e) {
        const float g = scale / (1.0F + std::exp(-logits[e]));
        for (std::size_t d = 0; d < D; ++d) out[d] += g * expertOut[e * D + d];
    }
}

void MoeCombineGrad::backward(const float* dOut, const float* logits, const float* expertOut,
                              std::size_t k, std::size_t D, float scale,
                              float* dLogits, float* dExpertOut) {
    for (std::size_t e = 0; e < k; ++e) {
        const float sig = 1.0F / (1.0F + std::exp(-logits[e]));
        double dot = 0;
        for (std::size_t d = 0; d < D; ++d) {
            dot += f64(dOut[d]) * f64(expertOut[e * D + d]);
            dExpertOut[e * D + d] = scale * sig * dOut[d];
        }
        dLogits[e] = static_cast<float>(f64(scale) * dot * f64(sig * (1.0F - sig)));
    }
}

} // namespace mimirmind::hugin::train
