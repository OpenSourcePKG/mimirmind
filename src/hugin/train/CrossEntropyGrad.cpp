// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "hugin/train/CrossEntropyGrad.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>


namespace {
/// Explicit float -> double widening (the build runs with -Wdouble-promotion).
constexpr double f64(float x) { return x; }
} // namespace

namespace mimirmind::hugin::train {

double CrossEntropyGrad::lossAndGrad(const float* logits, const std::int32_t* targets,
                                     std::size_t T, std::size_t V, float* dLogits) {
    std::size_t counted = 0;
    for (std::size_t t = 0; t < T; ++t) counted += targets[t] >= 0 ? 1U : 0U;
    if (dLogits != nullptr) std::fill(dLogits, dLogits + T * V, 0.0F);
    if (counted == 0) return 0.0;
    const double inv = 1.0 / static_cast<double>(counted);
    double loss = 0;
    for (std::size_t t = 0; t < T; ++t) {
        if (targets[t] < 0) continue;
        if (static_cast<std::size_t>(targets[t]) >= V) {
            throw std::runtime_error("CrossEntropyGrad: target id out of range");
        }
        const float* row = logits + t * V;
        const float  mx  = *std::max_element(row, row + V);
        double sum = 0;
        for (std::size_t v = 0; v < V; ++v) sum += std::exp(f64(row[v]) - f64(mx));
        const double lse = f64(mx) + std::log(sum);
        loss += lse - f64(row[targets[t]]);
        if (dLogits != nullptr) {
            float* d = dLogits + t * V;
            for (std::size_t v = 0; v < V; ++v) {
                d[v] = static_cast<float>(std::exp(static_cast<double>(row[v]) - lse) * inv);
            }
            d[targets[t]] -= static_cast<float>(inv);
        }
    }
    return loss * inv;
}

} // namespace mimirmind::hugin::train
