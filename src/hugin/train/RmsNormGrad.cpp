// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "hugin/train/RmsNormGrad.hpp"

#include "compute/Norm.hpp"

#include <cmath>


namespace {
/// Explicit float -> double widening (the build runs with -Wdouble-promotion).
constexpr double f64(float x) { return x; }
} // namespace

namespace mimirmind::hugin::train {

void RmsNormGrad::forward(const float* x, std::size_t M, std::size_t K,
                          const float* w, float eps, float* y) {
    compute::rmsNorm(x, M, K, w, eps, y);
}

void RmsNormGrad::backward(const float* dy, const float* x, std::size_t M, std::size_t K,
                           const float* w, float eps, float* dx, float* dw) {
    // r = (mean(x^2) + eps)^-1/2;  dx_i = r (dy_i w_i) - r^3 x_i / K * sum_j dy_j w_j x_j
    for (std::size_t m = 0; m < M; ++m) {
        const float* xr  = x + m * K;
        const float* dyr = dy + m * K;
        double ss = 0, dot = 0;
        for (std::size_t k = 0; k < K; ++k) {
            ss  += f64(xr[k]) * f64(xr[k]);
            dot += f64(dyr[k]) * f64(w[k]) * f64(xr[k]);
        }
        const double r  = 1.0 / std::sqrt(ss / static_cast<double>(K) + f64(eps));
        const double c  = r * r * r * dot / static_cast<double>(K);
        for (std::size_t k = 0; k < K; ++k) {
            dx[m * K + k] = static_cast<float>(r * f64(dyr[k]) * f64(w[k]) - c * f64(xr[k]));
            if (dw != nullptr) dw[k] += static_cast<float>(f64(dyr[k] * xr[k]) * r);
        }
    }
}

} // namespace mimirmind::hugin::train
