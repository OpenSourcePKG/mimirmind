// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "compute/Norm.hpp"

#include "core/log/Log.hpp"

#include <cmath>

namespace mimirmind::compute {

void rmsNorm(const float* x,
             std::size_t  M,
             std::size_t  K,
             const float* weight,
             float        eps,
             float*       y) {
    if (K == 0) {
        return;
    }
    const double invK = 1.0 / static_cast<double>(K);

    for (std::size_t m = 0; m < M; ++m) {
        const float* xr = x + m * K;
        float*       yr = y + m * K;

        double sumSq = 0.0;
        for (std::size_t k = 0; k < K; ++k) {
            const double v = static_cast<double>(xr[k]);
            sumSq += v * v;
        }
        const float invRms = 1.0F /
            std::sqrt(static_cast<float>(sumSq * invK) + eps);

        for (std::size_t k = 0; k < K; ++k) {
            yr[k] = xr[k] * weight[k] * invRms;
        }
    }
    MM_LOG_DEBUG("norm", "rmsNorm done — M={} K={} eps={}", M, K,
                 static_cast<double>(eps));
}

} // namespace mimirmind::compute