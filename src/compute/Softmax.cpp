// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "compute/Softmax.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace mimirmind::compute {

void softmaxRows(float*             data,
                 std::size_t        M,
                 std::size_t        K,
                 const std::size_t* causalKeepPerRow) noexcept {
    for (std::size_t m = 0; m < M; ++m) {
        float* row = data + m * K;
        const std::size_t live = causalKeepPerRow != nullptr
            ? std::min<std::size_t>(causalKeepPerRow[m], K)
            : K;
        if (live == 0) {
            // No live entries — zero the whole row.
            for (std::size_t k = 0; k < K; ++k) {
                row[k] = 0.0F;
            }
            continue;
        }

        float maxv = -std::numeric_limits<float>::infinity();
        for (std::size_t k = 0; k < live; ++k) {
            if (row[k] > maxv) {
                maxv = row[k];
            }
        }

        double sum = 0.0;
        for (std::size_t k = 0; k < live; ++k) {
            const float e = std::exp(row[k] - maxv);
            row[k] = e;
            sum += static_cast<double>(e);
        }
        const float invSum = static_cast<float>(1.0 / sum);
        for (std::size_t k = 0; k < live; ++k) {
            row[k] *= invSum;
        }
        for (std::size_t k = live; k < K; ++k) {
            row[k] = 0.0F;
        }
    }
}

} // namespace mimirmind::compute