// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "hugin/train/RopeGrad.hpp"

#include <cmath>
#include <stdexcept>

namespace mimirmind::hugin::train {

void RopeGrad::rotate(float* x, std::size_t T, std::size_t nHeads, std::size_t headDim,
                      std::size_t startPos, float base, Layout layout, float sign) {
    if (headDim % 2 != 0) throw std::runtime_error("RopeGrad: headDim must be even");
    const std::size_t half   = headDim / 2;
    const float       invDim = 1.0F / static_cast<float>(headDim);
    for (std::size_t p = 0; p < T; ++p) {
        const float pos = static_cast<float>(startPos + p);
        for (std::size_t h = 0; h < nHeads; ++h) {
            float* head = x + (p * nHeads + h) * headDim;
            for (std::size_t i = 0; i < half; ++i) {
                // Same float expression as compute/Rope.cpp, so forward is bit-equal.
                const float freq  = std::pow(base, -static_cast<float>(2 * i) * invDim);
                const float theta = pos * freq;
                const float c     = std::cos(theta);
                const float s     = sign * std::sin(theta);
                const std::size_t ia = layout == Layout::Split ? i : 2 * i;
                const std::size_t ib = layout == Layout::Split ? i + half : 2 * i + 1;
                const float a = head[ia];
                const float b = head[ib];
                head[ia] = a * c - b * s;
                head[ib] = a * s + b * c;
            }
        }
    }
}

void RopeGrad::forward(float* x, std::size_t T, std::size_t nHeads, std::size_t headDim,
                       std::size_t startPos, float base, Layout layout) {
    rotate(x, T, nHeads, headDim, startPos, base, layout, 1.0F);
}

void RopeGrad::backward(float* d, std::size_t T, std::size_t nHeads, std::size_t headDim,
                        std::size_t startPos, float base, Layout layout) {
    rotate(d, T, nHeads, headDim, startPos, base, layout, -1.0F);
}

} // namespace mimirmind::hugin::train
