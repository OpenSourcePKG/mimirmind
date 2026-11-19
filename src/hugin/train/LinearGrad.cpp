// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "hugin/train/LinearGrad.hpp"

#include <vector>


namespace {
/// Explicit float -> double widening (the build runs with -Wdouble-promotion).
constexpr double f64(float x) { return x; }
} // namespace

namespace mimirmind::hugin::train {

void LinearGrad::forward(const float* x, std::size_t M, std::size_t K,
                         const float* w, std::size_t N, const LoraParams* lora, float* y) {
    for (std::size_t m = 0; m < M; ++m) {
        const float* xr = x + m * K;
        for (std::size_t n = 0; n < N; ++n) {
            const float* wr = w + n * K;
            double acc = 0;
            for (std::size_t k = 0; k < K; ++k) acc += f64(xr[k]) * f64(wr[k]);
            y[m * N + n] = static_cast<float>(acc);
        }
    }
    if (lora == nullptr || lora->rank == 0) return;
    const std::size_t r = lora->rank;
    std::vector<float> h(r);
    for (std::size_t m = 0; m < M; ++m) {
        const float* xr = x + m * K;
        for (std::size_t j = 0; j < r; ++j) {
            double acc = 0;
            for (std::size_t k = 0; k < K; ++k) acc += f64(xr[k]) * f64(lora->a[j * K + k]);
            h[j] = static_cast<float>(acc);
        }
        for (std::size_t n = 0; n < N; ++n) {
            double acc = 0;
            for (std::size_t j = 0; j < r; ++j) acc += f64(h[j]) * f64(lora->b[n * r + j]);
            y[m * N + n] += lora->scale * static_cast<float>(acc);
        }
    }
}

void LinearGrad::backward(const float* dy, const float* x, std::size_t M, std::size_t K,
                          const float* w, std::size_t N, const LoraParams* lora,
                          float* dx, float* dA, float* dB) {
    // dx = dy W
    for (std::size_t m = 0; m < M; ++m) {
        for (std::size_t k = 0; k < K; ++k) {
            double acc = 0;
            for (std::size_t n = 0; n < N; ++n) acc += f64(dy[m * N + n]) * f64(w[n * K + k]);
            dx[m * K + k] = static_cast<float>(acc);
        }
    }
    if (lora == nullptr || lora->rank == 0) return;
    const std::size_t r = lora->rank;
    const float       s = lora->scale;
    std::vector<float> h(r), g(r);   // h = x A^T, g = dy B (per row)
    for (std::size_t m = 0; m < M; ++m) {
        const float* xr  = x + m * K;
        const float* dyr = dy + m * N;
        for (std::size_t j = 0; j < r; ++j) {
            double hx = 0, gy = 0;
            for (std::size_t k = 0; k < K; ++k) hx += f64(xr[k]) * f64(lora->a[j * K + k]);
            for (std::size_t n = 0; n < N; ++n) gy += f64(dyr[n]) * f64(lora->b[n * r + j]);
            h[j] = static_cast<float>(hx);
            g[j] = static_cast<float>(gy);
        }
        // dx += s * g A
        for (std::size_t k = 0; k < K; ++k) {
            double acc = 0;
            for (std::size_t j = 0; j < r; ++j) acc += f64(g[j]) * f64(lora->a[j * K + k]);
            dx[m * K + k] += s * static_cast<float>(acc);
        }
        if (dA != nullptr) {
            for (std::size_t j = 0; j < r; ++j) {
                for (std::size_t k = 0; k < K; ++k) dA[j * K + k] += s * g[j] * xr[k];
            }
        }
        if (dB != nullptr) {
            for (std::size_t n = 0; n < N; ++n) {
                for (std::size_t j = 0; j < r; ++j) dB[n * r + j] += s * dyr[n] * h[j];
            }
        }
    }
}

} // namespace mimirmind::hugin::train
