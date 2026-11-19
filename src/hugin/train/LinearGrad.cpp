// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "hugin/train/LinearGrad.hpp"

#include "hugin/train/Parallel.hpp"

#include <algorithm>
#include <vector>

namespace mimirmind::hugin::train {

namespace {

/// Explicit float -> double widening (the build runs with -Wdouble-promotion).
constexpr double f64(float x) { return x; }

/// Dot product with 8 independent float lanes (vectorisable without
/// -ffast-math), the lanes summed in double at the end.
double dot8(const float* a, const float* b, std::size_t n) {
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        for (std::size_t l = 0; l < 8; ++l) acc[l] += a[i + l] * b[i + l];
    }
    double s = 0;
    for (const float v : acc) s += f64(v);
    for (; i < n; ++i) s += f64(a[i]) * f64(b[i]);
    return s;
}

} // namespace

void LinearGrad::forward(const float* x, std::size_t M, std::size_t K,
                         const float* w, std::size_t N, const LoraParams* lora, float* y) {
    // Weight row outer (read once), token rows inner.
    Parallel::forRange(N, 16, [&](std::size_t n0, std::size_t n1) {
        for (std::size_t n = n0; n < n1; ++n) {
            const float* wr = w + n * K;
            for (std::size_t m = 0; m < M; ++m) y[m * N + n] = static_cast<float>(dot8(x + m * K, wr, K));
        }
    });
    if (lora == nullptr || lora->rank == 0) return;
    const std::size_t r = lora->rank;
    Parallel::forRange(M, 1, [&](std::size_t m0, std::size_t m1) {
        std::vector<float> h(r);
        for (std::size_t m = m0; m < m1; ++m) {
            const float* xr = x + m * K;
            for (std::size_t j = 0; j < r; ++j) h[j] = static_cast<float>(dot8(xr, lora->a + j * K, K));
            for (std::size_t n = 0; n < N; ++n) {
                y[m * N + n] += lora->scale * static_cast<float>(dot8(h.data(), lora->b + n * r, r));
            }
        }
    });
}

void LinearGrad::backward(const float* dy, const float* x, std::size_t M, std::size_t K,
                          const float* w, std::size_t N, const LoraParams* lora,
                          float* dx, float* dA, float* dB) {
    // dx = dy W: each worker owns a column range of dx and streams W's rows.
    Parallel::forRange(K, 64, [&](std::size_t k0, std::size_t k1) {
        const std::size_t len = k1 - k0;
        for (std::size_t m = 0; m < M; ++m) std::fill(dx + m * K + k0, dx + m * K + k1, 0.0F);
        for (std::size_t n = 0; n < N; ++n) {
            const float* wr = w + n * K + k0;
            for (std::size_t m = 0; m < M; ++m) {
                const float g = dy[m * N + n];
                if (g == 0.0F) continue;
                float* dr = dx + m * K + k0;
                for (std::size_t k = 0; k < len; ++k) dr[k] += g * wr[k];
            }
        }
    });
    if (lora == nullptr || lora->rank == 0) return;
    const std::size_t r = lora->rank;
    const float       s = lora->scale;
    // h = x A^T, g = dy B per row.
    std::vector<float> h(M * r), g(M * r);
    for (std::size_t m = 0; m < M; ++m) {
        for (std::size_t j = 0; j < r; ++j) {
            h[m * r + j] = static_cast<float>(dot8(x + m * K, lora->a + j * K, K));
            double gy = 0;
            for (std::size_t n = 0; n < N; ++n) gy += f64(dy[m * N + n]) * f64(lora->b[n * r + j]);
            g[m * r + j] = static_cast<float>(gy);
        }
    }
    // dx += s * g A ; dA += s * g^T x (column ranges of K, disjoint per worker).
    Parallel::forRange(K, 64, [&](std::size_t k0, std::size_t k1) {
        for (std::size_t m = 0; m < M; ++m) {
            for (std::size_t j = 0; j < r; ++j) {
                const float  gj = s * g[m * r + j];
                const float* ar = lora->a + j * K;
                float*       dr = dx + m * K;
                for (std::size_t k = k0; k < k1; ++k) dr[k] += gj * ar[k];
                if (dA != nullptr) {
                    const float* xr = x + m * K;
                    float*       da = dA + j * K;
                    for (std::size_t k = k0; k < k1; ++k) da[k] += gj * xr[k];
                }
            }
        }
    });
    // dB += s * dy^T h (row ranges of N).
    if (dB != nullptr) {
        Parallel::forRange(N, 256, [&](std::size_t n0, std::size_t n1) {
            for (std::size_t m = 0; m < M; ++m) {
                for (std::size_t n = n0; n < n1; ++n) {
                    const float gy = s * dy[m * N + n];
                    for (std::size_t j = 0; j < r; ++j) dB[n * r + j] += gy * h[m * r + j];
                }
            }
        });
    }
}

} // namespace mimirmind::hugin::train
