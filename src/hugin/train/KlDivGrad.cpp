// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "hugin/train/KlDivGrad.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {
/// Explicit float -> double widening (the build runs with -Wdouble-promotion).
constexpr double f64(float x) { return x; }
/// Mass below this counts as zero (no log of a vanishing bucket).
constexpr double kTiny = 1e-30;

double logSumExp(const float* z, std::size_t V) {
    const float mx = *std::max_element(z, z + V);
    double s = 0;
    for (std::size_t v = 0; v < V; ++v) s += std::exp(f64(z[v]) - f64(mx));
    return f64(mx) + std::log(s);
}
} // namespace

namespace mimirmind::hugin::train {

void KlDivGrad::topK(const float* logits, std::size_t V, std::size_t K, std::int32_t* ids, float* p,
                     float& tail) {
    if (K == 0 || K > V) throw std::runtime_error("KlDivGrad: K must be in [1, V]");
    std::vector<std::int32_t> idx(V);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(K), idx.end(),
                      [&](std::int32_t a, std::int32_t b) { return logits[a] > logits[b]; });
    const double lse = logSumExp(logits, V);
    double sum = 0;
    for (std::size_t k = 0; k < K; ++k) {
        ids[k] = idx[k];
        const double pk = std::exp(f64(logits[idx[k]]) - lse);
        p[k] = static_cast<float>(pk);
        sum += pk;
    }
    tail = static_cast<float>(std::max(0.0, 1.0 - sum));
}

double KlDivGrad::rowLossAndGrad(const float* logits, std::size_t V, std::size_t K, const std::int32_t* ids,
                                 const float* p, float tail, float weight, float* dLogits) {
    const double lse = logSumExp(logits, V);
    // q on the top-K, q_tail summed directly over the rest (no 1 - sum cancellation).
    std::vector<char> inTop(V, 0);
    double loss = 0;
    for (std::size_t k = 0; k < K; ++k) {
        if (ids[k] < 0 || static_cast<std::size_t>(ids[k]) >= V) {
            throw std::runtime_error("KlDivGrad: token id out of range");
        }
        inTop[static_cast<std::size_t>(ids[k])] = 1;
        const double pk = f64(p[k]);
        if (pk > kTiny) loss += pk * (std::log(pk) - (f64(logits[ids[k]]) - lse));
    }
    double qTail = 0;
    for (std::size_t v = 0; v < V; ++v) {
        if (inTop[v] == 0) qTail += std::exp(f64(logits[v]) - lse);
    }
    const double pt = f64(tail);
    if (pt > kTiny) loss += pt * (std::log(pt) - std::log(std::max(qTail, kTiny)));

    if (dLogits != nullptr) {
        const double w = f64(weight);
        const double tailScale = (pt > kTiny && qTail > kTiny) ? pt / qTail : 0.0;
        for (std::size_t v = 0; v < V; ++v) {
            const double q = std::exp(f64(logits[v]) - lse);
            // Off the top-K: q_j - p_tail * q_j / q_tail; on it the p_k is subtracted below.
            const double g = inTop[v] != 0 ? q : q * (1.0 - tailScale);
            dLogits[v] += static_cast<float>(w * g);
        }
        for (std::size_t k = 0; k < K; ++k) dLogits[ids[k]] -= static_cast<float>(w * f64(p[k]));
    }
    return loss;
}

} // namespace mimirmind::hugin::train
