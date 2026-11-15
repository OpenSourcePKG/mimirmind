// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "hugin/LogitMath.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace mimirmind::hugin {

void LogitMath::softcap(std::vector<float>& logits, float cap) noexcept {
    if (cap <= 0.0F) {
        return;
    }
    for (float& l : logits) {
        l = cap * std::tanh(l / cap);
    }
}

std::vector<double> LogitMath::logSoftmax(std::span<const float> logits) {
    std::vector<double> out(logits.size());
    if (logits.empty()) {
        return out;
    }
    double mx = -std::numeric_limits<double>::infinity();
    for (float l : logits) {
        mx = std::max(mx, static_cast<double>(l));
    }
    double sum = 0.0;
    for (float l : logits) {
        sum += std::exp(static_cast<double>(l) - mx);
    }
    const double lse = mx + std::log(sum);
    for (std::size_t i = 0; i < logits.size(); ++i) {
        out[i] = static_cast<double>(logits[i]) - lse;
    }
    return out;
}

std::vector<std::int32_t> LogitMath::topK(std::span<const double> v, std::size_t k) {
    std::vector<std::int32_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);
    k = std::min(k, idx.size());
    std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(k), idx.end(),
                      [&](std::int32_t a, std::int32_t b) {
                          const double va = v[static_cast<std::size_t>(a)];
                          const double vb = v[static_cast<std::size_t>(b)];
                          return va > vb || (va == vb && a < b);
                      });
    idx.resize(k);
    return idx;
}

std::size_t LogitMath::rankOf(std::span<const double> v, std::size_t index) {
    const double x = v[index];
    std::size_t rank = 0;
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (v[i] > x || (v[i] == x && i < index)) {
            ++rank;
        }
    }
    return rank;
}

double LogitMath::entropy(std::span<const double> logp) noexcept {
    double h = 0.0;
    for (double lp : logp) {
        if (std::isfinite(lp)) {
            h -= std::exp(lp) * lp;
        }
    }
    return h;
}

double LogitMath::kl(std::span<const double> logp, std::span<const double> logq) noexcept {
    double d = 0.0;
    const std::size_t n = std::min(logp.size(), logq.size());
    for (std::size_t i = 0; i < n; ++i) {
        if (std::isfinite(logp[i])) {
            d += std::exp(logp[i]) * (logp[i] - logq[i]);
        }
    }
    return std::max(0.0, d);
}

} // namespace mimirmind::hugin
