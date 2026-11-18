// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "hugin/SeparationMath.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace mimirmind::hugin {

namespace {

std::uint64_t splitmix64(std::uint64_t& s) noexcept {
    std::uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31U);
}

} // namespace

double SeparationMath::auc(std::span<const double> pos, std::span<const double> neg) {
    if (pos.empty() || neg.empty()) {
        return 0.5;
    }
    // Rank-sum over the pooled scores (average ranks for ties).
    std::vector<std::pair<double, bool>> all;
    all.reserve(pos.size() + neg.size());
    for (const double s : pos) all.emplace_back(s, true);
    for (const double s : neg) all.emplace_back(s, false);
    std::sort(all.begin(), all.end(),
              [](const auto& x, const auto& y) { return x.first < y.first; });
    double rankSumPos = 0;
    std::size_t i = 0;
    while (i < all.size()) {
        std::size_t j = i;
        while (j < all.size() && all[j].first == all[i].first) ++j;
        const double avgRank = (static_cast<double>(i) + static_cast<double>(j - 1)) / 2.0 + 1.0;
        for (std::size_t k = i; k < j; ++k) {
            if (all[k].second) rankSumPos += avgRank;
        }
        i = j;
    }
    const auto np = static_cast<double>(pos.size());
    const auto nn = static_cast<double>(neg.size());
    return (rankSumPos - np * (np + 1.0) / 2.0) / (np * nn);
}

SeparationReport SeparationMath::report(std::span<const double> pos, std::span<const double> neg,
                                        double threshold) {
    SeparationReport r;
    r.positives = pos.size();
    r.negatives = neg.size();
    r.auc       = auc(pos, neg);
    std::size_t correct = 0, negOn = 0;
    for (const double s : pos) correct += s > threshold ? 1U : 0U;
    for (const double s : neg) {
        const bool on = s > threshold;
        correct += on ? 0U : 1U;
        negOn += on ? 1U : 0U;
    }
    const std::size_t n = pos.size() + neg.size();
    r.accuracy         = n ? static_cast<double>(correct) / static_cast<double>(n) : 0.0;
    r.negativeFireRate = neg.empty() ? 0.0 : static_cast<double>(negOn) / static_cast<double>(neg.size());
    return r;
}

double SeparationMath::meanDifferenceAuc(std::span<const float> rows, std::size_t d,
                                         std::span<const std::uint8_t> isPositive) {
    if (d == 0 || rows.size() != isPositive.size() * d) {
        throw std::invalid_argument("meanDifferenceAuc: rows must be [labels, d]");
    }
    const std::size_t n = isPositive.size();
    std::vector<double> mp(d, 0.0), mn(d, 0.0);
    std::size_t np = 0, nn = 0;
    for (std::size_t r = 0; r < n; ++r) {
        auto& m = isPositive[r] ? mp : mn;
        (isPositive[r] ? np : nn) += 1;
        for (std::size_t i = 0; i < d; ++i) m[i] += static_cast<double>(rows[r * d + i]);
    }
    if (np == 0 || nn == 0) {
        return 0.5;
    }
    std::vector<double> dir(d);
    for (std::size_t i = 0; i < d; ++i) {
        dir[i] = mp[i] / static_cast<double>(np) - mn[i] / static_cast<double>(nn);
    }
    std::vector<double> ps, ns;
    ps.reserve(np);
    ns.reserve(nn);
    for (std::size_t r = 0; r < n; ++r) {
        double s = 0;
        for (std::size_t i = 0; i < d; ++i) s += static_cast<double>(rows[r * d + i]) * dir[i];
        (isPositive[r] ? ps : ns).push_back(s);
    }
    return auc(ps, ns);
}

NullStats SeparationMath::meanDifferenceNull(std::span<const float> rows, std::size_t d,
                                             std::size_t positives, std::size_t permutations,
                                             std::uint64_t seed) {
    if (d == 0 || rows.size() % d != 0) {
        throw std::invalid_argument("meanDifferenceNull: rows must be [n, d]");
    }
    const std::size_t n = rows.size() / d;
    std::vector<double> aucs;
    aucs.reserve(permutations);
    for (std::size_t p = 0; p < permutations; ++p) {
        const auto perm = permutation(n, seed + p);
        std::vector<std::uint8_t> lab(n, 0);
        for (std::size_t k = 0; k < positives && k < n; ++k) lab[perm[k]] = 1;
        aucs.push_back(meanDifferenceAuc(rows, d, lab));
    }
    return summarize(std::move(aucs));
}

std::vector<std::size_t> SeparationMath::permutation(std::size_t n, std::uint64_t seed) {
    std::vector<std::size_t> v(n);
    std::iota(v.begin(), v.end(), std::size_t{0});
    std::uint64_t s = seed;
    for (std::size_t i = n; i > 1; --i) {
        const std::size_t j = static_cast<std::size_t>(splitmix64(s) % i);
        std::swap(v[i - 1], v[j]);
    }
    return v;
}

NullStats SeparationMath::summarize(std::vector<double> values) {
    NullStats st;
    st.permutations = values.size();
    if (values.empty()) {
        st.mean = st.p95 = st.max = 0.0;
        return st;
    }
    std::sort(values.begin(), values.end());
    st.mean = std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
    st.max  = values.back();
    const auto idx = static_cast<std::size_t>(
        std::ceil(0.95 * static_cast<double>(values.size()))) - 1;
    st.p95 = values[std::min(idx, values.size() - 1)];
    return st;
}

} // namespace mimirmind::hugin
