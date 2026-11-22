// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "hugin/ExpertSaliency.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>

namespace mimirmind::hugin {

namespace {

std::vector<std::size_t> lowestBy(const std::vector<ExpertStats>& stats, std::size_t n,
                                  bool byReap) {
    std::vector<std::size_t> order(stats.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        const ExpertStats& x = stats[a];
        const ExpertStats& y = stats[b];
        if (byReap) {
            if (x.reap != y.reap) return x.reap < y.reap;
            return x.count < y.count;
        }
        if (x.count != y.count) return x.count < y.count;
        return x.weightSum < y.weightSum;
    });
    order.resize(std::min(n, order.size()));
    std::sort(order.begin(), order.end());
    return order;
}

} // namespace

ExpertSaliency::ExpertSaliency(std::size_t experts) : _stats(experts) {
    if (experts == 0) {
        throw std::invalid_argument("ExpertSaliency: experts must be > 0");
    }
    for (std::size_t j = 0; j < experts; ++j) {
        _stats[j].expert = j;
    }
}

void ExpertSaliency::add(std::span<const float> idx, std::span<const float> weight,
                         std::span<const float> norm, std::size_t topK) {
    if (topK == 0 || idx.size() % topK != 0 || weight.size() != idx.size() ||
        norm.size() != idx.size()) {
        throw std::invalid_argument("ExpertSaliency::add: idx / weight / norm must all be [tokens, topK]");
    }
    for (std::size_t i = 0; i < idx.size(); ++i) {
        const float id = idx[i];
        if (!(id >= 0.0F) || static_cast<std::size_t>(id) >= _stats.size()) {
            throw std::out_of_range("ExpertSaliency::add: expert id " + std::to_string(id) +
                                    " out of range");
        }
        ExpertStats& s = _stats[static_cast<std::size_t>(id)];
        ++s.count;
        s.weightSum += weight[i];
        s.normSum   += norm[i];
        s.reap      += static_cast<double>(weight[i]) * static_cast<double>(norm[i]);  // sum; mean in stats()
    }
    _tokens += idx.size() / topK;
}

std::vector<ExpertStats> ExpertSaliency::stats() const {
    std::vector<ExpertStats> out = _stats;
    for (auto& s : out) {
        s.reap = s.count == 0 ? 0.0 : s.reap / static_cast<double>(s.count);
    }
    return out;
}

std::vector<std::size_t> ExpertSaliency::pruneByReap(std::size_t n) const {
    return lowestBy(stats(), n, true);
}

std::vector<std::size_t> ExpertSaliency::pruneByFrequency(std::size_t n) const {
    return lowestBy(stats(), n, false);
}

double ExpertSaliency::jaccard(std::span<const std::size_t> a, std::span<const std::size_t> b) {
    std::vector<std::size_t> x(a.begin(), a.end()), y(b.begin(), b.end());
    std::sort(x.begin(), x.end());
    std::sort(y.begin(), y.end());
    x.erase(std::unique(x.begin(), x.end()), x.end());
    y.erase(std::unique(y.begin(), y.end()), y.end());
    std::vector<std::size_t> inter, uni;
    std::set_intersection(x.begin(), x.end(), y.begin(), y.end(), std::back_inserter(inter));
    std::set_union(x.begin(), x.end(), y.begin(), y.end(), std::back_inserter(uni));
    return uni.empty() ? 1.0 : static_cast<double>(inter.size()) / static_cast<double>(uni.size());
}

} // namespace mimirmind::hugin
