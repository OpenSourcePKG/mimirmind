// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "hugin/TreeParzenEstimator.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace mimirmind::hugin {

namespace {

constexpr double kPi = 3.141592653589793;

/// a dominates b (minimisation): no worse everywhere, better somewhere.
bool dominates(const std::vector<double>& a, const std::vector<double>& b) {
    bool better = false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i] > b[i]) return false;
        if (a[i] < b[i]) better = true;
    }
    return better;
}

double toInternal(const TreeParzenEstimator::Param& p, double v) {
    return p.log ? std::log(v) : v;
}

double fromInternal(const TreeParzenEstimator::Param& p, double u) {
    double v = p.log ? std::exp(u) : u;
    v = std::clamp(v, p.lo, p.hi);
    if (p.kind == TreeParzenEstimator::Param::Kind::Int) {
        v = std::clamp(std::round(v), p.lo, p.hi);
    }
    return v;
}

/// Parzen estimator over [lo, hi]: one Gaussian per observation plus a broad
/// prior component (mean = middle, sigma = range), equal weights. Each
/// observation's bandwidth is the larger gap to its sorted neighbours
/// (Optuna's rule), clipped to [range / min(100, n + 1), range].
struct Parzen {
    std::vector<double> mu;
    std::vector<double> sigma;

    Parzen(std::vector<double> obs, double lo, double hi) {
        const double range = hi - lo;
        obs.push_back((lo + hi) / 2.0);   // prior
        std::vector<std::size_t> idx(obs.size());
        std::iota(idx.begin(), idx.end(), std::size_t{0});
        std::sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) { return obs[a] < obs[b]; });
        std::vector<double> sg(obs.size(), range);
        const double minS = range / std::min(100.0, static_cast<double>(obs.size()));
        for (std::size_t r = 0; r < idx.size(); ++r) {
            const double left  = r > 0 ? obs[idx[r]] - obs[idx[r - 1]] : obs[idx[r]] - lo;
            const double right = r + 1 < idx.size() ? obs[idx[r + 1]] - obs[idx[r]] : hi - obs[idx[r]];
            sg[idx[r]] = std::clamp(std::max(left, right), minS, range);
        }
        sg.back() = range;                 // the prior stays broad
        mu    = std::move(obs);
        sigma = std::move(sg);
    }

    [[nodiscard]] double logPdf(double x) const {
        double sum = 0;
        for (std::size_t i = 0; i < mu.size(); ++i) {
            const double z = (x - mu[i]) / sigma[i];
            sum += std::exp(-0.5 * z * z) / (sigma[i] * std::sqrt(2.0 * kPi));
        }
        return std::log(sum / static_cast<double>(mu.size()) + 1e-300);
    }
};

} // namespace

TreeParzenEstimator::TreeParzenEstimator(std::vector<Param> space, std::size_t objectives, Config cfg)
    : _space(std::move(space)), _objectives(objectives), _cfg(cfg), _state(cfg.seed) {
    if (_space.empty() || _objectives == 0) {
        throw std::invalid_argument("TPE: empty search space or no objectives");
    }
    if (!(_cfg.gamma > 0.0 && _cfg.gamma < 1.0) || _cfg.candidates == 0) {
        throw std::invalid_argument("TPE: gamma must be in (0, 1) and candidates > 0");
    }
    for (const auto& p : _space) {
        if (p.kind == Param::Kind::Choice) {
            if (p.choices == 0) throw std::invalid_argument("TPE: choice '" + p.name + "' has no values");
        } else if (!(p.lo <= p.hi) || (p.log && !(p.lo > 0.0))) {
            throw std::invalid_argument("TPE: bad bounds for '" + p.name + "'");
        }
    }
}

double TreeParzenEstimator::uniform01() {
    std::uint64_t z = (_state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
    return static_cast<double>((z ^ (z >> 31U)) >> 11U) / 9007199254740992.0;
}

double TreeParzenEstimator::normal() {
    const double u1 = std::max(uniform01(), 1e-300);
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * kPi * uniform01());
}

double TreeParzenEstimator::randomValue(const Param& p) {
    if (p.kind == Param::Kind::Choice) {
        return std::min(std::floor(uniform01() * static_cast<double>(p.choices)),
                        static_cast<double>(p.choices - 1));
    }
    const double lo = toInternal(p, p.lo), hi = toInternal(p, p.hi);
    if (p.kind == Param::Kind::Int && !p.log) {
        // Uniform over the integers lo..hi.
        return std::clamp(std::floor(p.lo + uniform01() * (p.hi - p.lo + 1.0)), p.lo, p.hi);
    }
    return fromInternal(p, lo + uniform01() * (hi - lo));
}

double TreeParzenEstimator::sampleParam(std::size_t pi, const std::vector<std::size_t>& good,
                                        const std::vector<std::size_t>& bad) {
    const Param& p = _space[pi];
    if (p.kind == Param::Kind::Choice) {
        std::vector<double> wg(p.choices, 1.0), wb(p.choices, 1.0);   // +1 smoothing
        for (const auto i : good) wg[static_cast<std::size_t>(_trials[i].params[pi])] += 1.0;
        for (const auto i : bad) wb[static_cast<std::size_t>(_trials[i].params[pi])] += 1.0;
        const double sg = std::accumulate(wg.begin(), wg.end(), 0.0);
        const double sb = std::accumulate(wb.begin(), wb.end(), 0.0);
        double best = -std::numeric_limits<double>::infinity();
        double pick = 0;
        for (std::size_t c = 0; c < _cfg.candidates; ++c) {
            double r = uniform01() * sg;
            std::size_t k = 0;
            while (k + 1 < p.choices && r >= wg[k]) { r -= wg[k]; ++k; }
            const double score = std::log(wg[k] / sg) - std::log(wb[k] / sb);
            if (score > best) { best = score; pick = static_cast<double>(k); }
        }
        return pick;
    }
    const double lo = toInternal(p, p.lo), hi = toInternal(p, p.hi);
    std::vector<double> cg, cb;
    for (const auto i : good) cg.push_back(toInternal(p, _trials[i].params[pi]));
    for (const auto i : bad) cb.push_back(toInternal(p, _trials[i].params[pi]));
    const Parzen l(cg, lo, hi), g(cb, lo, hi);
    double best = -std::numeric_limits<double>::infinity();
    double pick = (lo + hi) / 2.0;
    for (std::size_t c = 0; c < _cfg.candidates; ++c) {
        // Draw from l(x): pick a component, sample, redraw out-of-range values.
        const auto k = std::min(static_cast<std::size_t>(uniform01() * static_cast<double>(l.mu.size())),
                                l.mu.size() - 1);
        double u = l.mu[k];
        for (int tries = 0; tries < 32; ++tries) {
            u = l.mu[k] + l.sigma[k] * normal();
            if (u >= lo && u <= hi) break;
        }
        u = std::clamp(u, lo, hi);
        const double score = l.logPdf(u) - g.logPdf(u);
        if (score > best) { best = score; pick = u; }
    }
    return fromInternal(p, pick);
}

std::vector<double> TreeParzenEstimator::suggest() {
    std::vector<double> out(_space.size());
    if (_trials.size() < _cfg.startupTrials) {
        for (std::size_t i = 0; i < _space.size(); ++i) out[i] = randomValue(_space[i]);
        return out;
    }
    const auto order = rank(_trials);
    const auto nGood = std::max<std::size_t>(
        1, static_cast<std::size_t>(std::ceil(_cfg.gamma * static_cast<double>(order.size()))));
    const std::vector<std::size_t> good(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(nGood));
    const std::vector<std::size_t> bad(order.begin() + static_cast<std::ptrdiff_t>(nGood), order.end());
    for (std::size_t i = 0; i < _space.size(); ++i) out[i] = sampleParam(i, good, bad);
    return out;
}

void TreeParzenEstimator::tell(std::vector<double> params, std::vector<double> objectives, bool feasible) {
    if (params.size() != _space.size() || objectives.size() != _objectives) {
        throw std::invalid_argument("TPE: tell() with wrong parameter / objective count");
    }
    for (double& o : objectives) {
        if (!std::isfinite(o)) {
            o        = std::numeric_limits<double>::max() / 4;
            feasible = false;
        }
    }
    _trials.push_back({std::move(params), std::move(objectives), feasible});
}

std::vector<std::size_t> TreeParzenEstimator::paretoFront(const std::vector<Trial>& trials) {
    std::vector<std::size_t> front;
    for (std::size_t i = 0; i < trials.size(); ++i) {
        if (!trials[i].feasible) continue;
        bool dominated = false;
        for (std::size_t j = 0; j < trials.size() && !dominated; ++j) {
            dominated = j != i && trials[j].feasible && dominates(trials[j].objectives, trials[i].objectives);
        }
        if (!dominated) front.push_back(i);
    }
    return front;
}

std::vector<std::size_t> TreeParzenEstimator::rank(const std::vector<Trial>& trials) {
    const std::size_t n = trials.size();
    if (n == 0) return {};
    const std::size_t m = trials.front().objectives.size();
    // Non-domination rank among feasible trials; infeasible trials share the last rank.
    std::vector<std::size_t> level(n, std::numeric_limits<std::size_t>::max());
    std::vector<std::size_t> rest;
    for (std::size_t i = 0; i < n; ++i) if (trials[i].feasible) rest.push_back(i);
    for (std::size_t lv = 0; !rest.empty(); ++lv) {
        std::vector<std::size_t> next, cur;
        for (const auto i : rest) {
            bool dominated = false;
            for (const auto j : rest) {
                if (j != i && dominates(trials[j].objectives, trials[i].objectives)) { dominated = true; break; }
            }
            (dominated ? next : cur).push_back(i);
        }
        for (const auto i : cur) level[i] = lv;
        rest = std::move(next);
    }
    std::vector<double> lo(m, std::numeric_limits<double>::infinity()), hi(m, -std::numeric_limits<double>::infinity());
    for (const auto& t : trials) {
        for (std::size_t k = 0; k < m; ++k) {
            lo[k] = std::min(lo[k], t.objectives[k]);
            hi[k] = std::max(hi[k], t.objectives[k]);
        }
    }
    std::vector<double> sum(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t k = 0; k < m; ++k) {
            const double r = hi[k] - lo[k];
            sum[i] += r > 0 ? (trials[i].objectives[k] - lo[k]) / r : 0.0;
        }
    }
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        if (level[a] != level[b]) return level[a] < level[b];
        return sum[a] < sum[b];
    });
    return order;
}

} // namespace mimirmind::hugin
