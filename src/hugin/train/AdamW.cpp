// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "hugin/train/AdamW.hpp"

#include <cmath>
#include <stdexcept>


namespace {
/// Explicit float -> double widening (the build runs with -Wdouble-promotion).
constexpr double f64(float x) { return x; }
} // namespace

namespace mimirmind::hugin::train {

AdamW::AdamW(std::size_t n, Config cfg) : _cfg(cfg), _m(n, 0.0F), _v(n, 0.0F) {}

void AdamW::step(std::span<float> params, std::span<const float> grads) {
    if (params.size() != _m.size() || grads.size() != _m.size()) {
        throw std::runtime_error("AdamW: params/grads size differs from the optimizer state");
    }
    ++_t;
    const double bc1 = 1.0 - std::pow(static_cast<double>(_cfg.beta1), static_cast<double>(_t));
    const double bc2 = 1.0 - std::pow(static_cast<double>(_cfg.beta2), static_cast<double>(_t));
    for (std::size_t i = 0; i < params.size(); ++i) {
        _m[i] = _cfg.beta1 * _m[i] + (1.0F - _cfg.beta1) * grads[i];
        _v[i] = _cfg.beta2 * _v[i] + (1.0F - _cfg.beta2) * grads[i] * grads[i];
        const double mhat = f64(_m[i]) / bc1;
        const double vhat = f64(_v[i]) / bc2;
        params[i] -= static_cast<float>(f64(_cfg.lr) * (mhat / (std::sqrt(vhat) + f64(_cfg.eps))
                                                         + f64(_cfg.weightDecay) * f64(params[i])));
    }
}

} // namespace mimirmind::hugin::train
