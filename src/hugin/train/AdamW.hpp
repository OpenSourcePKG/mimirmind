// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace mimirmind::hugin::train {

/**
 * AdamW (decoupled weight decay, bias-corrected moments) over one flat
 * parameter buffer (roadmap 7.4.1). Only adapter / edit parameters are ever
 * optimised; the frozen base has no optimizer state.
 */
class AdamW {
public:
    struct Config {
        float lr{1e-4F};
        float beta1{0.9F};
        float beta2{0.999F};
        float eps{1e-8F};
        float weightDecay{0.0F};
    };

    AdamW(std::size_t n, Config cfg);

    /// One step: params -= lr * (mhat / (sqrt(vhat) + eps) + wd * params).
    void step(std::span<float> params, std::span<const float> grads);

    [[nodiscard]] std::size_t steps() const { return _t; }

private:
    Config             _cfg;
    std::vector<float> _m;
    std::vector<float> _v;
    std::size_t        _t{0};
};

} // namespace mimirmind::hugin::train
