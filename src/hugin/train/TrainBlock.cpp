// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "hugin/train/TrainBlock.hpp"

#include "hugin/train/LinearGrad.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace mimirmind::hugin::train {

namespace {
std::size_t idx(TrainBlock::Proj p) { return static_cast<std::size_t>(p); }
} // namespace

void TrainBlock::setLora(Proj p, std::size_t rank, float scale, std::vector<float> a, std::vector<float> b) {
    const auto [in, out] = projShape(p);
    if (rank == 0) throw std::runtime_error("TrainBlock: LoRA rank must be > 0");
    if (a.size() != rank * in || b.size() != out * rank) {
        throw std::runtime_error("TrainBlock: LoRA A must be [rank, " + std::to_string(in) + "], B [" +
                                 std::to_string(out) + ", rank]");
    }
    Lora l;
    l.rank  = rank;
    l.scale = scale;
    l.a     = std::move(a);
    l.b     = std::move(b);
    l.dA.assign(l.a.size(), 0.0F);
    l.dB.assign(l.b.size(), 0.0F);
    _lora[idx(p)] = std::move(l);
}

TrainBlock::Lora* TrainBlock::lora(Proj p) {
    return _lora[idx(p)] ? &*_lora[idx(p)] : nullptr;
}

const TrainBlock::Lora* TrainBlock::lora(Proj p) const {
    return _lora[idx(p)] ? &*_lora[idx(p)] : nullptr;
}

bool TrainBlock::hasLora() const noexcept {
    return std::any_of(_lora.begin(), _lora.end(), [](const auto& l) { return l.has_value(); });
}

void TrainBlock::zeroGrad() {
    for (auto& l : _lora) {
        if (!l) continue;
        std::fill(l->dA.begin(), l->dA.end(), 0.0F);
        std::fill(l->dB.begin(), l->dB.end(), 0.0F);
    }
}

void TrainBlock::clearLora() {
    for (auto& l : _lora) l.reset();
}

void TrainBlock::loraLinear(Proj p, const float* w, const float* bias, const float* in, std::size_t T,
                            float* out) const {
    const auto [K, N] = projShape(p);
    const Lora* l = lora(p);
    LoraParams lp;
    if (l != nullptr) lp = LoraParams{l->a.data(), l->b.data(), l->rank, l->scale};
    LinearGrad::forward(in, T, K, w, N, l != nullptr ? &lp : nullptr, out);
    if (bias != nullptr) {
        for (std::size_t t = 0; t < T; ++t) {
            for (std::size_t n = 0; n < N; ++n) out[t * N + n] += bias[n];
        }
    }
}

void TrainBlock::loraLinearBack(Proj p, const float* w, const float* dOut, const float* in, std::size_t T,
                                float* dIn, bool accumulate) {
    const auto [K, N] = projShape(p);
    Lora* l = lora(p);
    LoraParams lp;
    if (l != nullptr) lp = LoraParams{l->a.data(), l->b.data(), l->rank, l->scale};
    std::vector<float> tmp;
    float* dst = dIn;
    if (accumulate) {
        tmp.resize(T * K);
        dst = tmp.data();
    }
    LinearGrad::backward(dOut, in, T, K, w, N, l != nullptr ? &lp : nullptr, dst,
                         l != nullptr ? l->dA.data() : nullptr, l != nullptr ? l->dB.data() : nullptr);
    if (accumulate) {
        for (std::size_t i = 0; i < T * K; ++i) dIn[i] += tmp[i];
    }
}

} // namespace mimirmind::hugin::train
