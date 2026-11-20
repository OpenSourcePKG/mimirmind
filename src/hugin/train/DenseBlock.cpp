// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "hugin/train/DenseBlock.hpp"

#include "hugin/train/AttentionGrad.hpp"
#include "hugin/train/RmsNormGrad.hpp"
#include "hugin/train/SwiGluGrad.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace mimirmind::hugin::train {

namespace {

void requireSize(const std::vector<float>& v, std::size_t n, const char* what) {
    if (v.size() != n) {
        throw std::runtime_error(std::string("DenseBlock: '") + what + "' has " +
                                 std::to_string(v.size()) + " values, expected " + std::to_string(n));
    }
}

} // namespace

DenseBlock::DenseBlock(Config cfg, Weights w) : _cfg(std::move(cfg)), _w(std::move(w)) {
    const std::size_t d = _cfg.dModel, qd = _cfg.heads * _cfg.headDim, kd = _cfg.kvHeads * _cfg.headDim;
    if (d == 0 || _cfg.heads == 0 || _cfg.kvHeads == 0 || _cfg.headDim == 0 || _cfg.ff == 0 ||
        _cfg.heads % _cfg.kvHeads != 0) {
        throw std::runtime_error("DenseBlock: invalid shape config");
    }
    if (!_cfg.ropeFactors.empty() && _cfg.ropeFactors.size() != _cfg.headDim / 2) {
        throw std::runtime_error("DenseBlock: ropeFactors must have headDim/2 values");
    }
    requireSize(_w.attnNorm, d, "attn_norm");
    requireSize(_w.wq, qd * d, "attn_q");
    requireSize(_w.wk, kd * d, "attn_k");
    requireSize(_w.wv, kd * d, "attn_v");
    requireSize(_w.wo, d * qd, "attn_output");
    requireSize(_w.ffnNorm, d, "ffn_norm");
    requireSize(_w.wGate, _cfg.ff * d, "ffn_gate");
    requireSize(_w.wUp, _cfg.ff * d, "ffn_up");
    requireSize(_w.wDown, d * _cfg.ff, "ffn_down");
    if (!_w.bq.empty()) requireSize(_w.bq, qd, "attn_q.bias");
    if (!_w.bk.empty()) requireSize(_w.bk, kd, "attn_k.bias");
    if (!_w.bv.empty()) requireSize(_w.bv, kd, "attn_v.bias");
}

std::pair<std::size_t, std::size_t> DenseBlock::projShape(Proj p) const {
    const std::size_t d = _cfg.dModel, qd = _cfg.heads * _cfg.headDim, kd = _cfg.kvHeads * _cfg.headDim;
    switch (p) {
        case Proj::Q:    return {d, qd};
        case Proj::K:    return {d, kd};
        case Proj::V:    return {d, kd};
        case Proj::O:    return {qd, d};
        case Proj::Gate: return {d, _cfg.ff};
        case Proj::Up:   return {d, _cfg.ff};
        case Proj::Down: return {_cfg.ff, d};
    }
    throw std::runtime_error("DenseBlock: unknown projection");
}

const std::vector<float>& DenseBlock::weight(Proj p) const {
    switch (p) {
        case Proj::Q:    return _w.wq;
        case Proj::K:    return _w.wk;
        case Proj::V:    return _w.wv;
        case Proj::O:    return _w.wo;
        case Proj::Gate: return _w.wGate;
        case Proj::Up:   return _w.wUp;
        case Proj::Down: return _w.wDown;
    }
    throw std::runtime_error("DenseBlock: unknown projection");
}

const std::vector<float>* DenseBlock::bias(Proj p) const {
    const std::vector<float>* b = nullptr;
    if (p == Proj::Q) b = &_w.bq;
    if (p == Proj::K) b = &_w.bk;
    if (p == Proj::V) b = &_w.bv;
    return (b != nullptr && !b->empty()) ? b : nullptr;
}

void DenseBlock::linear(Proj p, const float* in, std::size_t T, float* out) const {
    const auto* b = bias(p);
    loraLinear(p, weight(p).data(), b != nullptr ? b->data() : nullptr, in, T, out);
}

void DenseBlock::linearBack(Proj p, const float* dOut, const float* in, std::size_t T, float* dIn,
                            bool accumulate) {
    loraLinearBack(p, weight(p).data(), dOut, in, T, dIn, accumulate);
}

void DenseBlock::run(const float* x, std::size_t T, std::size_t startPos, Acts& a) const {
    const std::size_t d = _cfg.dModel, hd = _cfg.headDim;
    const std::size_t qd = _cfg.heads * hd, kd = _cfg.kvHeads * hd, ff = _cfg.ff;
    const float* rf = _cfg.ropeFactors.empty() ? nullptr : _cfg.ropeFactors.data();

    a.n1.resize(T * d);
    RmsNormGrad::forward(x, T, d, _w.attnNorm.data(), _cfg.eps, a.n1.data());
    a.q.resize(T * qd);
    a.k.resize(T * kd);
    a.v.resize(T * kd);
    linear(Proj::Q, a.n1.data(), T, a.q.data());
    linear(Proj::K, a.n1.data(), T, a.k.data());
    linear(Proj::V, a.n1.data(), T, a.v.data());
    RopeGrad::forward(a.q.data(), T, _cfg.heads, hd, startPos, _cfg.ropeBase, _cfg.ropeLayout, rf);
    RopeGrad::forward(a.k.data(), T, _cfg.kvHeads, hd, startPos, _cfg.ropeBase, _cfg.ropeLayout, rf);

    // Training forward = prefill of the block input: Tq == Tk, no cache.
    const AttentionGrad::Shape s{T, T, _cfg.heads, _cfg.kvHeads, hd, 0, 0, _cfg.attnScale};
    a.ctx.resize(T * qd);
    AttentionGrad::forward(a.q.data(), a.k.data(), a.v.data(), s, a.ctx.data(), a.probs);

    a.h.resize(T * d);
    linear(Proj::O, a.ctx.data(), T, a.h.data());
    for (std::size_t i = 0; i < T * d; ++i) a.h[i] += x[i];

    a.n2.resize(T * d);
    RmsNormGrad::forward(a.h.data(), T, d, _w.ffnNorm.data(), _cfg.eps, a.n2.data());
    a.g.resize(T * ff);
    a.u.resize(T * ff);
    linear(Proj::Gate, a.n2.data(), T, a.g.data());
    linear(Proj::Up, a.n2.data(), T, a.u.data());
    a.act.resize(T * ff);
    SwiGluGrad::forward(a.g.data(), a.u.data(), T * ff, a.act.data());
    a.y.resize(T * d);
    linear(Proj::Down, a.act.data(), T, a.y.data());
    for (std::size_t i = 0; i < T * d; ++i) a.y[i] += a.h[i];
}

void DenseBlock::forward(const float* x, std::size_t T, float* y, std::size_t startPos) const {
    Acts a;
    run(x, T, startPos, a);
    std::copy(a.y.begin(), a.y.end(), y);
}

void DenseBlock::backward(const float* x, std::size_t T, const float* dy, float* dx,
                          std::size_t startPos) {
    Acts a;
    run(x, T, startPos, a);   // recompute (only x was kept)
    const std::size_t d = _cfg.dModel, hd = _cfg.headDim;
    const std::size_t qd = _cfg.heads * hd, kd = _cfg.kvHeads * hd, ff = _cfg.ff;
    const float* rf = _cfg.ropeFactors.empty() ? nullptr : _cfg.ropeFactors.data();

    // FFN half: y = h + Down(swiglu(Gate n2, Up n2)).
    std::vector<float> dAct(T * ff), dG(T * ff), dU(T * ff), dN2(T * d), dH(dy, dy + T * d);
    linearBack(Proj::Down, dy, a.act.data(), T, dAct.data(), false);
    SwiGluGrad::backward(dAct.data(), a.g.data(), a.u.data(), T * ff, dG.data(), dU.data());
    linearBack(Proj::Gate, dG.data(), a.n2.data(), T, dN2.data(), false);
    linearBack(Proj::Up, dU.data(), a.n2.data(), T, dN2.data(), true);
    std::vector<float> dHn(T * d);
    RmsNormGrad::backward(dN2.data(), a.h.data(), T, d, _w.ffnNorm.data(), _cfg.eps, dHn.data(), nullptr);
    for (std::size_t i = 0; i < T * d; ++i) dH[i] += dHn[i];

    // Attention half: h = x + O(ctx).
    std::vector<float> dCtx(T * qd);
    linearBack(Proj::O, dH.data(), a.ctx.data(), T, dCtx.data(), false);
    const AttentionGrad::Shape s{T, T, _cfg.heads, _cfg.kvHeads, hd, 0, 0, _cfg.attnScale};
    std::vector<float> dQ(T * qd), dK(T * kd), dV(T * kd);
    AttentionGrad::backward(dCtx.data(), a.q.data(), a.k.data(), a.v.data(), s, a.probs,
                            dQ.data(), dK.data(), dV.data());
    RopeGrad::backward(dQ.data(), T, _cfg.heads, hd, startPos, _cfg.ropeBase, _cfg.ropeLayout, rf);
    RopeGrad::backward(dK.data(), T, _cfg.kvHeads, hd, startPos, _cfg.ropeBase, _cfg.ropeLayout, rf);
    std::vector<float> dN1(T * d);
    linearBack(Proj::Q, dQ.data(), a.n1.data(), T, dN1.data(), false);
    linearBack(Proj::K, dK.data(), a.n1.data(), T, dN1.data(), true);
    linearBack(Proj::V, dV.data(), a.n1.data(), T, dN1.data(), true);
    RmsNormGrad::backward(dN1.data(), x, T, d, _w.attnNorm.data(), _cfg.eps, dx, nullptr);
    for (std::size_t i = 0; i < T * d; ++i) dx[i] += dH[i];
}

} // namespace mimirmind::hugin::train
