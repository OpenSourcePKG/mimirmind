// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "hugin/train/MoeBlock.hpp"
#include "hugin/train/Parallel.hpp"

#include "hugin/train/AttentionGrad.hpp"
#include "hugin/train/LinearGrad.hpp"
#include "hugin/train/MoeCombineGrad.hpp"
#include "hugin/train/RmsNormGrad.hpp"
#include "hugin/train/SwiGluGrad.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>

namespace mimirmind::hugin::train {

namespace {

void requireSize(const std::vector<float>& v, std::size_t n, const char* what) {
    if (v.size() != n) {
        throw std::runtime_error(std::string("MoeBlock: '") + what + "' has " + std::to_string(v.size()) +
                                 " values, expected " + std::to_string(n));
    }
}

/// Rows of `src` [*, D] listed in `rows` gathered into `dst` [rows.size(), D].
void gather(const float* src, const std::vector<std::size_t>& rows, std::size_t D, float* dst) {
    for (std::size_t i = 0; i < rows.size(); ++i) {
        std::copy(src + rows[i] * D, src + (rows[i] + 1) * D, dst + i * D);
    }
}

} // namespace

MoeBlock::MoeBlock(Config cfg, Weights w, ExpertLoader experts)
    : _cfg(cfg), _w(std::move(w)), _experts(std::move(experts)) {
    const std::size_t d = _cfg.dModel, hd = _cfg.headDim;
    const std::size_t qd = _cfg.heads * hd, kd = _cfg.kvHeads * hd;
    if (d == 0 || _cfg.heads == 0 || _cfg.kvHeads == 0 || hd == 0 || _cfg.heads % _cfg.kvHeads != 0 ||
        _cfg.experts == 0 || _cfg.topK == 0 || _cfg.topK > _cfg.experts || _cfg.expertFf == 0 ||
        _cfg.sharedFf == 0) {
        throw std::runtime_error("MoeBlock: invalid shape config");
    }
    if (_cfg.sliding && _cfg.window == 0) throw std::runtime_error("MoeBlock: a sliding layer needs a window");
    if (!_experts) throw std::runtime_error("MoeBlock: no expert loader");
    requireSize(_w.attnNorm, d, "attn_norm");
    requireSize(_w.wq, qd * d, "attn_q");
    requireSize(_w.wk, kd * d, "attn_k");
    requireSize(_w.wv, kd * d, "attn_v");
    requireSize(_w.wo, d * qd, "attn_output");
    requireSize(_w.qNorm, hd, "attn_q_norm");
    requireSize(_w.kNorm, hd, "attn_k_norm");
    requireSize(_w.postAttnNorm, d, "post_attn_norm");
    requireSize(_w.ffnNorm, d, "post_attention_norm");
    requireSize(_w.postFfnNorm, d, "post_ffn_norm");
    requireSize(_w.router, _cfg.experts * d, "ffn_gate_inp");
    requireSize(_w.routerBias, _cfg.experts, "exp_probs_b");
    requireSize(_w.shGate, _cfg.sharedFf * d, "ffn_gate_shexp");
    requireSize(_w.shUp, _cfg.sharedFf * d, "ffn_up_shexp");
    requireSize(_w.shDown, d * _cfg.sharedFf, "ffn_down_shexp");
}

std::pair<std::size_t, std::size_t> MoeBlock::projShape(Proj p) const {
    const std::size_t d = _cfg.dModel, qd = _cfg.heads * _cfg.headDim, kd = _cfg.kvHeads * _cfg.headDim;
    switch (p) {
        case Proj::Q:    return {d, qd};
        case Proj::K:    return {d, kd};
        case Proj::V:    return {d, kd};
        case Proj::O:    return {qd, d};
        case Proj::Gate: return {d, _cfg.sharedFf};
        case Proj::Up:   return {d, _cfg.sharedFf};
        case Proj::Down: return {_cfg.sharedFf, d};
    }
    throw std::runtime_error("MoeBlock: unknown projection");
}

const std::vector<float>& MoeBlock::weight(Proj p) const {
    switch (p) {
        case Proj::Q:    return _w.wq;
        case Proj::K:    return _w.wk;
        case Proj::V:    return _w.wv;
        case Proj::O:    return _w.wo;
        case Proj::Gate: return _w.shGate;
        case Proj::Up:   return _w.shUp;
        case Proj::Down: return _w.shDown;
    }
    throw std::runtime_error("MoeBlock: unknown projection");
}

std::vector<MoeBlock::Use> MoeBlock::uses(const std::vector<std::int32_t>& sel) const {
    std::map<std::size_t, std::vector<std::size_t>> byExpert;
    for (std::size_t i = 0; i < sel.size(); ++i) byExpert[static_cast<std::size_t>(sel[i])].push_back(i);
    std::vector<Use> out;
    out.reserve(byExpert.size());
    for (auto& [e, pairs] : byExpert) out.push_back(Use{e, std::move(pairs)});
    return out;
}

void MoeBlock::run(const float* x, std::size_t T, std::size_t startPos, Acts& a) const {
    const std::size_t d = _cfg.dModel, hd = _cfg.headDim, H = _cfg.heads, Hkv = _cfg.kvHeads;
    const std::size_t qd = H * hd, kd = Hkv * hd, E = _cfg.experts, K = _cfg.topK, ff = _cfg.expertFf;

    // Attention half (sandwich: pre-norm, per-head QK-norm, post-attn norm).
    a.n1.resize(T * d);
    RmsNormGrad::forward(x, T, d, _w.attnNorm.data(), _cfg.eps, a.n1.data());
    a.q.resize(T * qd);
    a.k.resize(T * kd);
    a.v.resize(T * kd);
    loraLinear(Proj::Q, _w.wq.data(), nullptr, a.n1.data(), T, a.q.data());
    loraLinear(Proj::K, _w.wk.data(), nullptr, a.n1.data(), T, a.k.data());
    loraLinear(Proj::V, _w.wv.data(), nullptr, a.n1.data(), T, a.v.data());
    a.qn.resize(T * qd);
    a.kn.resize(T * kd);
    RmsNormGrad::forward(a.q.data(), T * H, hd, _w.qNorm.data(), _cfg.eps, a.qn.data());
    RmsNormGrad::forward(a.k.data(), T * Hkv, hd, _w.kNorm.data(), _cfg.eps, a.kn.data());
    if (_cfg.sliding) {
        RopeGrad::forward(a.qn.data(), T, H, hd, startPos, _cfg.ropeBase, _cfg.ropeLayout);
        RopeGrad::forward(a.kn.data(), T, Hkv, hd, startPos, _cfg.ropeBase, _cfg.ropeLayout);
    }
    const AttentionGrad::Shape s{T, T, H, Hkv, hd, 0, _cfg.sliding ? _cfg.window : 0, _cfg.attnScale};
    a.ctx.resize(T * qd);
    AttentionGrad::forward(a.qn.data(), a.kn.data(), a.v.data(), s, a.ctx.data(), a.probs);
    a.o.resize(T * d);
    loraLinear(Proj::O, _w.wo.data(), nullptr, a.ctx.data(), T, a.o.data());
    a.h.resize(T * d);
    RmsNormGrad::forward(a.o.data(), T, d, _w.postAttnNorm.data(), _cfg.eps, a.h.data());
    for (std::size_t i = 0; i < T * d; ++i) a.h[i] += x[i];

    // MoE half: router, frozen top-K selection on logits + bias.
    a.m.resize(T * d);
    RmsNormGrad::forward(a.h.data(), T, d, _w.ffnNorm.data(), _cfg.eps, a.m.data());
    a.z.resize(T * E);
    LinearGrad::forward(a.m.data(), T, d, _w.router.data(), E, nullptr, a.z.data());
    a.sel.assign(T * K, 0);
    std::vector<char> chosen(E);
    for (std::size_t t = 0; t < T; ++t) {
        const float* lg = a.z.data() + t * E;
        std::fill(chosen.begin(), chosen.end(), char{0});
        for (std::size_t k = 0; k < K; ++k) {
            std::size_t best      = E;
            float       bestScore = -std::numeric_limits<float>::infinity();
            for (std::size_t e = 0; e < E; ++e) {
                if (chosen[e] != 0) continue;
                const float sc = lg[e] + _w.routerBias[e];
                if (best == E || sc > bestScore) {
                    bestScore = sc;
                    best      = e;
                }
            }
            chosen[best] = 1;
            a.sel[t * K + k] = static_cast<std::int32_t>(best);
        }
    }
    _lastSel = a.sel;

    // Shared expert (always on, LoRA slots Gate / Up / Down).
    const std::size_t sf = _cfg.sharedFf;
    a.sg.resize(T * sf);
    a.su.resize(T * sf);
    a.sAct.resize(T * sf);
    loraLinear(Proj::Gate, _w.shGate.data(), nullptr, a.m.data(), T, a.sg.data());
    loraLinear(Proj::Up, _w.shUp.data(), nullptr, a.m.data(), T, a.su.data());
    SwiGluGrad::forward(a.sg.data(), a.su.data(), T * sf, a.sAct.data());
    a.f.resize(T * d);
    loraLinear(Proj::Down, _w.shDown.data(), nullptr, a.sAct.data(), T, a.f.data());

    // Routed experts, one at a time over the rows that picked them.
    // Experts run in parallel (each one is a few rows, too small to split
    // inside); every expert writes its own result slice, merged afterwards in
    // the fixed expert order, so the sum is the same as a serial loop.
    a.routedOut.assign(T * d, 0.0F);
    const std::vector<Use> used = uses(a.sel);
    std::vector<std::vector<float>> part(used.size());
    Parallel::forRange(used.size(), 1, [&](std::size_t u0, std::size_t u1) {
        std::vector<float> wg(ff * d), wu(ff * d), wd(d * ff);
        for (std::size_t ui = u0; ui < u1; ++ui) {
            const Use& u = used[ui];
            _experts(u.expert, wg.data(), wu.data(), wd.data());
            std::vector<std::size_t> rows;
            for (const auto p : u.pairs) rows.push_back(p / K);
            const std::size_t R = rows.size();
            std::vector<float> mr(R * d), g(R * ff), up(R * ff), act(R * ff), xe(R * d);
            gather(a.m.data(), rows, d, mr.data());
            LinearGrad::forward(mr.data(), R, d, wg.data(), ff, nullptr, g.data());
            LinearGrad::forward(mr.data(), R, d, wu.data(), ff, nullptr, up.data());
            SwiGluGrad::forward(g.data(), up.data(), R * ff, act.data());
            LinearGrad::forward(act.data(), R, ff, wd.data(), d, nullptr, xe.data());
            part[ui].resize(R * d);
            for (std::size_t i = 0; i < R; ++i) {
                const float zi = a.z[rows[i] * E + u.expert];
                MoeCombineGrad::forward(&zi, xe.data() + i * d, 1, d, 1.0F, part[ui].data() + i * d);
            }
        }
    });
    for (std::size_t ui = 0; ui < used.size(); ++ui) {
        const auto& pairs = used[ui].pairs;
        for (std::size_t i = 0; i < pairs.size(); ++i) {
            float*       dst = a.routedOut.data() + (pairs[i] / K) * d;
            const float* src = part[ui].data() + i * d;
            for (std::size_t j = 0; j < d; ++j) dst[j] += src[j];
        }
    }
    for (std::size_t i = 0; i < T * d; ++i) a.f[i] += a.routedOut[i];

    a.y.resize(T * d);
    RmsNormGrad::forward(a.f.data(), T, d, _w.postFfnNorm.data(), _cfg.eps, a.y.data());
    for (std::size_t i = 0; i < T * d; ++i) a.y[i] += a.h[i];
}

void MoeBlock::forward(const float* x, std::size_t T, float* y, std::size_t startPos) const {
    Acts a;
    run(x, T, startPos, a);
    std::copy(a.y.begin(), a.y.end(), y);
}

void MoeBlock::backward(const float* x, std::size_t T, const float* dy, float* dx, std::size_t startPos) {
    Acts a;
    run(x, T, startPos, a);   // recompute (only x was kept)
    const std::size_t d = _cfg.dModel, hd = _cfg.headDim, H = _cfg.heads, Hkv = _cfg.kvHeads;
    const std::size_t qd = H * hd, kd = Hkv * hd, E = _cfg.experts, K = _cfg.topK, ff = _cfg.expertFf;
    const std::size_t sf = _cfg.sharedFf;

    // y = h + RMS(f) w_post_ffn.
    std::vector<float> dH(dy, dy + T * d), dF(T * d);
    RmsNormGrad::backward(dy, a.f.data(), T, d, _w.postFfnNorm.data(), _cfg.eps, dF.data(), nullptr);

    // Shared expert: f += Down(swiglu(Gate m, Up m)).
    std::vector<float> dSAct(T * sf), dSg(T * sf), dSu(T * sf), dM(T * d);
    loraLinearBack(Proj::Down, _w.shDown.data(), dF.data(), a.sAct.data(), T, dSAct.data(), false);
    SwiGluGrad::backward(dSAct.data(), a.sg.data(), a.su.data(), T * sf, dSg.data(), dSu.data());
    loraLinearBack(Proj::Gate, _w.shGate.data(), dSg.data(), a.m.data(), T, dM.data(), false);
    loraLinearBack(Proj::Up, _w.shUp.data(), dSu.data(), a.m.data(), T, dM.data(), true);

    // Routed experts: f += sigmoid(z_e) E_e(m) per selected pair; gradient into
    // the selected router logits and through each expert into m.
    // Parallel over experts like run(); a (token, expert) pair occurs once, so
    // dZ entries are disjoint, and dM is merged afterwards in expert order.
    std::vector<float> dZ(T * E, 0.0F);
    const std::vector<Use> used = uses(a.sel);
    std::vector<std::vector<float>> part(used.size());
    Parallel::forRange(used.size(), 1, [&](std::size_t u0, std::size_t u1) {
        std::vector<float> wg(ff * d), wu(ff * d), wd(d * ff);
        for (std::size_t ui = u0; ui < u1; ++ui) {
            const Use& u = used[ui];
            _experts(u.expert, wg.data(), wu.data(), wd.data());
            std::vector<std::size_t> rows;
            for (const auto p : u.pairs) rows.push_back(p / K);
            const std::size_t R = rows.size();
            std::vector<float> mr(R * d), g(R * ff), up(R * ff), act(R * ff), xe(R * d);
            gather(a.m.data(), rows, d, mr.data());
            LinearGrad::forward(mr.data(), R, d, wg.data(), ff, nullptr, g.data());
            LinearGrad::forward(mr.data(), R, d, wu.data(), ff, nullptr, up.data());
            SwiGluGrad::forward(g.data(), up.data(), R * ff, act.data());
            LinearGrad::forward(act.data(), R, ff, wd.data(), d, nullptr, xe.data());
            std::vector<float> dXe(R * d), dAct(R * ff), dG(R * ff), dU(R * ff), dMr(R * d), dMr2(R * d);
            for (std::size_t i = 0; i < R; ++i) {
                const float zi = a.z[rows[i] * E + u.expert];
                float dz = 0.0F;
                MoeCombineGrad::backward(dF.data() + rows[i] * d, &zi, xe.data() + i * d, 1, d, 1.0F, &dz,
                                         dXe.data() + i * d);
                dZ[rows[i] * E + u.expert] += dz;
            }
            LinearGrad::backward(dXe.data(), act.data(), R, ff, wd.data(), d, nullptr, dAct.data(), nullptr, nullptr);
            SwiGluGrad::backward(dAct.data(), g.data(), up.data(), R * ff, dG.data(), dU.data());
            LinearGrad::backward(dG.data(), mr.data(), R, d, wg.data(), ff, nullptr, dMr.data(), nullptr, nullptr);
            LinearGrad::backward(dU.data(), mr.data(), R, d, wu.data(), ff, nullptr, dMr2.data(), nullptr, nullptr);
            part[ui].resize(R * d);
            for (std::size_t i = 0; i < R * d; ++i) part[ui][i] = dMr[i] + dMr2[i];
        }
    });
    for (std::size_t ui = 0; ui < used.size(); ++ui) {
        const auto& pairs = used[ui].pairs;
        for (std::size_t i = 0; i < pairs.size(); ++i) {
            float*       dst = dM.data() + (pairs[i] / K) * d;
            const float* src = part[ui].data() + i * d;
            for (std::size_t j = 0; j < d; ++j) dst[j] += src[j];
        }
    }
    {
        // Router logits z = m Wr^T (frozen weights; only the input gradient).
        std::vector<float> dMz(T * d);
        LinearGrad::backward(dZ.data(), a.m.data(), T, d, _w.router.data(), E, nullptr, dMz.data(), nullptr,
                             nullptr);
        for (std::size_t i = 0; i < T * d; ++i) dM[i] += dMz[i];
    }
    std::vector<float> dHn(T * d);
    RmsNormGrad::backward(dM.data(), a.h.data(), T, d, _w.ffnNorm.data(), _cfg.eps, dHn.data(), nullptr);
    for (std::size_t i = 0; i < T * d; ++i) dH[i] += dHn[i];

    // h = x + RMS(o) w_post_attn,  o = ctx Wo^T.
    std::vector<float> dO(T * d), dCtx(T * qd);
    RmsNormGrad::backward(dH.data(), a.o.data(), T, d, _w.postAttnNorm.data(), _cfg.eps, dO.data(), nullptr);
    loraLinearBack(Proj::O, _w.wo.data(), dO.data(), a.ctx.data(), T, dCtx.data(), false);
    const AttentionGrad::Shape s{T, T, H, Hkv, hd, 0, _cfg.sliding ? _cfg.window : 0, _cfg.attnScale};
    std::vector<float> dQn(T * qd), dKn(T * kd), dV(T * kd);
    AttentionGrad::backward(dCtx.data(), a.qn.data(), a.kn.data(), a.v.data(), s, a.probs, dQn.data(),
                            dKn.data(), dV.data());
    if (_cfg.sliding) {
        RopeGrad::backward(dQn.data(), T, H, hd, startPos, _cfg.ropeBase, _cfg.ropeLayout);
        RopeGrad::backward(dKn.data(), T, Hkv, hd, startPos, _cfg.ropeBase, _cfg.ropeLayout);
    }
    std::vector<float> dQ(T * qd), dK(T * kd), dN1(T * d);
    RmsNormGrad::backward(dQn.data(), a.q.data(), T * H, hd, _w.qNorm.data(), _cfg.eps, dQ.data(), nullptr);
    RmsNormGrad::backward(dKn.data(), a.k.data(), T * Hkv, hd, _w.kNorm.data(), _cfg.eps, dK.data(), nullptr);
    loraLinearBack(Proj::Q, _w.wq.data(), dQ.data(), a.n1.data(), T, dN1.data(), false);
    loraLinearBack(Proj::K, _w.wk.data(), dK.data(), a.n1.data(), T, dN1.data(), true);
    loraLinearBack(Proj::V, _w.wv.data(), dV.data(), a.n1.data(), T, dN1.data(), true);
    RmsNormGrad::backward(dN1.data(), x, T, d, _w.attnNorm.data(), _cfg.eps, dx, nullptr);
    for (std::size_t i = 0; i < T * d; ++i) dx[i] += dH[i];
}

} // namespace mimirmind::hugin::train
