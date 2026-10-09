// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/encoder/laya/LayaDecisionRunner.hpp"

#include "compute/ComputeBuffer.hpp"
#include "compute/ComputeMatmul.hpp"
#include "compute/ComputeOps.hpp"
#include "core/gguf/GgufTypes.hpp"
#include "runtime/encoder/laya/LayaDecisionHead.hpp"
#include "runtime/encoder/laya/LayaEncoderModel.hpp"
#include "runtime/encoder/laya/LayaEncoderRunner.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <vector>

namespace mimirmind::runtime::encoder::laya {

LayaDecisionRunner::LayaDecisionRunner(const LayaEncoderModel& model,
                                       LayaEncoderRunner& encoder,
                                       const LayaDecisionHead& head,
                                       compute::ComputeOps& ops,
                                       compute::ComputeMatmul& matmul)
    : _m{model}, _enc{encoder}, _head{head}, _ops{ops}, _mm{matmul} {}

LayaDecisionRunner::Result
LayaDecisionRunner::decide(std::span<const std::int32_t> inputIds,
                           std::span<const std::int32_t> markerPos,
                           std::int32_t qtype) {
    const LayaDecisionHeadConfig& hc = _head.config();
    const std::size_t d       = hc.hidden;
    const std::size_t heads   = hc.heads;
    const std::size_t headDim = hc.headDim;
    const std::size_t ffn     = hc.ffn;             // 4*d
    const std::size_t k       = markerPos.size();   // option count
    const float lnEps = 1e-5F;                      // nn.LayerNorm default
    const float scale = 1.0F / std::sqrt(static_cast<float>(headDim));
    const auto  WT    = _head.matmulType();
    if (k == 0) {
        throw std::runtime_error("LayaDecisionRunner::decide: no option markers");
    }
    if (qtype < 0 || qtype > 2) {
        throw std::runtime_error("LayaDecisionRunner::decide: qtype out of [0,2]");
    }

    // Backbone -> final hidden [T, d] on device (already flushed).
    std::size_t T = 0;
    compute::ComputeBuffer hb = _enc.lastHiddenDevice(inputIds, T);
    float* h = static_cast<float*>(hb.get());

    compute::ScopedExactF32 exactF32{_mm};

    auto alloc = [&](std::size_t n) { return _ops.allocate(n * sizeof(float)); };
    auto fp = [](compute::ComputeBuffer& b) { return static_cast<float*>(b.get()); };

    const std::size_t scratchElems = T * std::max<std::size_t>(3 * d, ffn);
    compute::ComputeBuffer x1b  = alloc(T * d);
    compute::ComputeBuffer qkvb = alloc(T * 3 * d);
    compute::ComputeBuffer qb   = alloc(T * d);
    compute::ComputeBuffer kb   = alloc(T * d);
    compute::ComputeBuffer vb   = alloc(T * d);
    compute::ComputeBuffer attnb = alloc(T * d);
    compute::ComputeBuffer aob  = alloc(T * d);
    compute::ComputeBuffer x2b  = alloc(T * d);
    compute::ComputeBuffer f1b  = alloc(T * ffn);
    compute::ComputeBuffer f2b  = alloc(T * d);
    compute::ComputeBuffer scrb = alloc(scratchElems);
    float* x1 = fp(x1b); float* qkv = fp(qkvb);
    float* q = fp(qb); float* kk = fp(kb); float* vv = fp(vb);
    float* attn = fp(attnb); float* ao = fp(aob); float* x2 = fp(x2b);
    float* f1 = fp(f1b); float* f2 = fp(f2b); float* scratch = fp(scrb);

    // type_emb add: h[t,:] += type_emb[qtype] (broadcast over tokens == addBias).
    _ops.addBiasAsync(h, T, d, _head.typeEmb() + static_cast<std::size_t>(qtype) * d);

    // 2 standard pre-norm TransformerEncoderLayers (norm_first, relu FFN).
    for (std::size_t i = 0; i < hc.nHeadLayers; ++i) {
        const LayaHeadLayerWeights& L = _head.layer(i);
        // self-attention block
        _ops.layerNormAsync(h, T, d, L.norm1W, L.norm1B, lnEps, x1);
        _mm.matmulAsync(WT, L.inProjW, 3 * d, d, x1, T, qkv, scratch);
        _ops.addBiasAsync(qkv, T, 3 * d, L.inProjB);
        for (std::size_t t = 0; t < T; ++t) {
            _ops.appendMemoryCopy(q  + t * d, qkv + t * 3 * d,         d * sizeof(float));
            _ops.appendMemoryCopy(kk + t * d, qkv + t * 3 * d + d,     d * sizeof(float));
            _ops.appendMemoryCopy(vv + t * d, qkv + t * 3 * d + 2 * d, d * sizeof(float));
        }
        _ops.attentionEncoderAsync(q, kk, vv, T, heads, heads, headDim, scale, attn);
        _mm.matmulAsync(WT, L.outProjW, d, d, attn, T, ao, scratch);
        _ops.addBiasAsync(ao, T, d, L.outProjB);
        _ops.addResidualAsync(h, ao, T * d);
        // feed-forward block
        _ops.layerNormAsync(h, T, d, L.norm2W, L.norm2B, lnEps, x2);
        _mm.matmulAsync(WT, L.lin1W, ffn, d, x2, T, f1, scratch);
        _ops.addBiasAsync(f1, T, ffn, L.lin1B);
        _ops.reluAsync(f1, T * ffn);
        _mm.matmulAsync(WT, L.lin2W, d, ffn, f1, T, f2, scratch);
        _ops.addBiasAsync(f2, T, d, L.lin2B);
        _ops.addResidualAsync(h, f2, T * d);
    }

    // Gather the k marker rows -> m[k, d].
    compute::ComputeBuffer mb  = alloc(k * d);
    compute::ComputeBuffer sctb = alloc(k * d);
    compute::ComputeBuffer s1b = alloc(k * d);
    compute::ComputeBuffer logb = alloc(k);
    float* m = fp(mb); float* sct = fp(sctb); float* s1 = fp(s1b); float* logd = fp(logb);
    for (std::size_t o = 0; o < k; ++o) {
        const std::int32_t mp = markerPos[o];
        if (mp < 0 || static_cast<std::size_t>(mp) >= T) {
            throw std::runtime_error("LayaDecisionRunner::decide: marker position out of range");
        }
        _ops.appendMemoryCopy(m + o * d, h + static_cast<std::size_t>(mp) * d, d * sizeof(float));
    }
    // scorer: LayerNorm(d) -> Linear(d,d) -> GELU(erf) -> Linear(d,1)
    _ops.layerNormAsync(m, k, d, _head.scorerLnW(), _head.scorerLnB(), lnEps, sct);
    _mm.matmulAsync(WT, _head.scorerW1(), d, d, sct, k, s1, scratch);
    _ops.addBiasAsync(s1, k, d, _head.scorerB1());
    _ops.geluErfAsync(s1, k * d);
    _mm.matmulAsync(WT, _head.scorerW3(), 1, d, s1, k, logd, scratch);
    _ops.addBiasAsync(logd, k, 1, _head.scorerB3());

    _ops.flush();
    Result res;
    res.logits.resize(k);
    _ops.readbackToHost(res.logits.data(), logd, k * sizeof(float));

    // act features (host): softmax over the k option logits, entropy, top-2.
    std::vector<float> p(k);
    {
        const float mx = *std::max_element(res.logits.begin(), res.logits.end());
        double sum = 0.0;
        for (std::size_t o = 0; o < k; ++o) { p[o] = std::exp(res.logits[o] - mx); sum += p[o]; }
        for (std::size_t o = 0; o < k; ++o) { p[o] = static_cast<float>(p[o] / sum); }
    }
    const float kClamp = static_cast<float>(std::max<std::size_t>(k, 2));
    double ent = 0.0;
    for (std::size_t o = 0; o < k; ++o) {
        const float pv = std::max(p[o], 1e-9F);
        ent -= static_cast<double>(p[o]) * std::log(static_cast<double>(pv));
    }
    const float entN = static_cast<float>(ent / std::log(static_cast<double>(kClamp)));
    std::vector<float> ps(p);
    std::sort(ps.begin(), ps.end(), std::greater<float>());
    const float top1 = ps[0];
    const float top2 = (k >= 2) ? ps[1] : 0.0F;
    const std::array<float, 4> feats{top1, top1 - top2, entN, kClamp / 255.0F};

    // act_head: Linear(d+4,256) -> GELU(erf) -> Linear(256, nAct) on [pooled ⊕ feats].
    const std::size_t din = d + 4;
    compute::ComputeBuffer ainb = alloc(din);
    compute::ComputeBuffer a0b  = alloc(256);
    compute::ComputeBuffer a2b  = alloc(hc.nAct);
    float* ain = fp(ainb); float* a0 = fp(a0b); float* a2 = fp(a2b);
    _ops.appendMemoryCopy(ain, h, d * sizeof(float));                 // pooled = h[0]
    _ops.uploadHostBytes(ain + d, feats.data(), feats.size() * sizeof(float));
    _mm.matmulAsync(WT, _head.actW0(), 256, din, ain, 1, a0, scratch);
    _ops.addBiasAsync(a0, 1, 256, _head.actB0());
    _ops.geluErfAsync(a0, 256);
    _mm.matmulAsync(WT, _head.actW2(), hc.nAct, 256, a0, 1, a2, scratch);
    _ops.addBiasAsync(a2, 1, hc.nAct, _head.actB2());
    _ops.flush();
    res.actLogits.resize(hc.nAct);
    _ops.readbackToHost(res.actLogits.data(), a2, hc.nAct * sizeof(float));

    // softmax(actLogits)[0] = act_probability.
    {
        const float mx = *std::max_element(res.actLogits.begin(), res.actLogits.end());
        double sum = 0.0;
        std::vector<double> e(hc.nAct);
        for (std::size_t o = 0; o < hc.nAct; ++o) { e[o] = std::exp(res.actLogits[o] - mx); sum += e[o]; }
        res.actProbability = static_cast<float>(e[0] / sum);
    }
    return res;
}

} // namespace mimirmind::runtime::encoder::laya
