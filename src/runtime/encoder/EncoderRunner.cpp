// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/encoder/EncoderRunner.hpp"

#include "compute/ComputeBuffer.hpp"
#include "compute/ComputeMatmul.hpp"
#include "compute/ComputeOps.hpp"
#include "compute/Embedding.hpp"
#include "core/gguf/GgufTypes.hpp"
#include "runtime/encoder/EncoderModel.hpp"

#include <cmath>
#include <stdexcept>

namespace mimirmind::runtime::encoder {

EncoderRunner::EncoderRunner(const EncoderModel& model,
                             compute::ComputeOps& ops,
                             compute::ComputeMatmul& matmul)
    : _m{model}, _ops{ops}, _mm{matmul} {}

std::vector<float>
EncoderRunner::forwardLogits(std::span<const std::int32_t> inputIds) {
    const EncoderConfig& c = _m.config();
    const std::size_t T   = inputIds.size();
    if (T == 0) {
        throw std::runtime_error("EncoderRunner: empty input");
    }
    const std::size_t H   = c.hidden;
    const std::size_t ffn = c.ffn;
    const std::size_t nL  = c.numLabels;
    const float eps       = c.lnEps;
    const float scale     = 1.0F / std::sqrt(static_cast<float>(c.headDim));
    const auto  F32       = core::gguf::GgmlType::F32;

    auto alloc = [&](std::size_t n) { return _ops.allocate(n * sizeof(float)); };
    auto fp = [](compute::ComputeBuffer& b) {
        return static_cast<float*>(b.get());
    };

    compute::ComputeBuffer xb   = alloc(T * H);
    compute::ComputeBuffer hb   = alloc(T * H);
    compute::ComputeBuffer qb   = alloc(T * H);
    compute::ComputeBuffer kb   = alloc(T * H);
    compute::ComputeBuffer vb   = alloc(T * H);
    compute::ComputeBuffer attnb = alloc(T * H);
    compute::ComputeBuffer aob  = alloc(T * H);
    compute::ComputeBuffer ln1b = alloc(T * H);
    compute::ComputeBuffer interb = alloc(T * ffn);
    compute::ComputeBuffer ffnb = alloc(T * H);
    compute::ComputeBuffer scr  = alloc(T * ffn);
    compute::ComputeBuffer clsb = alloc(H);
    compute::ComputeBuffer outb = alloc(nL);

    float* x    = fp(xb);
    float* h    = fp(hb);
    float* q    = fp(qb);
    float* k    = fp(kb);
    float* v    = fp(vb);
    float* attn = fp(attnb);
    float* ao   = fp(aob);
    float* ln1  = fp(ln1b);
    float* inter = fp(interb);
    float* ffnO = fp(ffnb);
    float* scratch = fp(scr);
    float* cls  = fp(clsb);
    float* out  = fp(outb);

    // ---- embeddings block: word (gather) + pos + type, then LayerNorm ----
    compute::embeddingLookup(F32, _m.wordEmb(), H, c.vocab, inputIds, x);
    _ops.encoderEmbedAddAsync(x, _m.posTable(), _m.typeVec(), T, H, c.posOffset);
    _ops.layerNormAsync(x, T, H, _m.embLnW(), _m.embLnB(), eps, h);

    // ---- encoder layers (post-LN) ----
    for (std::size_t i = 0; i < c.numLayers; ++i) {
        const EncoderLayerWeights& L = _m.layer(i);

        // Q/K/V = h * W^T + b
        _mm.matmulAsync(F32, L.qW, H, H, h, T, q, scratch);
        _ops.addBiasAsync(q, T, H, L.qB);
        _mm.matmulAsync(F32, L.kW, H, H, h, T, k, scratch);
        _ops.addBiasAsync(k, T, H, L.kB);
        _mm.matmulAsync(F32, L.vW, H, H, h, T, v, scratch);
        _ops.addBiasAsync(v, T, H, L.vB);

        // bidirectional self-attention
        _ops.attentionEncoderAsync(q, k, v, T, c.heads, c.heads, c.headDim,
                                   scale, attn);

        // attention.output.dense + residual(h) + LayerNorm
        _mm.matmulAsync(F32, L.aoW, H, H, attn, T, ao, scratch);
        _ops.addBiasAsync(ao, T, H, L.aoB);
        _ops.addResidualAsync(ao, h, T * H);
        _ops.layerNormAsync(ao, T, H, L.aoLnW, L.aoLnB, eps, ln1);

        // FFN: intermediate (erf-GELU) + output.dense + residual(ln1) + LN
        _mm.matmulAsync(F32, L.fiW, ffn, H, ln1, T, inter, scratch);
        _ops.addBiasAsync(inter, T, ffn, L.fiB);
        _ops.geluErfAsync(inter, T * ffn);
        _mm.matmulAsync(F32, L.foW, H, ffn, inter, T, ffnO, scratch);
        _ops.addBiasAsync(ffnO, T, H, L.foB);
        _ops.addResidualAsync(ffnO, ln1, T * H);
        // next layer's input goes back into h
        _ops.layerNormAsync(ffnO, T, H, L.outLnW, L.outLnB, eps, h);
    }

    // ---- classifier head on the <s>/CLS token (row 0 of h) ----
    _mm.matmulAsync(F32, _m.clsDenseW(), H, H, h, 1, cls, scratch);
    _ops.addBiasAsync(cls, 1, H, _m.clsDenseB());
    _ops.tanhInPlaceAsync(cls, H);
    _mm.matmulAsync(F32, _m.clsOutW(), nL, H, cls, 1, out, scratch);
    _ops.addBiasAsync(out, 1, nL, _m.clsOutB());

    _ops.flush();
    std::vector<float> logits(nL);
    _ops.readbackToHost(logits.data(), out, nL * sizeof(float));
    return logits;
}

float EncoderRunner::score(std::span<const std::int32_t> inputIds) {
    const std::vector<float> logits = forwardLogits(inputIds);
    return logits.empty() ? 0.0F : logits.front();
}

} // namespace mimirmind::runtime::encoder
