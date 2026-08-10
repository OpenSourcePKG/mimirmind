// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/encoder/EncoderRunner.hpp"

#include "compute/ComputeBuffer.hpp"
#include "compute/ComputeMatmul.hpp"
#include "compute/ComputeOps.hpp"
#include "compute/Embedding.hpp"
#include "core/gguf/GgufTypes.hpp"
#include "runtime/encoder/EncoderModel.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
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
    const auto  WT        = _m.matmulType();

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
        _mm.matmulAsync(WT, L.qW, H, H, h, T, q, scratch);
        _ops.addBiasAsync(q, T, H, L.qB);
        _mm.matmulAsync(WT, L.kW, H, H, h, T, k, scratch);
        _ops.addBiasAsync(k, T, H, L.kB);
        _mm.matmulAsync(WT, L.vW, H, H, h, T, v, scratch);
        _ops.addBiasAsync(v, T, H, L.vB);

        // bidirectional self-attention
        _ops.attentionEncoderAsync(q, k, v, T, c.heads, c.heads, c.headDim,
                                   scale, attn);

        // attention.output.dense + residual(h) + LayerNorm
        _mm.matmulAsync(WT, L.aoW, H, H, attn, T, ao, scratch);
        _ops.addBiasAsync(ao, T, H, L.aoB);
        _ops.addResidualAsync(ao, h, T * H);
        _ops.layerNormAsync(ao, T, H, L.aoLnW, L.aoLnB, eps, ln1);

        // FFN: intermediate (erf-GELU) + output.dense + residual(ln1) + LN
        _mm.matmulAsync(WT, L.fiW, ffn, H, ln1, T, inter, scratch);
        _ops.addBiasAsync(inter, T, ffn, L.fiB);
        _ops.geluErfAsync(inter, T * ffn);
        _mm.matmulAsync(WT, L.foW, H, ffn, inter, T, ffnO, scratch);
        _ops.addBiasAsync(ffnO, T, H, L.foB);
        _ops.addResidualAsync(ffnO, ln1, T * H);
        // next layer's input goes back into h
        _ops.layerNormAsync(ffnO, T, H, L.outLnW, L.outLnB, eps, h);
    }

    // ---- classifier head on the <s>/CLS token (row 0 of h) ----
    _mm.matmulAsync(WT, _m.clsDenseW(), H, H, h, 1, cls, scratch);
    _ops.addBiasAsync(cls, 1, H, _m.clsDenseB());
    _ops.tanhInPlaceAsync(cls, H);
    _mm.matmulAsync(WT, _m.clsOutW(), nL, H, cls, 1, out, scratch);
    _ops.addBiasAsync(out, 1, nL, _m.clsOutB());

    _ops.flush();
    std::vector<float> logits(nL);
    _ops.readbackToHost(logits.data(), out, nL * sizeof(float));
    return logits;
}

std::vector<std::vector<float>>
EncoderRunner::forwardLogitsBatch(
    std::span<const std::vector<std::int32_t>> sequences) {
    const EncoderConfig& c = _m.config();
    const std::size_t B = sequences.size();
    if (B == 0) {
        return {};
    }

    std::size_t Tmax = 0;
    for (const auto& s : sequences) {
        Tmax = std::max(Tmax, s.size());
    }
    if (Tmax == 0) {
        return std::vector<std::vector<float>>(B, std::vector<float>(c.numLabels, 0.0F));
    }

    const std::size_t H   = c.hidden;
    const std::size_t ffn = c.ffn;
    const std::size_t nL  = c.numLabels;
    const std::size_t R   = B * Tmax;                 // padded row count
    const float eps       = c.lnEps;
    const float scale     = 1.0F / std::sqrt(static_cast<float>(c.headDim));
    const auto  F32       = core::gguf::GgmlType::F32;
    const auto  WT        = _m.matmulType();

    // Padded token ids (pad rows filled with pad_token_id) + per-seq lengths.
    std::vector<std::int32_t> ids(R, c.padTokenId);
    std::vector<std::int32_t> seqLens(B, 0);
    for (std::size_t b = 0; b < B; ++b) {
        const auto& s = sequences[b];
        seqLens[b] = static_cast<std::int32_t>(s.size());
        std::copy(s.begin(), s.end(), ids.begin() + static_cast<std::ptrdiff_t>(b * Tmax));
    }

    auto alloc = [&](std::size_t n) { return _ops.allocate(n * sizeof(float)); };
    auto fp = [](compute::ComputeBuffer& b) { return static_cast<float*>(b.get()); };

    compute::ComputeBuffer xb   = alloc(R * H);
    compute::ComputeBuffer hb   = alloc(R * H);
    compute::ComputeBuffer qb   = alloc(R * H);
    compute::ComputeBuffer kb   = alloc(R * H);
    compute::ComputeBuffer vb   = alloc(R * H);
    compute::ComputeBuffer attnb = alloc(R * H);
    compute::ComputeBuffer aob  = alloc(R * H);
    compute::ComputeBuffer ln1b = alloc(R * H);
    compute::ComputeBuffer interb = alloc(R * ffn);
    compute::ComputeBuffer ffnb = alloc(R * H);
    compute::ComputeBuffer scr  = alloc(R * ffn);
    compute::ComputeBuffer clsInb = alloc(B * H);
    compute::ComputeBuffer clsb = alloc(B * H);
    compute::ComputeBuffer outb = alloc(B * nL);
    compute::ComputeBuffer lensb = _ops.allocate(B * sizeof(std::int32_t));
    _ops.uploadHostBytes(lensb.get(), seqLens.data(), B * sizeof(std::int32_t));

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
    float* clsIn = fp(clsInb);
    float* cls  = fp(clsb);
    float* out  = fp(outb);
    const auto* lens = static_cast<const std::int32_t*>(lensb.get());

    // Embeddings: word gather over all padded rows, then per-sequence
    // pos+type add on the real rows (padding rows keep the raw pad embedding
    // — they never affect a real row: masked out of attention, ignored at
    // the head). LayerNorm is per-row over the whole batch.
    compute::embeddingLookup(F32, _m.wordEmb(), H, c.vocab,
                             std::span<const std::int32_t>{ids}, x);
    for (std::size_t b = 0; b < B; ++b) {
        const std::size_t len = static_cast<std::size_t>(seqLens[b]);
        if (len > 0) {
            _ops.encoderEmbedAddAsync(x + b * Tmax * H, _m.posTable(),
                                      _m.typeVec(), len, H, c.posOffset);
        }
    }
    _ops.layerNormAsync(x, R, H, _m.embLnW(), _m.embLnB(), eps, h);

    for (std::size_t i = 0; i < c.numLayers; ++i) {
        const EncoderLayerWeights& L = _m.layer(i);

        _mm.matmulAsync(WT, L.qW, H, H, h, R, q, scratch);
        _ops.addBiasAsync(q, R, H, L.qB);
        _mm.matmulAsync(WT, L.kW, H, H, h, R, k, scratch);
        _ops.addBiasAsync(k, R, H, L.kB);
        _mm.matmulAsync(WT, L.vW, H, H, h, R, v, scratch);
        _ops.addBiasAsync(v, R, H, L.vB);

        _ops.attentionEncoderBatchedAsync(q, k, v, attn, lens, B, Tmax,
                                          c.heads, c.heads, c.headDim, scale);

        _mm.matmulAsync(WT, L.aoW, H, H, attn, R, ao, scratch);
        _ops.addBiasAsync(ao, R, H, L.aoB);
        _ops.addResidualAsync(ao, h, R * H);
        _ops.layerNormAsync(ao, R, H, L.aoLnW, L.aoLnB, eps, ln1);

        _mm.matmulAsync(WT, L.fiW, ffn, H, ln1, R, inter, scratch);
        _ops.addBiasAsync(inter, R, ffn, L.fiB);
        _ops.geluErfAsync(inter, R * ffn);
        _mm.matmulAsync(WT, L.foW, H, ffn, inter, R, ffnO, scratch);
        _ops.addBiasAsync(ffnO, R, H, L.foB);
        _ops.addResidualAsync(ffnO, ln1, R * H);
        _ops.layerNormAsync(ffnO, R, H, L.outLnW, L.outLnB, eps, h);
    }

    // Classifier head on the <s>/CLS token (row 0) of each sequence: gather
    // those B rows contiguous, then dense -> tanh -> out_proj (batched, M=B).
    for (std::size_t b = 0; b < B; ++b) {
        _ops.appendMemoryCopy(clsIn + b * H, h + b * Tmax * H, H * sizeof(float));
    }
    _mm.matmulAsync(WT, _m.clsDenseW(), H, H, clsIn, B, cls, scratch);
    _ops.addBiasAsync(cls, B, H, _m.clsDenseB());
    _ops.tanhInPlaceAsync(cls, B * H);
    _mm.matmulAsync(WT, _m.clsOutW(), nL, H, cls, B, out, scratch);
    _ops.addBiasAsync(out, B, nL, _m.clsOutB());

    _ops.flush();
    std::vector<float> flat(B * nL);
    _ops.readbackToHost(flat.data(), out, B * nL * sizeof(float));

    std::vector<std::vector<float>> logits(B);
    for (std::size_t b = 0; b < B; ++b) {
        logits[b].assign(flat.begin() + static_cast<std::ptrdiff_t>(b * nL),
                         flat.begin() + static_cast<std::ptrdiff_t>((b + 1) * nL));
    }
    return logits;
}

float EncoderRunner::score(std::span<const std::int32_t> inputIds) {
    const std::vector<float> logits = forwardLogits(inputIds);
    return logits.empty() ? 0.0F : logits.front();
}

} // namespace mimirmind::runtime::encoder
