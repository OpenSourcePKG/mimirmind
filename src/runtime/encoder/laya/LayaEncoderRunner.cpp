// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/encoder/laya/LayaEncoderRunner.hpp"

#include "compute/ComputeMatmul.hpp"
#include "compute/ComputeOps.hpp"
#include "compute/Embedding.hpp"
#include "core/gguf/GgufTypes.hpp"
#include "runtime/encoder/laya/LayaEncoderModel.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace mimirmind::runtime::encoder::laya {

LayaEncoderRunner::LayaEncoderRunner(const LayaEncoderModel& model,
                                     compute::ComputeOps& ops,
                                     compute::ComputeMatmul& matmul)
    : _m{model}, _ops{ops}, _mm{matmul} {}

compute::ComputeBuffer
LayaEncoderRunner::encode(std::span<const std::int32_t> inputIds,
                          std::size_t& Tout) {
    const LayaEncoderConfig& c = _m.config();
    const std::size_t T = inputIds.size();
    if (T == 0) {
        throw std::runtime_error("LayaEncoderRunner::encode: empty input");
    }
    Tout = T;
    const std::size_t H       = c.hidden;
    const std::size_t ffn     = c.ffn;
    const std::size_t heads   = c.heads;
    const std::size_t headDim = c.headDim;
    const float eps   = c.normEps;
    const float scale = 1.0F / std::sqrt(static_cast<float>(headDim));
    const auto  F32   = core::gguf::GgmlType::F32;
    const auto  WT    = _m.matmulType();

    // ModernBERT linear projections are precision-sensitive; keep them exact
    // even if the LLM prefill opted the F32->TC downcast in (see EncoderRunner).
    compute::ScopedExactF32 exactF32{_mm};

    auto alloc = [&](std::size_t n) { return _ops.allocate(n * sizeof(float)); };
    auto fp = [](compute::ComputeBuffer& b) { return static_cast<float*>(b.get()); };

    // LayerNorm has no bias in ModernBERT; feed a zeroed beta buffer so the
    // bias'd layerNormAsync computes pure (x-mean)/std * gamma.
    compute::ComputeBuffer zbiasB = alloc(H);
    {
        const std::vector<float> zeros(H, 0.0F);
        _ops.uploadHostBytes(zbiasB.get(), zeros.data(), H * sizeof(float));
    }
    const float* zbias = fp(zbiasB);

    const std::size_t scratchElems = std::max(static_cast<std::size_t>(3) * H,
                                              static_cast<std::size_t>(2) * ffn);
    compute::ComputeBuffer xb      = alloc(T * H);
    compute::ComputeBuffer hb      = alloc(T * H);
    compute::ComputeBuffer normedb = alloc(T * H);
    compute::ComputeBuffer qkvb    = alloc(T * 3 * H);
    compute::ComputeBuffer qb      = alloc(T * H);
    compute::ComputeBuffer kb      = alloc(T * H);
    compute::ComputeBuffer vb      = alloc(T * H);
    compute::ComputeBuffer attnb   = alloc(T * H);
    compute::ComputeBuffer aob     = alloc(T * H);
    compute::ComputeBuffer wib     = alloc(T * 2 * ffn);
    compute::ComputeBuffer ginb    = alloc(T * ffn);
    compute::ComputeBuffer ggateb  = alloc(T * ffn);
    compute::ComputeBuffer mob     = alloc(T * H);
    compute::ComputeBuffer scrb    = alloc(T * scratchElems);
    compute::ComputeBuffer resb    = alloc(T * H);

    float* x       = fp(xb);
    float* h       = fp(hb);
    float* normed  = fp(normedb);
    float* qkv     = fp(qkvb);
    float* q       = fp(qb);
    float* k       = fp(kb);
    float* vv      = fp(vb);
    float* attn    = fp(attnb);
    float* ao      = fp(aob);
    float* wi      = fp(wib);
    float* gin     = fp(ginb);
    float* ggate   = fp(ggateb);
    float* mo      = fp(mob);
    float* scratch = fp(scrb);
    float* res     = fp(resb);

    // Embeddings: token lookup (no positional/type add — RoPE handles position)
    // then the embedding LayerNorm.
    compute::embeddingLookup(F32, _m.tokEmb(), H, c.vocab, inputIds, x);
    _ops.layerNormAsync(x, T, H, _m.embNormW(), zbias, eps, h);

    for (std::size_t i = 0; i < c.numLayers; ++i) {
        const LayaLayerWeights& L = _m.layer(i);
        const float theta = c.ropeTheta(i);

        // ---- attention block (pre-LN; layer 0 attn_norm is Identity) ----
        const float* attnIn = h;
        if (L.attnNormW != nullptr) {
            _ops.layerNormAsync(h, T, H, L.attnNormW, zbias, eps, normed);
            attnIn = normed;
        }
        // fused Q|K|V = Wqkv @ attnIn, then split the per-row [q(H)|k(H)|v(H)].
        _mm.matmulAsync(WT, L.qkvW, 3 * H, H, attnIn, T, qkv, scratch);
        for (std::size_t t = 0; t < T; ++t) {
            _ops.appendMemoryCopy(q  + t * H, qkv + t * 3 * H,            H * sizeof(float));
            _ops.appendMemoryCopy(k  + t * H, qkv + t * 3 * H + H,        H * sizeof(float));
            _ops.appendMemoryCopy(vv + t * H, qkv + t * 3 * H + 2 * H,    H * sizeof(float));
        }
        // RoPE on Q and K (split/NEOX), per-layer theta (global vs local base).
        _ops.ropeInPlaceAsync(q, T, heads, headDim, /*startPos=*/0, theta);
        _ops.ropeInPlaceAsync(k, T, heads, headDim, /*startPos=*/0, theta);
        // Global layers: full-window attention. Sliding layers: local window
        // |kk-pq| <= localRadius (ModernBERT's alternating attention).
        if (c.isGlobalLayer(i)) {
            _ops.attentionEncoderAsync(q, k, vv, T, heads, heads, headDim, scale, attn);
        } else {
            _ops.attentionEncoderWindowedAsync(q, k, vv, T, heads, heads, headDim,
                                               scale, c.localRadius(i), attn);
        }
        _mm.matmulAsync(WT, L.oW, H, H, attn, T, ao, scratch);
        _ops.addResidualAsync(h, ao, T * H);   // h += attn_out

        // ---- MLP block (pre-LN) : GeGLU out = gelu(input) * gate ----
        _ops.layerNormAsync(h, T, H, L.mlpNormW, zbias, eps, normed);
        _mm.matmulAsync(WT, L.wiW, 2 * ffn, H, normed, T, wi, scratch);
        // De-interleave the per-row [input(ffn) | gate(ffn)] halves.
        for (std::size_t t = 0; t < T; ++t) {
            _ops.appendMemoryCopy(gin   + t * ffn, wi + t * 2 * ffn,       ffn * sizeof(float));
            _ops.appendMemoryCopy(ggate + t * ffn, wi + t * 2 * ffn + ffn, ffn * sizeof(float));
        }
        _ops.geluErfMulAsync(gin, ggate, T * ffn);   // gin = gelu_erf(gin) * ggate (ModernBERT uses exact erf GELU)
        _mm.matmulAsync(WT, L.woW, H, ffn, gin, T, mo, scratch);
        _ops.addResidualAsync(h, mo, T * H);   // h += mlp_out
    }

    // Final norm -> res (so the body scratch can be released after the flush).
    _ops.layerNormAsync(h, T, H, _m.finalNormW(), zbias, eps, res);
    _ops.flush();
    return resb;
}

std::vector<float>
LayaEncoderRunner::lastHiddenHost(std::span<const std::int32_t> inputIds,
                                  std::size_t& Tout) {
    compute::ComputeBuffer hb = encode(inputIds, Tout);   // already flushed
    const std::size_t H = _m.config().hidden;
    std::vector<float> host(Tout * H);
    _ops.readbackToHost(host.data(), hb.get(), Tout * H * sizeof(float));
    return host;
}

std::vector<float>
LayaEncoderRunner::embed(std::span<const std::int32_t> inputIds) {
    std::size_t T = 0;
    const std::vector<float> hidden = lastHiddenHost(inputIds, T);
    const std::size_t H = _m.config().hidden;

    std::vector<float> emb(H, 0.0F);
    if (_m.config().meanPooling) {
        for (std::size_t t = 0; t < T; ++t) {
            for (std::size_t d = 0; d < H; ++d) {
                emb[d] += hidden[t * H + d];
            }
        }
        const float inv = (T > 0) ? (1.0F / static_cast<float>(T)) : 0.0F;
        for (float& f : emb) {
            f *= inv;
        }
    } else {
        // CLS pooling: row 0.
        std::copy_n(hidden.begin(), H, emb.begin());
    }
    return emb;
}

} // namespace mimirmind::runtime::encoder::laya
