// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/audio/WhisperRunner.hpp"

#include "compute/ComputeMatmul.hpp"
#include "compute/ComputeOps.hpp"
#include "compute/Embedding.hpp"
#include "core/gguf/GgufTypes.hpp"
#include "runtime/audio/WhisperConvStem.hpp"
#include "runtime/audio/WhisperModel.hpp"

#include <cmath>
#include <stdexcept>

namespace mimirmind::runtime::audio {

namespace {
constexpr float kLnEps = 1e-5F;   // Whisper LayerNorm eps (fixed, not in config)
} // namespace

WhisperRunner::WhisperRunner(const WhisperModel& model, compute::ComputeOps& ops,
                             compute::ComputeMatmul& matmul)
    : _m{model}, _ops{ops}, _mm{matmul} {}

compute::ComputeBuffer
WhisperRunner::runEncoder(const float* mel, std::size_t nMels,
                          std::size_t nFrames, std::size_t& nCtxOut) {
    const WhisperConfig& c = _m.config();
    const std::size_t d   = c.dModel;
    const std::size_t ffn = c.encoderFfn;
    const std::size_t heads = c.encoderHeads;
    const std::size_t headDim = c.encoderHeadDim();
    const float scale = 1.0F / std::sqrt(static_cast<float>(headDim));
    const auto WT = _m.matmulType();

    // Bit-exact F32 projections for the ASR parity gate (same rationale as the
    // encoder reranker path): keep TF32 downcast out of this forward.
    compute::ScopedExactF32 exactF32{_mm};

    // ---- Conv stem on the host --------------------------------------------
    // Read the (small) conv weights back to host so the stem is backend-neutral,
    // run it, then upload the [nCtx x d] result to the device.
    auto readbackHost = [&](const float* devPtr, std::size_t n) {
        std::vector<float> h(n);
        _ops.readbackToHost(h.data(), devPtr, n * sizeof(float));
        return h;
    };
    const std::vector<float> conv1W = readbackHost(_m.conv1W(), d * nMels * 3);
    const std::vector<float> conv1B = readbackHost(_m.conv1B(), d);
    const std::vector<float> conv2W = readbackHost(_m.conv2W(), d * d * 3);
    const std::vector<float> conv2B = readbackHost(_m.conv2B(), d);

    const ConvStemOutput stem =
        whisperConvStem(mel, nMels, nFrames, d, conv1W.data(), conv1B.data(),
                        conv2W.data(), conv2B.data());
    const std::size_t T = stem.nCtx;
    nCtxOut = T;
    if (T == 0) {
        throw std::runtime_error("WhisperRunner: empty encoder context");
    }

    auto alloc = [&](std::size_t n) { return _ops.allocate(n * sizeof(float)); };
    auto fp = [](compute::ComputeBuffer& b) { return static_cast<float*>(b.get()); };

    compute::ComputeBuffer hb   = alloc(T * d);
    compute::ComputeBuffer lnb  = alloc(T * d);
    compute::ComputeBuffer qb   = alloc(T * d);
    compute::ComputeBuffer kb   = alloc(T * d);
    compute::ComputeBuffer vb   = alloc(T * d);
    compute::ComputeBuffer attnb = alloc(T * d);
    compute::ComputeBuffer aob  = alloc(T * d);
    compute::ComputeBuffer interb = alloc(T * ffn);
    compute::ComputeBuffer ffnb = alloc(T * d);
    compute::ComputeBuffer scr  = alloc(T * ffn);

    float* h    = fp(hb);
    float* ln   = fp(lnb);
    float* q    = fp(qb);
    float* k    = fp(kb);
    float* v    = fp(vb);
    float* attn = fp(attnb);
    float* ao   = fp(aob);
    float* inter = fp(interb);
    float* ffnO = fp(ffnb);
    float* scratch = fp(scr);

    // Upload conv-stem output, add the encoder positional table (first T rows).
    _ops.uploadHostBytes(h, stem.data.data(), T * d * sizeof(float));
    _ops.addResidualAsync(h, _m.encPosEmb(), T * d);

    for (std::size_t i = 0; i < c.encoderLayers; ++i) {
        const WhisperEncoderLayer& L = _m.encoderLayer(i);

        // Self-attention sublayer (pre-norm).
        _ops.layerNormAsync(h, T, d, L.attnLnW, L.attnLnB, kLnEps, ln);
        _mm.matmulAsync(WT, L.qW, d, d, ln, T, q, scratch);
        _ops.addBiasAsync(q, T, d, L.qB);
        _mm.matmulAsync(WT, L.kW, d, d, ln, T, k, scratch);   // k_proj: no bias
        _mm.matmulAsync(WT, L.vW, d, d, ln, T, v, scratch);
        _ops.addBiasAsync(v, T, d, L.vB);
        _ops.attentionEncoderAsync(q, k, v, T, heads, heads, headDim, scale, attn);
        _mm.matmulAsync(WT, L.oW, d, d, attn, T, ao, scratch);
        _ops.addBiasAsync(ao, T, d, L.oB);
        _ops.addResidualAsync(ao, h, T * d);                  // residual add

        // Feed-forward sublayer (pre-norm).
        _ops.layerNormAsync(ao, T, d, L.finalLnW, L.finalLnB, kLnEps, ln);
        _mm.matmulAsync(WT, L.fc1W, ffn, d, ln, T, inter, scratch);
        _ops.addBiasAsync(inter, T, ffn, L.fc1B);
        _ops.geluErfAsync(inter, T * ffn);
        _mm.matmulAsync(WT, L.fc2W, d, ffn, inter, T, ffnO, scratch);
        _ops.addBiasAsync(ffnO, T, d, L.fc2B);
        _ops.addResidualAsync(ffnO, ao, T * d);
        _ops.appendMemoryCopy(h, ffnO, T * d * sizeof(float));  // h := layer out
    }

    // Post-encoder layer norm -> final encoder states (reuse ln buffer's slot).
    compute::ComputeBuffer encb = alloc(T * d);
    _ops.layerNormAsync(h, T, d, _m.encLnW(), _m.encLnB(), kLnEps, fp(encb));
    _ops.flush();
    return encb;
}

std::vector<std::int32_t>
WhisperRunner::transcribeGreedy(const float* mel, std::size_t nMels,
                                std::size_t nFrames,
                                const WhisperDecodeOptions& opt) {
    const WhisperConfig& c = _m.config();
    const std::size_t d   = c.dModel;
    const std::size_t ffn = c.decoderFfn;
    const std::size_t heads = c.decoderHeads;
    const std::size_t headDim = c.decoderHeadDim();
    const std::size_t vocab = c.vocab;
    const float scale = 1.0F / std::sqrt(static_cast<float>(headDim));
    const auto WT = _m.matmulType();
    const auto F32 = core::gguf::GgmlType::F32;

    std::size_t nCtx = 0;
    compute::ComputeBuffer encb = runEncoder(mel, nMels, nFrames, nCtx);
    const float* enc = static_cast<const float*>(encb.get());

    // Tied output projection: proj_out if present, else embed_tokens (F32).
    const float* projW = _m.projOut() != nullptr ? _m.projOut() : _m.decTokEmb();
    const auto   projT = _m.projOut() != nullptr ? WT : F32;

    compute::ScopedExactF32 exactF32{_mm};

    std::vector<std::int32_t> tokens = whisperInitialPromptTokens(opt);
    const std::size_t Tmax = tokens.size() + opt.maxNewTokens;

    auto alloc = [&](std::size_t n) { return _ops.allocate(n * sizeof(float)); };
    auto fp = [](compute::ComputeBuffer& b) { return static_cast<float*>(b.get()); };

    // Decoder scratch, sized for the worst-case prefix Tmax; each step uses the
    // first t rows. Cross-attn K/V are over the fixed nCtx encoder states.
    compute::ComputeBuffer xb    = alloc(Tmax * d);
    compute::ComputeBuffer hb    = alloc(Tmax * d);
    compute::ComputeBuffer lnb   = alloc(Tmax * d);
    compute::ComputeBuffer qb    = alloc(Tmax * d);
    compute::ComputeBuffer kb    = alloc(Tmax * d);
    compute::ComputeBuffer vb    = alloc(Tmax * d);
    compute::ComputeBuffer sab   = alloc(Tmax * d);
    compute::ComputeBuffer aob   = alloc(Tmax * d);
    compute::ComputeBuffer cqb   = alloc(Tmax * d);
    compute::ComputeBuffer ckb   = alloc(nCtx * d);
    compute::ComputeBuffer cvb   = alloc(nCtx * d);
    compute::ComputeBuffer cab   = alloc(Tmax * d);
    compute::ComputeBuffer cob   = alloc(Tmax * d);
    compute::ComputeBuffer interb = alloc(Tmax * ffn);
    compute::ComputeBuffer ffnb  = alloc(Tmax * d);
    compute::ComputeBuffer scr   = alloc(Tmax * ffn);
    compute::ComputeBuffer logitb = alloc(vocab);

    float* x    = fp(xb);
    float* h    = fp(hb);
    float* ln   = fp(lnb);
    float* q    = fp(qb);
    float* k    = fp(kb);
    float* v    = fp(vb);
    float* sa   = fp(sab);
    float* ao   = fp(aob);
    float* cq   = fp(cqb);
    float* ck   = fp(ckb);
    float* cv   = fp(cvb);
    float* ca   = fp(cab);
    float* co   = fp(cob);
    float* inter = fp(interb);
    float* ffnO = fp(ffnb);
    float* scratch = fp(scr);
    float* logit = fp(logitb);

    std::vector<float> hostLogits(vocab);

    while (tokens.size() < Tmax) {
        const std::size_t t = tokens.size();

        // Token + learned positional embeddings for the whole prefix.
        compute::embeddingLookup(F32, _m.decTokEmb(), d, vocab,
                                 std::span<const std::int32_t>{tokens}, x);
        _ops.appendMemoryCopy(h, x, t * d * sizeof(float));
        _ops.addResidualAsync(h, _m.decPosEmb(), t * d);   // pos rows [0, t)

        for (std::size_t i = 0; i < c.decoderLayers; ++i) {
            const WhisperDecoderLayer& L = _m.decoderLayer(i);

            // Masked self-attention (pre-norm).
            _ops.layerNormAsync(h, t, d, L.selfLnW, L.selfLnB, kLnEps, ln);
            _mm.matmulAsync(WT, L.qW, d, d, ln, t, q, scratch);
            _ops.addBiasAsync(q, t, d, L.qB);
            _mm.matmulAsync(WT, L.kW, d, d, ln, t, k, scratch);   // no bias
            _mm.matmulAsync(WT, L.vW, d, d, ln, t, v, scratch);
            _ops.addBiasAsync(v, t, d, L.vB);
            _ops.attentionAsync(q, k, v, t, t, heads, heads, headDim,
                                /*positionOffset=*/0, scale, sa);
            _mm.matmulAsync(WT, L.oW, d, d, sa, t, ao, scratch);
            _ops.addBiasAsync(ao, t, d, L.oB);
            _ops.addResidualAsync(ao, h, t * d);

            // Cross-attention over encoder states (pre-norm).
            _ops.layerNormAsync(ao, t, d, L.crossLnW, L.crossLnB, kLnEps, ln);
            _mm.matmulAsync(WT, L.cqW, d, d, ln, t, cq, scratch);
            _ops.addBiasAsync(cq, t, d, L.cqB);
            _mm.matmulAsync(WT, L.ckW, d, d, enc, nCtx, ck, scratch); // no bias
            _mm.matmulAsync(WT, L.cvW, d, d, enc, nCtx, cv, scratch);
            _ops.addBiasAsync(cv, nCtx, d, L.cvB);
            _ops.attentionEncoderCrossAsync(cq, ck, cv, t, nCtx, heads, heads,
                                            headDim, scale, ca);
            _mm.matmulAsync(WT, L.coW, d, d, ca, t, co, scratch);
            _ops.addBiasAsync(co, t, d, L.coB);
            _ops.addResidualAsync(co, ao, t * d);

            // Feed-forward (pre-norm).
            _ops.layerNormAsync(co, t, d, L.finalLnW, L.finalLnB, kLnEps, ln);
            _mm.matmulAsync(WT, L.fc1W, ffn, d, ln, t, inter, scratch);
            _ops.addBiasAsync(inter, t, ffn, L.fc1B);
            _ops.geluErfAsync(inter, t * ffn);
            _mm.matmulAsync(WT, L.fc2W, d, ffn, inter, t, ffnO, scratch);
            _ops.addBiasAsync(ffnO, t, d, L.fc2B);
            _ops.addResidualAsync(ffnO, co, t * d);
            _ops.appendMemoryCopy(h, ffnO, t * d * sizeof(float));
        }

        // Post-decoder LN, then project only the last row to logits.
        _ops.layerNormAsync(h, t, d, _m.decLnW(), _m.decLnB(), kLnEps, ln);
        const float* lastRow = ln + (t - 1) * d;
        _mm.matmulAsync(projT, projW, vocab, d, lastRow, 1, logit, scratch);

        _ops.flush();
        _ops.readbackToHost(hostLogits.data(), logit, vocab * sizeof(float));

        // Language auto-detect: at decoder position 1 predict the language
        // token (argmax over the language-token range), then force the task
        // and no-timestamps tokens at positions 2/3 before free decoding.
        std::int32_t next;
        if (opt.detectLanguage && t == 1) {
            next = argmaxRange(std::span<const float>{hostLogits},
                               opt.special.langEn, opt.special.translate);
        } else if (opt.detectLanguage && t == 2) {
            next = opt.translate ? opt.special.translate
                                 : opt.special.transcribe;
        } else if (opt.detectLanguage && t == 3 && !opt.timestamps) {
            next = opt.special.noTimestamps;
        } else {
            next = argmaxRow(std::span<const float>{hostLogits});
            if (next == opt.special.eot) {
                break;
            }
        }
        tokens.push_back(next);
    }

    return tokens;
}

} // namespace mimirmind::runtime::audio
