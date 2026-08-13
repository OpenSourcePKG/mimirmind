// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/dflash/DFlashDraftRunner.hpp"

#include "runtime/dflash/DFlashDraftModel.hpp"
#include "compute/ComputeMatmul.hpp"
#include "compute/ComputeOps.hpp"
#include "core/gguf/GgufTypes.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace mimirmind::runtime::dflash {

namespace {
// Qwen3.6 drafter config.json constants (match the golden / the checkpoint).
// TODO(P3.3): parse these from config.json instead of hard-coding.
constexpr float kRmsEps   = 1e-6F;    // rms_norm_eps
constexpr float kRopeBase = 1e7F;     // rope_parameters.rope_theta
} // namespace

void DFlashDraftRunner::materializeContext(const float* target_hidden,
                                           std::size_t  ctxLen,
                                           float*       out) {
    const auto& c = _m.config();
    const std::size_t H = c.hidden;         // 2048
    const std::size_t K = c.taps * c.hidden; // 16384 (fc in-features)

    // fc: [ctxLen, K] x fc.weight[H, K]^T -> tmp[ctxLen, H]. BF16-TC GEMM.
    // scratch covers the matmul workspace (rows x max(N,K)); K dominates.
    compute::ComputeBuffer tmpBuf     = _ops.allocate(ctxLen * H * sizeof(float));
    compute::ComputeBuffer scratchBuf = _ops.allocate(ctxLen * K * sizeof(float));
    auto* tmp     = static_cast<float*>(tmpBuf.get());
    auto* scratch = static_cast<float*>(scratchBuf.get());

    _mm.matmulAsync(core::gguf::GgmlType::BF16, _m.fc(), H, K,
                    target_hidden, ctxLen, tmp, scratch);

    // hidden_norm: RMSNorm(tmp) with the F32 hidden_norm weight -> out.
    _ops.rmsNormAsync(tmp, ctxLen, H, _m.hiddenNorm(), kRmsEps, out);
}

void DFlashDraftRunner::draftForward(const float* noise,
                                     const float* target_hidden,
                                     std::size_t  bs,
                                     std::size_t  ctxLen,
                                     float*       out) {
    const auto& c = _m.config();
    const std::size_t H     = c.hidden;              // 2048
    const std::size_t hd    = c.headDim;             // 128
    const std::size_t nQ    = c.nQHeads;             // 32
    const std::size_t nKv   = c.nKvHeads;            // 8
    const std::size_t inter = c.inter;               // 6144
    const std::size_t gqa   = nQ / nKv;              // 4
    const std::size_t qDim  = nQ * hd;               // 4096
    const std::size_t kvDim = nKv * hd;              // 1024
    const std::size_t S     = ctxLen + bs;           // total K/V length
    const float       scale = 1.0F / std::sqrt(static_cast<float>(hd));
    using core::gguf::GgmlType;

    auto alloc = [&](std::size_t n) { return _ops.allocate(n * sizeof(float)); };
    auto fp    = [](compute::ComputeBuffer& b) { return static_cast<float*>(b.get()); };

    // Context projection hidden_norm(fc(target_hidden)) -> [ctxLen, H].
    compute::ComputeBuffer ctxProjB = alloc(ctxLen * H);
    materializeContext(target_hidden, ctxLen, fp(ctxProjB));
    const float* ctxProj = fp(ctxProjB);

    compute::ComputeBuffer hB    = alloc(bs * H);
    compute::ComputeBuffer hlnB  = alloc(bs * H);
    compute::ComputeBuffer hln2B = alloc(bs * H);
    compute::ComputeBuffer qB    = alloc(bs * qDim);
    compute::ComputeBuffer kB    = alloc(S * kvDim);
    compute::ComputeBuffer vB    = alloc(S * kvDim);
    compute::ComputeBuffer attnB = alloc(bs * qDim);
    compute::ComputeBuffer aoB   = alloc(bs * H);
    compute::ComputeBuffer gateB = alloc(bs * inter);
    compute::ComputeBuffer upB   = alloc(bs * inter);
    compute::ComputeBuffer mlpB  = alloc(bs * H);
    compute::ComputeBuffer scrB  = alloc(S * c.taps * H);   // matmul workspace (largest K)
    float* h    = fp(hB);
    float* hln  = fp(hlnB);
    float* hln2 = fp(hln2B);
    float* q    = fp(qB);
    float* k    = fp(kB);
    float* v    = fp(vB);
    float* attn = fp(attnB);
    float* ao   = fp(aoB);
    float* gate = fp(gateB);
    float* up   = fp(upB);
    float* mlp  = fp(mlpB);
    float* scr  = fp(scrB);

    // h = noise (device->device copy on the stream).
    _ops.appendMemoryCopy(h, noise, bs * H * sizeof(float));

    std::vector<float> hq(bs * qDim), hk(S * kvDim), hv(S * kvDim), ha(bs * qDim);

    for (std::size_t L = 0; L < _m.layerCount(); ++L) {
        const auto& w = _m.layer(L);

        // input_layernorm(h) -> hln.
        _ops.rmsNormAsync(h, bs, H, w.inputLn, kRmsEps, hln);

        // Projections. Q from noise, K/V from [ctx_proj ; noise].
        _mm.matmulAsync(GgmlType::BF16, w.qProj, qDim, H, hln, bs, q, scr);
        _mm.matmulAsync(GgmlType::BF16, w.kProj, kvDim, H, ctxProj, ctxLen, k, scr);
        _mm.matmulAsync(GgmlType::BF16, w.kProj, kvDim, H, hln, bs, k + ctxLen * kvDim, scr);
        _mm.matmulAsync(GgmlType::BF16, w.vProj, kvDim, H, ctxProj, ctxLen, v, scr);
        _mm.matmulAsync(GgmlType::BF16, w.vProj, kvDim, H, hln, bs, v + ctxLen * kvDim, scr);

        // Per-head QK-norm (RMSNorm over head_dim), then DFlash rope:
        // Q positions [ctxLen .. ctxLen+bs-1], K positions [0 .. S-1].
        _ops.rmsNormAsync(q, bs * nQ, hd, w.qNorm, kRmsEps, q);
        _ops.rmsNormAsync(k, S * nKv, hd, w.kNorm, kRmsEps, k);
        _ops.ropeInPlaceAsync(q, bs, nQ, hd, ctxLen, kRopeBase);
        _ops.ropeInPlaceAsync(k, S, nKv, hd, 0, kRopeBase);

        // Host attention (parity path): softmax(Q·Kᵀ·scale)·V, GQA, non-causal.
        _ops.flush();
        _ops.readbackToHost(hq.data(), q, bs * qDim * sizeof(float));
        _ops.readbackToHost(hk.data(), k, S * kvDim * sizeof(float));
        _ops.readbackToHost(hv.data(), v, S * kvDim * sizeof(float));
        std::vector<float> sc(S);
        for (std::size_t qh = 0; qh < nQ; ++qh) {
            const std::size_t kvh = qh / gqa;
            for (std::size_t i = 0; i < bs; ++i) {
                const float* qp = &hq[(i * nQ + qh) * hd];
                float maxs = -std::numeric_limits<float>::infinity();
                for (std::size_t j = 0; j < S; ++j) {
                    const float* kp = &hk[(j * nKv + kvh) * hd];
                    float dot = 0.0F;
                    for (std::size_t d = 0; d < hd; ++d) { dot += qp[d] * kp[d]; }
                    sc[j] = dot * scale;
                    maxs  = std::max(maxs, sc[j]);
                }
                float sum = 0.0F;
                for (std::size_t j = 0; j < S; ++j) { sc[j] = std::exp(sc[j] - maxs); sum += sc[j]; }
                const float inv = 1.0F / sum;
                float* ap = &ha[(i * nQ + qh) * hd];
                for (std::size_t d = 0; d < hd; ++d) {
                    float acc = 0.0F;
                    for (std::size_t j = 0; j < S; ++j) {
                        acc += sc[j] * hv[(j * nKv + kvh) * hd + d];
                    }
                    ap[d] = acc * inv;
                }
            }
        }
        _ops.uploadHostBytes(attn, ha.data(), bs * qDim * sizeof(float));

        // o_proj, attention residual + post_attn_ln, MLP, MLP residual.
        _mm.matmulAsync(GgmlType::BF16, w.oProj, H, qDim, attn, bs, ao, scr);
        _ops.addRmsNormAsync(h, ao, bs, H, w.postAttnLn, kRmsEps, hln2);
        _mm.matmulAsync(GgmlType::BF16, w.gateProj, inter, H, hln2, bs, gate, scr);
        _mm.matmulAsync(GgmlType::BF16, w.upProj, inter, H, hln2, bs, up, scr);
        _ops.siluMulAsync(gate, up, bs * inter);
        _mm.matmulAsync(GgmlType::BF16, w.downProj, H, inter, gate, bs, mlp, scr);
        _ops.addResidualAsync(h, mlp, bs * H);
    }

    // Final norm.
    _ops.rmsNormAsync(h, bs, H, _m.norm(), kRmsEps, out);
    _ops.flush();
}

} // namespace mimirmind::runtime::dflash
