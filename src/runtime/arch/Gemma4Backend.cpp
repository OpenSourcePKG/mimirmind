#include "runtime/arch/Gemma4Backend.hpp"

#include "compute/Attention.hpp"
#include "compute/GpuMatmul.hpp"
#include "compute/GpuOps.hpp"
#include "model/GgufTypes.hpp"
#include "model/LlmConfig.hpp"
#include "model/WeightsMap.hpp"
#include "runtime/BlockBuffers.hpp"
#include "runtime/KvCache.hpp"
#include "runtime/Log.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace mimirmind::runtime::arch {

Gemma4Backend::Gemma4Backend(const model::LlmConfig&   config,
                             const model::WeightsMap&  weights,
                             compute::GpuOps&          ops,
                             compute::GpuMatmul&       gmm)
    : _config{config}, _weights{weights}, _ops{ops}, _gmm{gmm} {
    buildKvSharePattern();

    // Optional global `rope_freqs.weight` (F32 [head_dim/2]) — used as
    // `freq_factors` in proportional RoPE for non-SWA (global) layers.
    // The 26B-A4B Q6_K we run during M8 bring-up ships this tensor with
    // count = head_dim (256), but ggml_rope_ext expects [head_dim/2]:
    // we accept any tensor that has *at least* halfDim usable floats and
    // ignore the trailing slots.
    if (const auto* rf = _weights.find("rope_freqs.weight");
        rf != nullptr && rf->type == model::GgmlType::F32) {
        const std::size_t halfDim = _config.headDim() / 2;
        if (rf->nelements >= halfDim) {
            _ropeFreqsForFullAttn = static_cast<const float*>(rf->usmPtr);
            MM_LOG_INFO("gemma4",
                        "proportional RoPE enabled — rope_freqs.weight has "
                        "{} float(s), using first {} for non-SWA layers",
                        rf->nelements, halfDim);
        } else {
            MM_LOG_WARN("gemma4",
                        "rope_freqs.weight has {} float(s) < halfDim={} — "
                        "proportional RoPE disabled, falling back to plain "
                        "base for non-SWA layers", rf->nelements, halfDim);
        }
    } else {
        MM_LOG_INFO("gemma4",
                    "no rope_freqs.weight — non-SWA layers use plain RoPE "
                    "with base={}", _config.ropeFreqBase);
    }

    MM_LOG_INFO("gemma4",
                "Gemma4Backend ready — blocks={} d_model={} ff={} heads={} "
                "kv={} experts={} top_k={} rope_base={}/{} swa_layers={}",
                _config.blockCount, _config.embeddingLength,
                _config.feedForwardLength, _config.headCount,
                _config.headCountKv, _config.expertCount,
                _config.expertUsedCount,
                _config.ropeFreqBase, _config.ropeFreqBaseSwa,
                _config.slidingWindowPattern.size());
}

bool Gemma4Backend::isSwaLayer(std::size_t blockIdx) const noexcept {
    return blockIdx < _config.slidingWindowPattern.size() &&
           _config.slidingWindowPattern[blockIdx];
}

void Gemma4Backend::buildKvSharePattern() {
    _kvSource.assign(_config.blockCount, 0);
    std::optional<std::size_t> lastKv;
    std::string pattern;
    pattern.reserve(_config.blockCount);
    for (std::size_t b = 0; b < _config.blockCount; ++b) {
        const bool hasOwn =
            _weights.findBlock(b, "attn_k.weight") != nullptr &&
            _weights.findBlock(b, "attn_v.weight") != nullptr;
        if (hasOwn) {
            _kvSource[b] = b;
            lastKv       = b;
            pattern.push_back('K');
        } else if (lastKv) {
            _kvSource[b] = *lastKv;
            pattern.push_back('.');
        } else {
            throw std::runtime_error(
                "gemma4: block " + std::to_string(b) +
                " has no K/V and no earlier source — model is malformed");
        }
    }
    MM_LOG_INFO("gemma4",
                "KV-share pattern: {} ({} owns, {} shared)",
                pattern,
                std::count(pattern.begin(), pattern.end(), 'K'),
                std::count(pattern.begin(), pattern.end(), '.'));
}

void Gemma4Backend::runBlock(std::size_t   blockIdx,
                             float*        x,
                             std::size_t   T,
                             KvCache&      cache,
                             BlockBuffers& s,
                             bool          traceBlock0) {
    namespace cmp = mimirmind::compute;

    const bool diag = (blockIdx == 0 && cache.length() == 0 && traceBlock0);
    auto trace = [&](const char* tag) {
        if (diag) MM_LOG_INFO("blkdiag-g4", "blk0 {}", tag);
    };
    trace("enter");

    auto require = [&](const char* suffix) {
        const auto* t = _weights.findBlock(blockIdx, suffix);
        if (t == nullptr) {
            throw std::runtime_error(
                "Gemma4Backend: missing tensor blk." +
                std::to_string(blockIdx) + "." + suffix);
        }
        return t;
    };

    const auto* attnNorm    = require("attn_norm.weight");
    const auto* qW          = require("attn_q.weight");
    const auto* qNorm       = require("attn_q_norm.weight");
    const auto* oW          = require("attn_output.weight");
    const auto* attnPost    = require("post_attention_norm.weight");
    const auto* ffnNorm     = require("ffn_norm.weight");
    const auto* ffnGate     = require("ffn_gate.weight");
    const auto* ffnUp       = require("ffn_up.weight");
    const auto* ffnDown     = require("ffn_down.weight");
    const auto* ffwPost1    = require("post_ffw_norm_1.weight");
    const auto* ffwPost     = require("post_ffw_norm.weight");
    const auto* outScale    = require("layer_output_scale.weight");

    // Path B (MoE) tensors. Required for gemma4 26B-A4B; every block has them.
    const auto* preNorm2     = require("pre_ffw_norm_2.weight");
    const auto* postNorm2    = require("post_ffw_norm_2.weight");
    const auto* routerScale  = require("ffn_gate_inp.scale");
    const auto* routerW      = require("ffn_gate_inp.weight");
    const auto* expGateUp    = require("ffn_gate_up_exps.weight");
    const auto* expDown      = require("ffn_down_exps.weight");
    const auto* expDownScale = require("ffn_down_exps.scale");

    // KV-sharing: only blocks that own K/V do the K/V projection +
    // K-norm + K-RoPE. Other blocks read K/V from the source block's
    // cache slot.
    const std::size_t kvLayer  = _kvSource[blockIdx];
    const bool        hasOwnKv = (kvLayer == blockIdx);
    const auto* kW    = hasOwnKv ? require("attn_k.weight")      : nullptr;
    const auto* vW    = hasOwnKv ? require("attn_v.weight")      : nullptr;
    const auto* kNorm = hasOwnKv ? require("attn_k_norm.weight") : nullptr;

    // SWA layers use a different RoPE base AND no freq_factors.
    // Global-attention layers use the long-context base AND proportional
    // RoPE freq_factors (if the model shipped them). Mirrors gemma4.cpp:
    //   freq_base_l = is_swa(il) ? rope_freq_base_train_swa : rope_freq_base;
    //   freq_factors = is_swa(il) ? nullptr : layer.rope_freqs;
    const bool   isSwa    = isSwaLayer(blockIdx);
    const float  ropeBase = isSwa ? _config.ropeFreqBaseSwa
                                  : _config.ropeFreqBase;
    const float* freqFactors = isSwa ? nullptr : _ropeFreqsForFullAttn;

    const std::size_t d_model  = s.d_model;
    const std::size_t q_dim    = s.q_dim;
    const std::size_t kv_dim   = _config.headCountKv * _config.headDim();
    const std::size_t ff_dim   = s.ff_dim;
    const std::size_t head_dim = _config.headDim();
    const std::size_t curLen   = cache.length();
    const std::size_t totalLen = curLen + T;

    float* const normBuf       = s.normBuf.as<float>();
    float* const qBuf          = s.qBuf.as<float>();
    float* const attnOutBuf    = s.attnOut.as<float>();
    float* const projOutBuf    = s.projOut.as<float>();
    float* const gateOutBuf    = s.gateOut.as<float>();
    float* const upOutBuf      = s.upOut.as<float>();
    float* const matmulScratch = s.matmulScratch.as<float>();
    float* const scoreScratch  = s.scoreScratch.as<float>();
    float* const moeAccumBuf   = s.moeAccumBuf.as<float>();
    float* const expertOutBuf  = s.expertOutBuf.as<float>();

    // --- pre-attention RMSNorm ----------------------------------------

    trace("attn rmsNorm");
    _ops.rmsNormAsync(x, T, d_model,
                      static_cast<const float*>(attnNorm->usmPtr),
                      _config.rmsNormEps,
                      normBuf);

    auto projectAsync = [&](const model::GgufTensor* W,
                            std::size_t N, float* dst) {
        _gmm.matmulAsync(W->type, W->usmPtr, N, d_model,
                         normBuf, T, dst, matmulScratch);
    };

    trace("Q projection");
    projectAsync(qW, q_dim,  qBuf);

    if (hasOwnKv) {
        float* const kSlot = cache.writeSlotK(blockIdx);
        float* const vSlot = cache.writeSlotV(blockIdx);
        trace("K projection");
        projectAsync(kW, kv_dim, kSlot);
        trace("V projection");
        projectAsync(vW, kv_dim, vSlot);

        // K-norm BEFORE RoPE (per-head, like Q-norm below).
        trace("K-norm");
        _ops.rmsNormAsync(kSlot, T * _config.headCountKv, head_dim,
                          static_cast<const float*>(kNorm->usmPtr),
                          _config.rmsNormEps,
                          kSlot);              // in-place

        // V-norm: bare RMSNorm over head_dim, no learned weight. Gemma 4
        // pushes V through `ggml_rms_norm` before it enters the KV cache
        // (gemma4.cpp ~line 256).
        trace("V-norm (no weight)");
        _ops.rmsNormNoWeightAsync(vSlot, T * _config.headCountKv, head_dim,
                                  _config.rmsNormEps,
                                  vSlot);            // in-place

        trace("RoPE K");
        if (freqFactors != nullptr) {
            _ops.ropeInPlaceWithFactorsAsync(kSlot, freqFactors, T,
                                             _config.headCountKv, head_dim,
                                             curLen, ropeBase);
        } else {
            _ops.ropeInPlaceAsync(kSlot, T,
                                  _config.headCountKv, head_dim, curLen,
                                  ropeBase);
        }
    } else {
        trace("KV reused from earlier layer");
    }

    // Q-K-norm BEFORE RoPE (per-head RMSNorm over head_dim).
    trace("Q-norm");
    _ops.rmsNormAsync(qBuf, T * _config.headCount, head_dim,
                      static_cast<const float*>(qNorm->usmPtr),
                      _config.rmsNormEps,
                      qBuf);                   // in-place

    trace("RoPE Q");
    if (freqFactors != nullptr) {
        _ops.ropeInPlaceWithFactorsAsync(qBuf, freqFactors, T,
                                         _config.headCount, head_dim,
                                         curLen, ropeBase);
    } else {
        _ops.ropeInPlaceAsync(qBuf, T,
                              _config.headCount, head_dim, curLen,
                              ropeBase);
    }

    // f_attention_scale = 1.0 (gemma4.cpp:11). Our multiHeadAttention
    // divides Q·K by 1/sqrt(headDim) internally; pre-scale Q by
    // sqrt(head_dim) so it cancels and effective scale = 1.0.
    {
        const float qScale = std::sqrt(static_cast<float>(head_dim));
        trace("Q pre-scale sqrt(head_dim)");
        _ops.mulScalarAsync(qBuf, qScale, T * q_dim);
    }

    trace("QKV sync (before CPU attention)");
    _gmm.sync();

    trace("multiHeadAttention (CPU)");
    cmp::multiHeadAttention(
        qBuf,
        cache.baseK(kvLayer),
        cache.baseV(kvLayer),
        T, totalLen,
        _config.headCount, _config.headCountKv, head_dim,
        curLen,
        scoreScratch,
        attnOutBuf);

    trace("O projection");
    _gmm.matmul(oW->type, oW->usmPtr, d_model, q_dim,
                attnOutBuf, T,
                projOutBuf, matmulScratch);

    trace("attn_post_norm");
    _ops.rmsNormAsync(projOutBuf, T, d_model,
                      static_cast<const float*>(attnPost->usmPtr),
                      _config.rmsNormEps,
                      projOutBuf);            // in-place

    trace("attn residual");
    _ops.addResidualAsync(x, projOutBuf, T * d_model);
    // x now holds sa_out = inpL + attn_post_norm(attn_out).

    // --- FFN path A — dense SwiGLU with GELU ---------------------------

    trace("ffn_norm (path A pre)");
    _ops.rmsNormAsync(x, T, d_model,
                      static_cast<const float*>(ffnNorm->usmPtr),
                      _config.rmsNormEps,
                      normBuf);

    trace("FFN gate proj");
    _gmm.matmulAsync(ffnGate->type, ffnGate->usmPtr, ff_dim, d_model,
                     normBuf, T, gateOutBuf, matmulScratch);
    trace("FFN up proj");
    _gmm.matmulAsync(ffnUp->type, ffnUp->usmPtr, ff_dim, d_model,
                     normBuf, T, upOutBuf, matmulScratch);

    trace("GELU + mul (fused)");
    _ops.geluMulAsync(gateOutBuf, upOutBuf, T * ff_dim);

    trace("FFN down proj");
    _gmm.matmul(ffnDown->type, ffnDown->usmPtr, d_model, ff_dim,
                gateOutBuf, T,
                projOutBuf, matmulScratch);

    trace("post_ffw_norm_1 (path A post)");
    _ops.rmsNormAsync(projOutBuf, T, d_model,
                      static_cast<const float*>(ffwPost1->usmPtr),
                      _config.rmsNormEps,
                      projOutBuf);            // in-place

    // --- Path B — MoE -------------------------------------------------

    trace("path B: pre_ffw_norm_2");
    _ops.rmsNormAsync(x, T, d_model,
                      static_cast<const float*>(preNorm2->usmPtr),
                      _config.rmsNormEps,
                      normBuf);

    // router_in = RMSNorm(x, ffn_gate_inp.scale) * 1/sqrt(d_model).
    trace("path B: router rmsNorm + scale");
    _ops.rmsNormAsync(x, T, d_model,
                      static_cast<const float*>(routerScale->usmPtr),
                      _config.rmsNormEps,
                      attnOutBuf);
    const float invSqrtDm = 1.0F /
        std::sqrt(static_cast<float>(d_model));
    _ops.mulScalarAsync(attnOutBuf, invSqrtDm, T * d_model);

    const std::size_t nExperts = _config.expertCount;
    const std::size_t K        = _config.expertUsedCount;
    trace("path B: router matmul (CPU)");
    _gmm.matmul(routerW->type, routerW->usmPtr,
                nExperts, d_model,
                attnOutBuf, T,
                upOutBuf, matmulScratch);

    // CPU softmax + top-K per token (renormalised so kept K sums to 1).
    std::vector<std::int32_t> topKIdx(T * K);
    std::vector<float>        topKWeight(T * K);
    {
        std::vector<float> probs(nExperts);
        for (std::size_t t = 0; t < T; ++t) {
            const float* row = upOutBuf + t * nExperts;
            float maxL = row[0];
            for (std::size_t e = 1; e < nExperts; ++e) {
                if (row[e] > maxL) maxL = row[e];
            }
            double sum = 0.0;
            for (std::size_t e = 0; e < nExperts; ++e) {
                probs[e] = std::exp(row[e] - maxL);
                sum += static_cast<double>(probs[e]);
            }
            const float invSum = static_cast<float>(1.0 / sum);
            for (auto& p : probs) p *= invSum;

            std::vector<std::size_t> idx(nExperts);
            std::iota(idx.begin(), idx.end(), 0);
            std::partial_sort(idx.begin(),
                              idx.begin() + static_cast<std::ptrdiff_t>(K),
                              idx.end(),
                              [&](std::size_t a, std::size_t b) {
                                  return probs[a] > probs[b];
                              });
            double kept = 0.0;
            for (std::size_t k = 0; k < K; ++k) {
                kept += static_cast<double>(probs[idx[k]]);
            }
            const float invKept = kept > 0.0 ? static_cast<float>(1.0 / kept) : 1.0F;
            for (std::size_t k = 0; k < K; ++k) {
                topKIdx[t * K + k]    = static_cast<std::int32_t>(idx[k]);
                topKWeight[t * K + k] = probs[idx[k]] * invKept;
            }
        }
    }

    trace("path B: zero accumulator");
    _ops.mulScalarAsync(moeAccumBuf, 0.0F, T * d_model);

    //   gate_up_exps: Q6_K [K=d_model, N=gate_up_fused, n_exp]
    //                  per-expert bytes = N * (K/256) * 210
    //   down_exps:    Q8_0 [K=ff_per_expert, N=d_model, n_exp]
    //                  per-expert bytes = N * (K/32)  * 34
    const std::size_t gateUpFused = expGateUp->dimensions.size() >= 2
                                      ? expGateUp->dimensions[1] : 0;
    const std::size_t ffPerExpert = gateUpFused / 2;
    if (gateUpFused == 0 || (gateUpFused % 2) != 0) {
        throw std::runtime_error(
            "Gemma4Backend: ffn_gate_up_exps has unexpected fused dim " +
            std::to_string(gateUpFused));
    }

    constexpr std::size_t kQ6KBlockElems = 256;
    constexpr std::size_t kQ6KBlockBytes = 210;
    constexpr std::size_t kQ8_0BlockElems = 32;
    constexpr std::size_t kQ8_0BlockBytes = 34;

    const std::size_t expertBytesGateUp =
        gateUpFused * (d_model / kQ6KBlockElems) * kQ6KBlockBytes;
    const std::size_t expertBytesDown =
        d_model * (ffPerExpert / kQ8_0BlockElems) * kQ8_0BlockBytes;

    auto* const expGateUpBase =
        static_cast<const std::uint8_t*>(expGateUp->usmPtr);
    auto* const expDownBase =
        static_cast<const std::uint8_t*>(expDown->usmPtr);
    const float* const expDownScalePtr =
        static_cast<const float*>(expDownScale->usmPtr);

    trace("path B: per-token expert dispatch");
    for (std::size_t t = 0; t < T; ++t) {
        float* const pathBInT  = normBuf      + t * d_model;
        float* const accumT    = moeAccumBuf  + t * d_model;
        for (std::size_t k = 0; k < K; ++k) {
            const std::size_t e =
                static_cast<std::size_t>(topKIdx[t * K + k]);
            const float       routerWeight = topKWeight[t * K + k];

            const void* Wgu = expGateUpBase + e * expertBytesGateUp;
            const void* Wd  = expDownBase   + e * expertBytesDown;

            _gmm.matmulAsync(expGateUp->type, Wgu,
                             gateUpFused, d_model,
                             pathBInT, 1,
                             gateOutBuf, matmulScratch);

            _ops.geluMulAsync(gateOutBuf, gateOutBuf + ffPerExpert,
                              ffPerExpert);

            _gmm.matmul(expDown->type, Wd,
                        d_model, ffPerExpert,
                        gateOutBuf, 1,
                        expertOutBuf, matmulScratch);

            // ffn_down_exps.scale[e] · routerWeight, fused into one mulScalar.
            const float combined = routerWeight * expDownScalePtr[e];
            _ops.mulScalarAsync(expertOutBuf, combined, d_model);
            _ops.addResidualAsync(accumT, expertOutBuf, d_model);
        }
    }

    trace("path B: post_ffw_norm_2");
    _ops.rmsNormAsync(moeAccumBuf, T, d_model,
                      static_cast<const float*>(postNorm2->usmPtr),
                      _config.rmsNormEps,
                      moeAccumBuf);

    trace("combined = pathA_out + pathB_out");
    _ops.addResidualAsync(moeAccumBuf, projOutBuf, T * d_model);

    trace("post_ffw_norm (combined)");
    _ops.rmsNormAsync(moeAccumBuf, T, d_model,
                      static_cast<const float*>(ffwPost->usmPtr),
                      _config.rmsNormEps,
                      moeAccumBuf);

    trace("ffn residual");
    _ops.addResidualAsync(x, moeAccumBuf, T * d_model);

    // layer_output_scale.weight is a USM F32[1] — read directly from CPU.
    const float scaleVal = *static_cast<const float*>(outScale->usmPtr);
    trace("layer_output_scale");
    _ops.mulScalarAsync(x, scaleVal, T * d_model);
}

} // namespace mimirmind::runtime::arch