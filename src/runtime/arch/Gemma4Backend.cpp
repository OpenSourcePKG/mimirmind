#include "runtime/arch/Gemma4Backend.hpp"

#include "compute/GpuMatmul.hpp"
#include "compute/GpuOps.hpp"
#include "compute/MoeRouting.hpp"
#include "compute/QuantType.hpp"
#include "compute/QuantTypeRegistry.hpp"
#include "model/GgufTypes.hpp"
#include "model/LlmConfig.hpp"
#include "model/WeightsMap.hpp"
#include "runtime/BlockBuffers.hpp"
#include "runtime/KvCache.hpp"
#include "runtime/Log.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mimirmind::runtime::arch {

Gemma4Backend::Gemma4Backend(const model::LlmConfig&   config,
                             const model::WeightsMap&  weights,
                             compute::GpuOps&          ops,
                             compute::GpuMatmul&       gmm)
    : _config{config}, _weights{weights}, _ops{ops}, _gmm{gmm} {
    buildLayerInfos();

    if (const auto* rf = _weights.find("rope_freqs.weight");
        rf != nullptr && rf->type == model::GgmlType::F32) {
        // ggml_rope_ext expects [head_dim/2]; the 26B-A4B GGUF stores 256
        // floats (full head_dim/2 = 256). Accept any tensor whose element
        // count is at least the largest layer's halfDim and use the first
        // halfDim entries per layer.
        std::size_t maxHalfDim = 0;
        for (const auto& li : _layers) {
            maxHalfDim = std::max(maxHalfDim, li.headDim / 2);
        }
        if (rf->nelements >= maxHalfDim) {
            _ropeFreqsForFullAttn = static_cast<const float*>(rf->usmPtr);
            MM_LOG_INFO("gemma4",
                        "proportional RoPE enabled — rope_freqs.weight "
                        "has {} float(s), using first {} for full-attn layers",
                        rf->nelements, maxHalfDim);
        } else {
            MM_LOG_WARN("gemma4",
                        "rope_freqs.weight has {} float(s) < halfDim={} — "
                        "proportional RoPE disabled, full-attn layers will "
                        "use plain RoPE", rf->nelements, maxHalfDim);
        }
    } else {
        MM_LOG_INFO("gemma4",
                    "no rope_freqs.weight — full-attn layers use plain RoPE");
    }

    MM_LOG_INFO("gemma4",
                "Gemma4Backend ready — blocks={} d_model={} ff={} "
                "experts={} top_k={} swa head_dim={} kv={}, "
                "full head_dim={} kv={}",
                _config.blockCount, _config.embeddingLength,
                _config.feedForwardLength,
                _config.expertCount, _config.expertUsedCount,
                _config.keyLengthSwa,
                _config.headCountKvFor(0),
                _config.keyLength,
                _layers.empty() ? 0 : _layers.front().nKvHeads);
}

void Gemma4Backend::buildLayerInfos() {
    _layers.reserve(_config.blockCount);

    std::size_t swaCount = 0, fullCount = 0, altCount = 0;
    std::string pattern;
    pattern.reserve(_config.blockCount);

    for (std::size_t b = 0; b < _config.blockCount; ++b) {
        LayerInfo li{};
        li.isSwa = (b < _config.slidingWindowPattern.size())
                     ? _config.slidingWindowPattern[b]
                     : true; // default to SWA if pattern missing
        li.headDim  = _config.headDim(b);
        li.nHeads   = _config.headCount;
        li.nKvHeads = _config.headCountKvFor(b);
        li.qDim     = li.nHeads   * li.headDim;
        li.kvDim    = li.nKvHeads * li.headDim;
        li.ropeBase = li.isSwa ? _config.ropeFreqBaseSwa
                                : _config.ropeFreqBase;

        // Alternative attention: layer omits attn_v.weight, so V is
        // taken from the raw K projection. Every gemma4 layer must
        // still own attn_k.weight (the assumption that some layers
        // share K from earlier ones was wrong).
        const bool hasV = _weights.findBlock(b, "attn_v.weight") != nullptr;
        const bool hasK = _weights.findBlock(b, "attn_k.weight") != nullptr;
        if (!hasK) {
            throw std::runtime_error(
                "gemma4: block " + std::to_string(b) +
                " missing attn_k.weight — model is malformed");
        }
        li.altAttention = !hasV;

        _layers.push_back(li);
        pattern.push_back(li.isSwa ? (li.altAttention ? 'a' : 's')
                                   : (li.altAttention ? 'A' : 'F'));
        if (li.isSwa) ++swaCount; else ++fullCount;
        if (li.altAttention) ++altCount;
    }

    MM_LOG_INFO("gemma4",
                "layer map: {} ({} SWA, {} full, {} alt-attn V=K)",
                pattern, swaCount, fullCount, altCount);
}

std::vector<std::size_t> Gemma4Backend::kvDimPerLayer() const {
    std::vector<std::size_t> out;
    out.reserve(_layers.size());
    for (const auto& li : _layers) {
        out.push_back(li.kvDim);
    }
    return out;
}

std::pair<std::size_t, std::size_t> Gemma4Backend::maxQKVDims() const {
    std::size_t qMax = 0, kvMax = 0;
    for (const auto& li : _layers) {
        qMax  = std::max(qMax,  li.qDim);
        kvMax = std::max(kvMax, li.kvDim);
    }
    return {qMax, kvMax};
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

    // Per-stage parity dump (writes only when MIMIRMIND_PARITY_DUMP is set).
    // Matches the binary layout llama-parity-dump produces so parity-diff
    // can pair files by name.
    auto dumpStage = [&](const char* stage,
                         const float* p, std::size_t Trow, std::size_t dim) {
        if (_parityDumpPrefix.empty()) {
            return;
        }
        _gmm.sync();
        const std::string fname =
            _parityDumpPrefix + "-blk" + std::to_string(blockIdx) +
            "-" + stage + ".bin";
        std::ofstream f(fname, std::ios::binary);
        if (!f) {
            return;
        }
        const std::uint32_t header[3] = {
            static_cast<std::uint32_t>(blockIdx),
            static_cast<std::uint32_t>(Trow),
            static_cast<std::uint32_t>(dim),
        };
        f.write(reinterpret_cast<const char*>(header), sizeof(header));
        f.write(reinterpret_cast<const char*>(p),
                static_cast<std::streamsize>(Trow * dim * sizeof(float)));
    };

    const auto& li = _layers[blockIdx];

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
    const auto* kW          = require("attn_k.weight");
    const auto* kNorm       = require("attn_k_norm.weight");
    const auto* oW          = require("attn_output.weight");
    const auto* attnPost    = require("post_attention_norm.weight");
    const auto* ffnNorm     = require("ffn_norm.weight");
    const auto* ffnGate     = require("ffn_gate.weight");
    const auto* ffnUp       = require("ffn_up.weight");
    const auto* ffnDown     = require("ffn_down.weight");
    const auto* ffwPost1    = require("post_ffw_norm_1.weight");
    const auto* ffwPost     = require("post_ffw_norm.weight");
    const auto* outScale    = require("layer_output_scale.weight");

    const auto* preNorm2     = require("pre_ffw_norm_2.weight");
    const auto* postNorm2    = require("post_ffw_norm_2.weight");
    const auto* routerScale  = require("ffn_gate_inp.scale");
    const auto* routerW      = require("ffn_gate_inp.weight");
    const auto* expGateUp    = require("ffn_gate_up_exps.weight");
    const auto* expDown      = require("ffn_down_exps.weight");
    const auto* expDownScale = require("ffn_down_exps.scale");

    // vW is optional — when the layer uses alternative attention,
    // V is derived from the raw K projection (see Gemma 4 reference).
    const model::GgufTensor* vW = nullptr;
    if (!li.altAttention) {
        vW = require("attn_v.weight");
    }

    const std::size_t d_model  = s.d_model;
    const std::size_t ff_dim   = s.ff_dim;
    const std::size_t q_dim    = li.qDim;
    const std::size_t kv_dim   = li.kvDim;
    const std::size_t head_dim = li.headDim;
    const std::size_t curLen   = cache.length();
    const std::size_t totalLen = curLen + T;

    float* const normBuf       = s.normBuf.as<float>();
    float* const qBuf          = s.qBuf.as<float>();
    float* const attnOutBuf    = s.attnOut.as<float>();
    float* const projOutBuf    = s.projOut.as<float>();
    float* const gateOutBuf    = s.gateOut.as<float>();
    float* const upOutBuf      = s.upOut.as<float>();
    float* const matmulScratch = s.matmulScratch.as<float>();
    // scoreScratch was the CPU-attention softmax row buffer; the GPU
    // attention kernel keeps the score row in SLM, so it's unused here.
    (void)s.scoreScratch;
    float* const moeAccumBuf   = s.moeAccumBuf.as<float>();
    float* const expertOutBuf  = s.expertOutBuf.as<float>();

    float* const kSlot = cache.writeSlotK(blockIdx);
    float* const vSlot = cache.writeSlotV(blockIdx);

    // --- pre-attention RMSNorm ----------------------------------------

    trace("attn rmsNorm");
    _ops.rmsNormAsync(x, T, d_model,
                      static_cast<const float*>(attnNorm->usmPtr),
                      _config.rmsNormEps,
                      normBuf);
    dumpStage("attn_norm", normBuf, T, d_model);

    auto projectAsync = [&](const model::GgufTensor* W,
                            std::size_t N, float* dst) {
        _gmm.matmulAsync(W->type, W->usmPtr, N, d_model,
                         normBuf, T, dst, matmulScratch);
    };

    trace("Q projection");
    projectAsync(qW, q_dim, qBuf);

    trace("K projection");
    projectAsync(kW, kv_dim, kSlot);

    if (li.altAttention) {
        // V = raw K projection. We need a copy of K *before* K-norm and
        // RoPE so V keeps its raw layout. KvCache writes V into vSlot;
        // copy the raw K rows there now (sync first so the K matmul has
        // settled on USM).
        trace("alt-attn V = raw K (memcpy)");
        _gmm.sync();
        std::memcpy(vSlot, kSlot, T * kv_dim * sizeof(float));
    } else {
        trace("V projection");
        projectAsync(vW, kv_dim, vSlot);
    }

    // K-norm BEFORE RoPE (per-head, like Q-norm below).
    trace("K-norm");
    _ops.rmsNormAsync(kSlot, T * li.nKvHeads, head_dim,
                      static_cast<const float*>(kNorm->usmPtr),
                      _config.rmsNormEps,
                      kSlot);              // in-place

    // V-norm: bare RMSNorm over head_dim, no learned weight.
    trace("V-norm (no weight)");
    _ops.rmsNormNoWeightAsync(vSlot, T * li.nKvHeads, head_dim,
                              _config.rmsNormEps,
                              vSlot);            // in-place

    // RoPE K only (V never gets RoPE).
    trace("RoPE K");
    if (!li.isSwa && _ropeFreqsForFullAttn != nullptr) {
        _ops.ropeInPlaceWithFactorsAsync(kSlot, _ropeFreqsForFullAttn, T,
                                         li.nKvHeads, head_dim, curLen,
                                         li.ropeBase);
    } else {
        _ops.ropeInPlaceAsync(kSlot, T, li.nKvHeads, head_dim, curLen,
                              li.ropeBase);
    }
    dumpStage("Kcur_pos",    kSlot, T, kv_dim);
    dumpStage("Vcur_normed", vSlot, T, kv_dim);

    // Q-K-norm BEFORE RoPE (per-head RMSNorm over head_dim).
    trace("Q-norm");
    _ops.rmsNormAsync(qBuf, T * li.nHeads, head_dim,
                      static_cast<const float*>(qNorm->usmPtr),
                      _config.rmsNormEps,
                      qBuf);                   // in-place

    trace("RoPE Q");
    if (!li.isSwa && _ropeFreqsForFullAttn != nullptr) {
        _ops.ropeInPlaceWithFactorsAsync(qBuf, _ropeFreqsForFullAttn, T,
                                         li.nHeads, head_dim, curLen,
                                         li.ropeBase);
    } else {
        _ops.ropeInPlaceAsync(qBuf, T, li.nHeads, head_dim, curLen,
                              li.ropeBase);
    }
    dumpStage("Qcur_pos", qBuf, T, q_dim);

    // M5f.3: attention on the GPU. Gemma 4's f_attention_scale = 1.0
    // (gemma4.cpp:11), so we pass scale=1.0 directly — no sqrt(head_dim)
    // pre-scale needed anymore. The old CPU path had to pre-scale Q
    // because compute::multiHeadAttention bakes in 1/sqrt(headDim);
    // the GPU kernel takes scale as a parameter, which makes the Gemma
    // branch one mulScalarAsync + one sync cheaper than before.
    trace("attention (GPU, scale=1)");
    _ops.attentionAsync(qBuf,
                        cache.baseK(blockIdx),
                        cache.baseV(blockIdx),
                        T, totalLen,
                        li.nHeads, li.nKvHeads, head_dim,
                        curLen, /*scale=*/1.0F,
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
    dumpStage("attn_post_norm", projOutBuf, T, d_model);

    trace("attn residual");
    _ops.addResidualAsync(x, projOutBuf, T * d_model);
    // x now holds sa_out = inpL + attn_post_norm(attn_out).
    dumpStage("attn_out", x, T, d_model);

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
    dumpStage("ffn_mlp", projOutBuf, T, d_model);

    // --- Path B — MoE -------------------------------------------------

    trace("path B: pre_ffw_norm_2");
    _ops.rmsNormAsync(x, T, d_model,
                      static_cast<const float*>(preNorm2->usmPtr),
                      _config.rmsNormEps,
                      normBuf);

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

    std::vector<std::int32_t> topKIdx(T * K);
    std::vector<float>        topKWeight(T * K);
    cmp::moeTopKRoute(upOutBuf, T, nExperts, K,
                      topKIdx.data(), topKWeight.data());

    trace("path B: zero accumulator");
    _ops.mulScalarAsync(moeAccumBuf, 0.0F, T * d_model);

    const std::size_t gateUpFused = expGateUp->dimensions.size() >= 2
                                      ? expGateUp->dimensions[1] : 0;
    const std::size_t ffPerExpert = gateUpFused / 2;
    if (gateUpFused == 0 || (gateUpFused % 2) != 0) {
        throw std::runtime_error(
            "Gemma4Backend: ffn_gate_up_exps has unexpected fused dim " +
            std::to_string(gateUpFused));
    }

    const compute::QuantType* const qtGateUp = compute::quantType(expGateUp->type);
    const compute::QuantType* const qtDown   = compute::quantType(expDown->type);
    if (qtGateUp == nullptr || qtDown == nullptr) {
        throw std::runtime_error(
            "Gemma4Backend: expert weight type(s) not in QuantType registry");
    }

    const std::size_t expertBytesGateUp =
        gateUpFused * (d_model / qtGateUp->blockElements()) * qtGateUp->blockBytes();
    const std::size_t expertBytesDown =
        d_model * (ffPerExpert / qtDown->blockElements()) * qtDown->blockBytes();

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
    dumpStage("ffn_moe", moeAccumBuf, T, d_model);

    trace("combined = pathA_out + pathB_out");
    _ops.addResidualAsync(moeAccumBuf, projOutBuf, T * d_model);
    dumpStage("ffn_moe_combined", moeAccumBuf, T, d_model);

    trace("post_ffw_norm (combined)");
    _ops.rmsNormAsync(moeAccumBuf, T, d_model,
                      static_cast<const float*>(ffwPost->usmPtr),
                      _config.rmsNormEps,
                      moeAccumBuf);
    dumpStage("ffn_post_norm", moeAccumBuf, T, d_model);

    trace("ffn residual");
    _ops.addResidualAsync(x, moeAccumBuf, T * d_model);

    const float scaleVal = *static_cast<const float*>(outScale->usmPtr);
    trace("layer_output_scale");
    _ops.mulScalarAsync(x, scaleVal, T * d_model);
    dumpStage("out_scaled", x, T, d_model);
    dumpStage("l_out",      x, T, d_model);
}

} // namespace mimirmind::runtime::arch