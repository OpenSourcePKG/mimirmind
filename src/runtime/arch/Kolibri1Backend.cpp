// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/arch/Kolibri1Backend.hpp"

#include "compute/ComputeMatmul.hpp"
#include "compute/ComputeOps.hpp"
#include "core/gguf/WeightsMap.hpp"
#include "core/modelopt/BlockScaleSwizzle.hpp"
#include "core/log/Log.hpp"
#include "model/LlmConfig.hpp"
#include "runtime/BlockBuffers.hpp"
#include "runtime/KvCache.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace mimirmind::runtime::arch {

namespace {
const core::gguf::GgufTensor&
requireBlk(const core::gguf::WeightsMap& w, std::size_t blockIdx,
           std::string_view suffix) {
    const auto* t = w.findBlock(blockIdx, suffix);
    if (t == nullptr) {
        throw std::runtime_error(
            "Kolibri1Backend: block " + std::to_string(blockIdx) +
            " missing tensor '" + std::string(suffix) + "'");
    }
    return *t;
}
} // namespace

Kolibri1Backend::Kolibri1Backend(const model::LlmConfig&       config,
                                 const core::gguf::WeightsMap& weights,
                                 const model::FusedQkvWeights* fusedQkv,
                                 compute::ComputeOps&          ops,
                                 compute::ComputeMatmul&       gmm,
                                 runtime::OpProfiler&          opProfiler,
                                 bool                          moeGroupEnabled,
                                 bool                          moeFusedDownEnabled)
    : _config(config),
      _weights(weights),
      _fusedQkv(fusedQkv),
      _ops(ops),
      _gmm(gmm),
      _opProfiler(opProfiler),
      _moeGroupEnabled(moeGroupEnabled),
      _moeFusedDownEnabled(moeFusedDownEnabled) {}

void Kolibri1Backend::runBlock(std::size_t   blockIdx,
                               float*        x,
                               std::size_t   T,
                               KvCache&      cache,
                               BlockBuffers& buffers,
                               bool          traceBlock0) {
    const std::size_t L       = blockIdx;
    const std::size_t d_model = _config.embeddingLength;
    const std::size_t headDim = _config.headDim(L);
    const std::size_t nHeads  = _config.headCount;
    const std::size_t nKvHeads = _config.headCountKvFor(L);
    const std::size_t qDim    = nHeads * headDim;
    const std::size_t kvDim   = nKvHeads * headDim;
    const float       eps     = _config.rmsNormEps;
    const std::size_t curLen  = cache.length();
    const std::size_t totalLen = curLen + T;
    const bool isSliding = (L < _config.slidingWindowPattern.size())
                           && static_cast<bool>(_config.slidingWindowPattern[L]);
    const bool diag = (blockIdx == 0 && curLen == 0 && traceBlock0);
    if (diag) {
        MM_LOG_INFO("blkdiag-k",
                    "blk0 kolibri1 T={} d_model={} qDim={} kvDim={} {} (window={})",
                    T, d_model, qDim, kvDim,
                    isSliding ? "sliding" : "full/NoPE",
                    isSliding ? _config.slidingWindow : 0);
    }

    // --- weights ---------------------------------------------------------
    const auto& attnNorm = requireBlk(_weights, L, "attn_norm.weight");
    const auto& qW       = requireBlk(_weights, L, "attn_q.weight");
    const auto& kW       = requireBlk(_weights, L, "attn_k.weight");
    const auto& vW       = requireBlk(_weights, L, "attn_v.weight");
    const auto& oW       = requireBlk(_weights, L, "attn_output.weight");
    const auto& qNorm    = requireBlk(_weights, L, "attn_q_norm.weight");
    const auto& kNorm    = requireBlk(_weights, L, "attn_k_norm.weight");
    const auto& postAttn = requireBlk(_weights, L, "post_attn_norm.weight");
    const auto& preFfn   = requireBlk(_weights, L, "post_attention_norm.weight");
    const auto& postFfn  = requireBlk(_weights, L, "post_ffn_norm.weight");

    float* const normBuf  = buffers.normBuf.as<float>();
    float* const qBuf     = buffers.qBuf.as<float>();
    float* const attnOut  = buffers.attnOut.as<float>();
    float* const projOut  = buffers.projOut.as<float>();
    float* const scratch  = buffers.matmulScratch.as<float>();

    // --- attention sublayer ---------------------------------------------
    // a = RMS(x) * w_input  (plain RMSNorm — NOT the Gemma (1+w) variant)
    _ops.rmsNormAsync(x, T, d_model,
                      static_cast<const float*>(attnNorm.usmPtr), eps, normBuf);
    dumpStage("attn_norm", L, normBuf, T, d_model);

    // Q/K/V projections (FP8/BF16 weights dispatch on GgmlType). K/V go
    // straight into this layer's cache slot at row curLen (F32-KV path);
    // Q into the fresh qBuf.
    float* const kSlot = cache.writeSlotKf32(L);
    float* const vSlot = cache.writeSlotVf32(L);
    {
        compute::UnorderedScope u{_ops};
        _gmm.matmulAsync(qW.type, qW.usmPtr, qDim,  d_model, normBuf, T, qBuf,  scratch);
        _gmm.matmulAsync(kW.type, kW.usmPtr, kvDim, d_model, normBuf, T, kSlot, scratch);
        _gmm.matmulAsync(vW.type, vW.usmPtr, kvDim, d_model, normBuf, T, vSlot, scratch);
    }

    // Per-head QK-RMSNorm (V passes through un-normed), writing the normed
    // K/V back into the cache at row curLen. Q normed in place in qBuf.
    _ops.rmsNormQkvAsync(qBuf, static_cast<const float*>(qNorm.usmPtr),
                         const_cast<void*>(cache.baseK(L)),
                         static_cast<const float*>(kNorm.usmPtr),
                         const_cast<void*>(cache.baseV(L)),
                         T * nHeads, T * nKvHeads, headDim, eps,
                         /*writeOffset=*/curLen, kvDim,
                         KvDtype::F32, /*useStagingSlot=*/false);

    // RoPE — ONLY on sliding layers (full-attention layers are NoPE).
    if (isSliding) {
        const float ropeBase = _config.ropeFreqBaseSwa;
        compute::UnorderedScope u{_ops};
        _ops.ropeInPlaceAsync(qBuf, T, nHeads, headDim, curLen, ropeBase);
        _ops.ropeInPlaceAsync(const_cast<void*>(cache.baseK(L)), T, nKvHeads,
                              headDim, curLen, ropeBase,
                              /*writeOffsetStride=*/kvDim, KvDtype::F32);
    }
    if (diag) {
        dumpStage("q_pos", L, qBuf, T, qDim);
    }

    // Attention: full-causal on full layers (window 0), sliding-513 on SWA.
    const std::size_t window = isSliding ? _config.slidingWindow : 0;
    const float       scale  = _config.attentionScaleFor(headDim);
    _ops.attentionAsync(qBuf, cache.baseK(L), cache.baseV(L),
                        /*T_q=*/T, /*T_k=*/totalLen, nHeads, nKvHeads, headDim,
                        /*positionOffset=*/curLen, scale, attnOut,
                        window, cache.dtype());

    // Output projection, post-attn norm, residual add.
    _gmm.matmulAsync(oW.type, oW.usmPtr, d_model, qDim, attnOut, T, projOut, scratch);
    _ops.rmsNormAsync(projOut, T, d_model,
                      static_cast<const float*>(postAttn.usmPtr), eps, projOut);
    dumpStage("attn_out", L, projOut, T, d_model);
    _ops.addResidualAsync(x, projOut, T * d_model);

    // --- MoE FFN sublayer ------------------------------------------------
    // d = RMS(x) * w_post_attention  (pre-MLP norm)
    _ops.rmsNormAsync(x, T, d_model,
                      static_cast<const float*>(preFfn.usmPtr), eps, normBuf);
    runKolibriMoe(L, normBuf, T, buffers);
    float* const moeAccum = buffers.moeAccumBuf.as<float>();
    // f = RMS(moe) * w_post_ffn  (post-MLP norm), then residual add.
    _ops.rmsNormAsync(moeAccum, T, d_model,
                      static_cast<const float*>(postFfn.usmPtr), eps, moeAccum);
    dumpStage("ffn_out", L, moeAccum, T, d_model);
    _ops.addResidualAsync(x, moeAccum, T * d_model);
    dumpStage("l_out", L, x, T, d_model);
}

void Kolibri1Backend::runKolibriMoe(std::size_t blockIdx, const float* din,
                                    std::size_t T, BlockBuffers& buffers) {
    const std::size_t d_model   = _config.embeddingLength;
    const std::size_t nExperts  = _config.expertCount;          // 384
    const std::size_t K         = _config.expertUsedCount;      // 6
    const std::size_t nFfExp    = _config.expertFeedForwardLength;        // 512
    const std::size_t nFfShexp  = _config.expertSharedFeedForwardLength;  // 512

    const auto& routerW = requireBlk(_weights, blockIdx, "ffn_gate_inp.weight");
    const auto& biasT   = requireBlk(_weights, blockIdx, "exp_probs_b.weight");
    const auto& gExps   = requireBlk(_weights, blockIdx, "ffn_gate_exps.weight");
    const auto& uExps   = requireBlk(_weights, blockIdx, "ffn_up_exps.weight");
    const auto& dExps   = requireBlk(_weights, blockIdx, "ffn_down_exps.weight");

    float* const gateOut   = buffers.gateOut.as<float>();
    float* const upOut     = buffers.upOut.as<float>();
    float* const expertOut = buffers.expertOutBuf.as<float>();
    float* const moeAccum  = buffers.moeAccumBuf.as<float>();
    float* const scratch   = buffers.matmulScratch.as<float>();

    // --- router: logits = din @ gate^T  [T, nExperts] --------------------
    // Reuse gateOut as the logits buffer (freed before the expert loop below;
    // [maxT, ff] >= [T, nExperts] since nExperts <= ff capacity).
    float* const logits = gateOut;
    _gmm.matmul(routerW.type, routerW.usmPtr, nExperts, d_model, din, T, logits,
                scratch);  // synchronous — host reads logits next

    // Sigmoid top-6 with DeepSeek-V3-style bias: SELECT on (logits + bias),
    // WEIGHT by sigmoid(unbiased logit), NO renormalisation (norm_topk_prob
    // false). Bias affects selection only.
    const float* const bias = static_cast<const float*>(biasT.usmPtr);
    _selIdx.assign(T * K, 0);
    _selW.assign(T * K, 0.0F);
    std::vector<char> chosen(nExperts);
    for (std::size_t t = 0; t < T; ++t) {
        const float* lg = logits + t * nExperts;
        std::fill(chosen.begin(), chosen.end(), char{0});
        for (std::size_t k = 0; k < K; ++k) {
            int   best      = -1;
            float bestScore = -std::numeric_limits<float>::infinity();
            for (std::size_t e = 0; e < nExperts; ++e) {
                if (chosen[e]) continue;
                const float sc = lg[e] + bias[e];
                if (sc > bestScore) { bestScore = sc; best = static_cast<int>(e); }
            }
            chosen[static_cast<std::size_t>(best)] = 1;
            _selIdx[t * K + k] = best;
            _selW[t * K + k]   = 1.0F / (1.0F + std::exp(-lg[static_cast<std::size_t>(best)]));
        }
    }

    _ops.mulScalarAsync(moeAccum, 0.0F, T * d_model);

    if (gExps.type == core::gguf::GgmlType::NVFP4_TC) {
        // --- routed experts, FP4-tensor-core fused-K (I-9 / 5.34.10) --------
        // One launch each for gate+up and down over all T rows x K experts,
        // instead of the T*K*3 per-token GEMVs below. The host sigmoid-top6
        // selection (above) is kept (Kolibri's select-on-bias routing is
        // bespoke); its indices/weights ride the per-token USM slots the
        // batched TC kernels read on the stream. moeGateCompact holds the
        // [T*K, nFfExp] silu(gate)*up activation the down kernel consumes.
        namespace mo = core::modelopt;
        float* const gateActAll = buffers.moeGateCompact.as<float>();
        auto* const  expIdx     = buffers.moePrefillExpIdx.as<std::int32_t>();
        auto* const  kwDev      = buffers.moePrefillKw.as<float>();
        for (std::size_t i = 0; i < T * K; ++i) {
            expIdx[i] = _selIdx[i];
            kwDev[i]  = _selW[i];
        }
        const std::size_t sfbGu = mo::moeSwizzledScaleStride(nFfExp, d_model / 16);
        const std::size_t sfbDn = mo::moeSwizzledScaleStride(d_model, nFfExp / 16);
        _gmm.moeGateUpFusedKTcBatchedAsync(
            din, gExps.tcNibblePtr, uExps.tcNibblePtr,
            gExps.tcSfbPtr, uExps.tcSfbPtr,
            static_cast<const float*>(gExps.tcGlobalsPtr),
            static_cast<const float*>(uExps.tcGlobalsPtr),
            expIdx, gateActAll, T, d_model, nFfExp, K, sfbGu);
        _gmm.moeDownFusedKTcBatchedAsync(
            gateActAll, dExps.tcNibblePtr, dExps.tcSfbPtr,
            static_cast<const float*>(dExps.tcGlobalsPtr),
            expIdx, kwDev, moeAccum, T, nFfExp, d_model, K, sfbDn);
    } else {
        // --- routed experts (sequential per-token FP8 GEMV) ----------------
        // Per-expert byte strides for the stacked FP8_E4M3 32-block banks
        // (34 B / 32 elems). gate/up: [nFfExp, d_model]; down: [d_model, nFfExp].
        constexpr std::size_t kBlkElems = 32, kBlkBytes = 34;
        const std::size_t bytesGU   = (nFfExp * d_model / kBlkElems) * kBlkBytes;
        const std::size_t bytesDown = (d_model * nFfExp / kBlkElems) * kBlkBytes;
        const auto* const gateBase = static_cast<const std::uint8_t*>(gExps.usmPtr);
        const auto* const upBase   = static_cast<const std::uint8_t*>(uExps.usmPtr);
        const auto* const downBase = static_cast<const std::uint8_t*>(dExps.usmPtr);

        for (std::size_t t = 0; t < T; ++t) {
            const float* const xt     = din      + t * d_model;
            float* const       accumT = moeAccum + t * d_model;
            for (std::size_t k = 0; k < K; ++k) {
                const std::size_t e  = static_cast<std::size_t>(_selIdx[t * K + k]);
                const float       wk = _selW[t * K + k];
                const void* Wg = gateBase + e * bytesGU;
                const void* Wu = upBase   + e * bytesGU;
                const void* Wd = downBase + e * bytesDown;
                _gmm.matmulAsync(gExps.type, Wg, nFfExp, d_model, xt, 1, gateOut, scratch);
                _gmm.matmulAsync(uExps.type, Wu, nFfExp, d_model, xt, 1, upOut,   scratch);
                _ops.siluMulAsync(gateOut, upOut, nFfExp);          // silu(gate)*up
                _gmm.matmulAsync(dExps.type, Wd, d_model, nFfExp, gateOut, 1, expertOut, scratch);
                _ops.scaledAddResidualAsync(accumT, expertOut, wk, d_model);
            }
        }
    }

    // --- always-on shared expert (ungated SwiGLU), added to the accum -----
    const auto& gSh = requireBlk(_weights, blockIdx, "ffn_gate_shexp.weight");
    const auto& uSh = requireBlk(_weights, blockIdx, "ffn_up_shexp.weight");
    const auto& dSh = requireBlk(_weights, blockIdx, "ffn_down_shexp.weight");
    _gmm.matmulAsync(gSh.type, gSh.usmPtr, nFfShexp, d_model, din, T, gateOut, scratch);
    _gmm.matmulAsync(uSh.type, uSh.usmPtr, nFfShexp, d_model, din, T, upOut,   scratch);
    _ops.siluMulAsync(gateOut, upOut, T * nFfShexp);
    _gmm.matmulAsync(dSh.type, dSh.usmPtr, d_model, nFfShexp, gateOut, T, expertOut, scratch);
    _ops.addResidualAsync(moeAccum, expertOut, T * d_model);
}

void Kolibri1Backend::dumpStage(const char* stage, std::size_t blockIdx,
                                const float* p, std::size_t Trow,
                                std::size_t dim) {
    if (_parityDumpPrefix.empty()) {
        return;
    }
    _gmm.sync();  // the buffer is managed USM; make the device writes visible
    const std::string path = _parityDumpPrefix + "-blk" + std::to_string(blockIdx)
                             + "-" + stage + ".bin";
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        return;
    }
    const std::uint32_t hdr[3] = {static_cast<std::uint32_t>(blockIdx),
                                  static_cast<std::uint32_t>(Trow),
                                  static_cast<std::uint32_t>(dim)};
    f.write(reinterpret_cast<const char*>(hdr), sizeof(hdr));
    f.write(reinterpret_cast<const char*>(p),
            static_cast<std::streamsize>(Trow * dim * sizeof(float)));
}

std::vector<std::size_t> Kolibri1Backend::kvDimPerLayer() const {
    std::vector<std::size_t> dims;
    dims.reserve(_config.blockCount);
    for (std::size_t l = 0; l < _config.blockCount; ++l) {
        dims.push_back(static_cast<std::size_t>(_config.headCountKvFor(l))
                       * _config.headDim(l));
    }
    return dims;
}

std::pair<std::size_t, std::size_t> Kolibri1Backend::maxQKVDims() const {
    const std::size_t qDim =
        static_cast<std::size_t>(_config.headCount) * _config.headDim();
    std::size_t kvDimMax = 0;
    for (std::size_t l = 0; l < _config.blockCount; ++l) {
        kvDimMax = std::max(kvDimMax,
                            static_cast<std::size_t>(_config.headCountKvFor(l))
                                * _config.headDim(l));
    }
    return {qDim, kvDimMax};
}

ArchCapabilities Kolibri1Backend::computeCapabilities() const {
    // Pure-attention MoE. Batched decode / CLR-safe MoE decode are not wired at
    // the I-4 forward (they turn on as the serving paths land, I-6+). The
    // ArchCapabilities defaults (everything false except decodeQkvClrSafe) are
    // the correct conservative starting point.
    return ArchCapabilities{};
}

} // namespace mimirmind::runtime::arch
