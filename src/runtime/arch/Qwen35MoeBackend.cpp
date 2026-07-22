// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/arch/Qwen35MoeBackend.hpp"

#include "compute/ComputeMatmul.hpp"
#include "compute/ComputeOps.hpp"
#include "compute/MoeRouting.hpp"
#include "compute/QuantType.hpp"
#include "compute/QuantTypeRegistry.hpp"
#include "core/gguf/GgufTypes.hpp"
#include "core/gguf/WeightsMap.hpp"
#include "core/log/Log.hpp"
#include "model/LlmConfig.hpp"
#include "runtime/BlockBuffers.hpp"
#include "runtime/KvCache.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace mimirmind::runtime::arch {

namespace {

const core::gguf::GgufTensor&
requireBlock(const core::gguf::WeightsMap& w, std::size_t blockIdx,
             std::string_view suffix) {
    const auto* t = w.findBlock(blockIdx, suffix);
    if (t == nullptr) {
        throw std::runtime_error(
            "Qwen35MoeBackend: block " + std::to_string(blockIdx) +
            " missing tensor '" + std::string(suffix) + "'");
    }
    return *t;
}

} // namespace

Qwen35MoeBackend::Qwen35MoeBackend(const model::LlmConfig&       config,
                                   const core::gguf::WeightsMap& weights,
                                   const model::FusedQkvWeights* fusedQkv,
                                   compute::ComputeOps&          ops,
                                   compute::ComputeMatmul&       gmm,
                                   runtime::OpProfiler&          opProfiler,
                                   bool                          moeGroupEnabled,
                                   bool                          moeFusedDownEnabled)
    : _config{config}, _weights{weights}, _fusedQkv{fusedQkv},
      _ops{ops}, _gmm{gmm}, _op{opProfiler},
      _moeGroupEnabled{moeGroupEnabled},
      _moeFusedDownEnabled{moeFusedDownEnabled} {
    _ssmTrace = (std::getenv("MIMIRMIND_SSM_TRACE") != nullptr);
    for (std::size_t i = 0; i < 4; ++i) {
        _ropeSections[i] = i < _config.ropeSections.size()
                               ? _config.ropeSections[i]
                               : 0;
    }
    std::size_t recurrent = 0;
    for (std::size_t b = 0; b < _config.blockCount; ++b) {
        if (_config.isRecurrentLayer(b)) ++recurrent;
    }
    MM_LOG_INFO("qwen35moe",
                "Qwen35MoeBackend ready — blocks={} ({} full / {} linear) "
                "d_model={} heads={} kv={} head_dim={} experts={}/{} "
                "ff_exp={} ff_shexp={} sections=[{},{},{},{}]",
                _config.blockCount, _config.blockCount - recurrent, recurrent,
                _config.embeddingLength, _config.headCount, _config.headCountKv,
                _config.headDim(), _config.expertCount, _config.expertUsedCount,
                _config.expertFeedForwardLength,
                _config.expertSharedFeedForwardLength,
                _ropeSections[0], _ropeSections[1],
                _ropeSections[2], _ropeSections[3]);
}

std::vector<std::size_t> Qwen35MoeBackend::kvDimPerLayer() const {
    // Full-attention layers own a KV cache of nKvHeads*headDim. The
    // recurrent (GatedDeltaNet) layers keep an SSM state instead of KV;
    // M-Q3N.2 sizes them the same (unused) and M-Q3N.3 replaces that with
    // a dedicated SSM state pool + zero KV.
    const std::size_t kvDim = _config.headCountKv * _config.headDim();
    return std::vector<std::size_t>(_config.blockCount, kvDim);
}

std::pair<std::size_t, std::size_t> Qwen35MoeBackend::maxQKVDims() const {
    const std::size_t qDim  = _config.headCount   * _config.headDim();
    const std::size_t kvDim = _config.headCountKv * _config.headDim();
    return {qDim, kvDim};
}

bool Qwen35MoeBackend::needsSsmScratch() const noexcept {
    return _config.isHybridRecurrent();
}

void Qwen35MoeBackend::traceNorm(const char* tag, std::size_t blockIdx,
                                 std::size_t pos, const float* p,
                                 std::size_t n) const {
    _gmm.sync();  // p is a unified-memory pointer; readable after sync.
    double sumSq = 0.0;
    float  maxAbs = 0.0F;
    for (std::size_t i = 0; i < n; ++i) {
        const float v = p[i];
        sumSq += static_cast<double>(v) * static_cast<double>(v);
        maxAbs = std::fmax(maxAbs, std::fabs(v));
    }
    MM_LOG_INFO("ssm-trace", "pos={} blk={} {} l2={:.5g} max={:.5g}",
                pos, blockIdx, tag, std::sqrt(sumSq), maxAbs);
}

void Qwen35MoeBackend::runBlock(std::size_t   blockIdx,
                                float*        x,
                                std::size_t   T,
                                KvCache&      cache,
                                BlockBuffers& s,
                                bool          traceBlock0) {
    const bool diag = (blockIdx == 0 && cache.length() == 0 && traceBlock0);

    if (_config.isRecurrentLayer(blockIdx)) {
        runLinearBlock(blockIdx, x, T, cache, s, diag);
    } else {
        runFullAttentionBlock(blockIdx, x, T, cache, s, diag);
    }

    if (_ssmTrace) {
        const std::size_t pos = cache.length() + (T > 0 ? T - 1 : 0);
        const char* kind = _config.isRecurrentLayer(blockIdx) ? "xout(lin)"
                                                              : "xout(full)";
        traceNorm(kind, blockIdx, pos, x, T * s.d_model);
    }
}

void Qwen35MoeBackend::runFullAttentionBlock(std::size_t   blockIdx,
                                             float*        x,
                                             std::size_t   T,
                                             KvCache&      cache,
                                             BlockBuffers& s,
                                             bool          diag) {
    auto trace = [&](const char* tag) {
        if (diag) MM_LOG_INFO("blkdiag-q35", "blk {} {}", blockIdx, tag);
    };
    trace("enter (full-attn)");

    const auto kvDtype = cache.dtype();
    if (kvDtype != KvDtype::F32) {
        throw std::runtime_error(
            "Qwen35MoeBackend: only KvDtype::F32 is supported "
            "(M-Q3N.2 F32-only IMRoPE + staging path)");
    }

    const auto& w    = _weights;
    const auto& attnNorm = requireBlock(w, blockIdx, "attn_norm.weight");
    const auto& qW       = requireBlock(w, blockIdx, "attn_q.weight");
    const auto& kW       = requireBlock(w, blockIdx, "attn_k.weight");
    const auto& vW       = requireBlock(w, blockIdx, "attn_v.weight");
    const auto& qNorm    = requireBlock(w, blockIdx, "attn_q_norm.weight");
    const auto& kNorm    = requireBlock(w, blockIdx, "attn_k_norm.weight");
    const auto& oW       = requireBlock(w, blockIdx, "attn_output.weight");
    const auto& attnPost = requireBlock(w, blockIdx, "post_attention_norm.weight");

    const std::size_t d_model  = s.d_model;
    const std::size_t head_dim = _config.headDim();
    const std::size_t nHeads   = _config.headCount;
    const std::size_t nKvHeads = _config.headCountKv;
    const std::size_t q_dim    = nHeads   * head_dim;
    const std::size_t kv_dim   = nKvHeads * head_dim;
    const std::size_t curLen   = cache.length();
    const std::size_t totalLen = curLen + T;

    float* const normBuf       = s.normBuf.as<float>();
    float* const qBuf          = s.qBuf.as<float>();
    float* const gateBuf       = s.gateScratch.as<float>();
    float* const qGateFused    = s.qGateFused.as<float>();
    float* const attnOutBuf    = s.attnOut.as<float>();
    float* const projOutBuf    = s.projOut.as<float>();
    float* const matmulScratch = s.matmulScratch.as<float>();

    void* const kSlot = cache.writeSlotK(blockIdx);
    void* const vSlot = cache.writeSlotV(blockIdx);
    void* const kBase = const_cast<void*>(cache.baseK(blockIdx));
    void* const vBase = const_cast<void*>(cache.baseV(blockIdx));

    // --- pre-attention RMSNorm ---------------------------------------
    trace("attn rmsNorm");
    _ops.rmsNormAsync(x, T, d_model,
                      static_cast<const float*>(attnNorm.usmPtr),
                      _config.rmsNormEps, normBuf);

    // --- Q(+gate) / K / V projections --------------------------------
    // attn_q fuses query + per-head output gate: output width is 2*q_dim
    // laid out [Q_h | gate_h] per head. splitHeadPair de-interleaves it
    // into qBuf (Q, roped + attended) and gateBuf (raw gate, applied as a
    // sigmoid after attention).
    trace("Q|gate / K / V projections");
    {
        compute::UnorderedScope u{_ops};
        _gmm.matmulAsync(qW.type, qW.usmPtr, 2 * q_dim, d_model,
                         normBuf, T, qGateFused, matmulScratch);
        _gmm.matmulAsync(kW.type, kW.usmPtr, kv_dim, d_model,
                         normBuf, T, static_cast<float*>(kSlot), matmulScratch);
        _gmm.matmulAsync(vW.type, vW.usmPtr, kv_dim, d_model,
                         normBuf, T, static_cast<float*>(vSlot), matmulScratch);
    }

    trace("split Q|gate");
    _ops.splitHeadPairAsync(qGateFused, qBuf, gateBuf, T, nHeads, head_dim);

    // --- QK-norm (per-head RMS over head_dim) + V passthrough --------
    trace("QK-norm");
    _ops.rmsNormQkvAsync(
        qBuf,  static_cast<const float*>(qNorm.usmPtr),
        kBase, static_cast<const float*>(kNorm.usmPtr),
        vBase,
        T * nHeads, T * nKvHeads, head_dim,
        _config.rmsNormEps,
        /*writeOffset=*/curLen, kv_dim,
        kvDtype, /*useStagingSlot=*/false);

    // --- IMRoPE on Q and K -------------------------------------------
    trace("IMRoPE Q+K");
    {
        compute::UnorderedScope u{_ops};
        _ops.mropeInPlaceAsync(qBuf, T, nHeads, head_dim, curLen,
                               _config.ropeFreqBase, _ropeSections);
        _ops.mropeInPlaceAsync(kBase, T, nKvHeads, head_dim, curLen,
                               _config.ropeFreqBase, _ropeSections,
                               /*writeOffsetStride=*/kv_dim, kvDtype);
    }

    // --- GQA attention -----------------------------------------------
    trace("attention");
    const float attnScale = _config.attentionScaleFor(head_dim);
    _ops.attentionAsync(qBuf, cache.baseK(blockIdx), cache.baseV(blockIdx),
                        T, totalLen, nHeads, nKvHeads, head_dim,
                        curLen, attnScale, attnOutBuf,
                        /*slidingWindow=*/0, kvDtype);

    // --- per-head output gate: attn *= sigmoid(gate) -----------------
    trace("output sigmoid gate");
    _ops.sigmoidGateMulAsync(attnOutBuf, gateBuf, T, q_dim, /*gateDim=*/q_dim);

    // --- O projection + attn residual --------------------------------
    trace("O projection");
    _gmm.matmulAsync(oW.type, oW.usmPtr, d_model, q_dim,
                     attnOutBuf, T, projOutBuf, matmulScratch);

    trace("attn residual");
    _ops.addResidualAsync(x, projOutBuf, T * d_model);   // x = x + attn_out

    // --- post-attention norm -> MoE FFN -> FFN residual --------------
    trace("post_attention_norm");
    _ops.rmsNormAsync(x, T, d_model,
                      static_cast<const float*>(attnPost.usmPtr),
                      _config.rmsNormEps, normBuf);

    trace("MoE FFN");
    runMoeFfn(blockIdx, normBuf, T, s);

    trace("ffn residual");
    _ops.addResidualAsync(x, s.moeAccumBuf.as<float>(), T * d_model);
}

void Qwen35MoeBackend::runLinearBlock(std::size_t   blockIdx,
                                      float*        x,
                                      std::size_t   T,
                                      KvCache&      cache,
                                      BlockBuffers& s,
                                      bool          diag) {
    auto trace = [&](const char* tag) {
        if (diag) MM_LOG_INFO("blkdiag-q35", "blk {} lin {}", blockIdx, tag);
    };
    trace("enter (linear/GatedDeltaNet)");

    const auto& w = _weights;
    const auto& attnNorm  = requireBlock(w, blockIdx, "attn_norm.weight");
    const auto& qkvW      = requireBlock(w, blockIdx, "attn_qkv.weight");
    const auto& gateW     = requireBlock(w, blockIdx, "attn_gate.weight");
    const auto& betaW     = requireBlock(w, blockIdx, "ssm_beta.weight");
    const auto& alphaW    = requireBlock(w, blockIdx, "ssm_alpha.weight");
    const auto& ssmA      = requireBlock(w, blockIdx, "ssm_a");
    const auto& ssmDt     = requireBlock(w, blockIdx, "ssm_dt.bias");
    const auto& convW     = requireBlock(w, blockIdx, "ssm_conv1d.weight");
    const auto& ssmNormW  = requireBlock(w, blockIdx, "ssm_norm.weight");
    const auto& ssmOutW   = requireBlock(w, blockIdx, "ssm_out.weight");
    const auto& attnPost  = requireBlock(w, blockIdx, "post_attention_norm.weight");

    const std::size_t d_model   = s.d_model;
    const std::size_t S         = _config.ssmStateSize;      // head_dim
    const std::size_t hK        = _config.ssmNumKHeads();
    const std::size_t hV        = _config.ssmNumVHeads();
    const std::size_t valueDim  = _config.ssmInnerSize;      // = hV * S
    const std::size_t convDim    = _config.ssmConvDim();
    const std::size_t dConv     = _config.ssmConvKernel;
    const std::size_t keyDim    = S * hK;
    const std::size_t stateElems     = _config.ssmStateElemsPerLayer();
    const std::size_t convStateElems = _config.ssmConvStateElemsPerLayer();
    const float       eps       = _config.rmsNormEps;

    // Persistent per-layer recurrent state (survives across decode steps);
    // zeroed only at sequence start (curLen == 0).
    const bool isSeqStart = (cache.length() == 0);

    float* const normBuf   = s.normBuf.as<float>();
    float* const qkvMixed  = s.ssmQkvMixed.as<float>();
    float* const convInput = s.ssmConvInput.as<float>();
    float* const zBuf      = s.ssmZ.as<float>();
    float* const qBuf      = s.ssmQ.as<float>();
    float* const kBuf      = s.ssmK.as<float>();
    float* const vBuf      = s.ssmV.as<float>();
    float* const deltaOut  = s.ssmDeltaOut.as<float>();
    float* const alphaBuf  = s.ssmAlpha.as<float>();
    float* const betaBuf   = s.ssmBeta.as<float>();
    float* const gateBuf   = s.ssmGate.as<float>();
    // Persistent per-sequence recurrent state, bound by the engine into
    // BlockBuffers from the per-sequence SsmState (survives BlockBuffers
    // reallocation); indexed by blockIdx.
    float* const stateBuf  = s.ssmStatePtr     + blockIdx * stateElems;
    float* const convState = s.ssmConvStatePtr + blockIdx * convStateElems;
    float* const projOut   = s.projOut.as<float>();
    float* const matmulScr = s.matmulScratch.as<float>();

    // --- pre-attention RMSNorm ---------------------------------------
    trace("attn rmsNorm");
    _ops.rmsNormAsync(x, T, d_model,
                      static_cast<const float*>(attnNorm.usmPtr), eps, normBuf);

    // --- projections (all read normBuf, disjoint outputs) ------------
    trace("qkv / gate / beta / alpha projections");
    {
        compute::UnorderedScope u{_ops};
        _gmm.matmulAsync(qkvW.type,   qkvW.usmPtr,   convDim,  d_model, normBuf, T, qkvMixed, matmulScr);
        _gmm.matmulAsync(gateW.type,  gateW.usmPtr,  valueDim, d_model, normBuf, T, zBuf,     matmulScr);
        _gmm.matmulAsync(betaW.type,  betaW.usmPtr,  hV,       d_model, normBuf, T, betaBuf,  matmulScr);
        _gmm.matmulAsync(alphaW.type, alphaW.usmPtr, hV,       d_model, normBuf, T, alphaBuf, matmulScr);
    }

    // beta = sigmoid(beta); gLog = -exp(ssm_a) * softplus(alpha + ssm_dt).
    trace("beta sigmoid + decay gate");
    _ops.sigmoidInPlaceAsync(betaBuf, T * hV);
    _ops.deltanetGateAsync(alphaBuf,
                           static_cast<const float*>(ssmA.usmPtr),
                           static_cast<const float*>(ssmDt.usmPtr),
                           gateBuf, T, hV);

    // --- causal conv1d + silu ----------------------------------------
    // conv_input = concat(conv_state[d_conv-1], qkv_mixed[T]) along time.
    // conv_state persists across decode steps (rolling tail); it is zeroed
    // only at sequence start. After the conv, the last (d_conv-1) rows of
    // conv_input become the new conv_state. conv output reuses qkvMixed.
    trace("conv1d (rolling state-concat + silu)");
    const std::size_t stateRows = (dConv > 0 ? dConv - 1 : 0);
    if (isSeqStart) {
        _ops.mulScalarAsync(convState, 0.0F, convStateElems);
    }
    // conv_input = [conv_state | qkv_mixed]
    _ops.appendMemoryCopy(convInput, convState,
                          convStateElems * sizeof(float));
    _ops.appendMemoryCopy(convInput + stateRows * convDim, qkvMixed,
                          T * convDim * sizeof(float));
    _ops.causalConv1dSiluAsync(convInput,
                               static_cast<const float*>(convW.usmPtr),
                               qkvMixed, T, convDim, dConv);      // conv out -> qkvMixed
    // Save the trailing (d_conv-1) rows as the next conv_state.
    _ops.appendMemoryCopy(convState, convInput + T * convDim,
                          convStateElems * sizeof(float));

    // --- split conv into q / k / v (+ GQA repeat H_k -> H_v) ---------
    trace("gather q/k/v");
    _ops.gatherHeadsFromChannelsAsync(qkvMixed, qBuf, T, 0,          hK, hV, S, convDim);
    _ops.gatherHeadsFromChannelsAsync(qkvMixed, kBuf, T, keyDim,     hK, hV, S, convDim);
    _ops.gatherHeadsFromChannelsAsync(qkvMixed, vBuf, T, 2 * keyDim, hV, hV, S, convDim);

    // --- L2-norm q, k over head_dim ----------------------------------
    trace("L2-norm q,k");
    _ops.l2NormInPlaceAsync(qBuf, T * hV, S, eps);
    _ops.l2NormInPlaceAsync(kBuf, T * hV, S, eps);

    // --- gated delta-rule recurrence (persistent state) -------------
    // state zeroed only at sequence start; decode steps evolve it in place.
    trace("delta-rule recurrence");
    if (isSeqStart) {
        _ops.mulScalarAsync(stateBuf, 0.0F, stateElems);
    }
    _ops.gatedDeltaNetRecurrentAsync(qBuf, kBuf, vBuf, gateBuf, betaBuf,
                                     stateBuf, deltaOut, T, hV, S);

    if (_ssmTrace) {
        // For T>1 (prefill) the state reflects the last token; for decode
        // (T==1) cache.length() is the position of the token just added.
        const std::size_t pos = cache.length() + (T > 0 ? T - 1 : 0);
        traceNorm("gate",  blockIdx, pos, gateBuf, T * hV);
        traceNorm("state", blockIdx, pos, stateBuf, stateElems);
        traceNorm("dnet",  blockIdx, pos, deltaOut, T * valueDim);
    }

    // --- gated output norm: ssm_norm(out) * silu(z) ------------------
    // rmsNorm(out) over head_dim -> qBuf (reused as norm buffer), then
    // siluMul(z, n) = silu(z) * n, in place into zBuf.
    trace("gated ssm_norm x silu(z)");
    _ops.rmsNormAsync(deltaOut, T * hV, S,
                      static_cast<const float*>(ssmNormW.usmPtr), eps, qBuf);
    _ops.siluMulAsync(zBuf, qBuf, T * valueDim);

    // --- output projection ssm_out -----------------------------------
    trace("ssm_out projection");
    _gmm.matmulAsync(ssmOutW.type, ssmOutW.usmPtr, d_model, valueDim,
                     zBuf, T, projOut, matmulScr);

    // --- attn residual + post-attn-norm -> MoE FFN -> FFN residual ---
    trace("attn residual");
    _ops.addResidualAsync(x, projOut, T * d_model);

    trace("post_attention_norm");
    _ops.rmsNormAsync(x, T, d_model,
                      static_cast<const float*>(attnPost.usmPtr), eps, normBuf);

    trace("MoE FFN");
    runMoeFfn(blockIdx, normBuf, T, s);

    trace("ffn residual");
    _ops.addResidualAsync(x, s.moeAccumBuf.as<float>(), T * d_model);
}

void Qwen35MoeBackend::runMoeFfn(std::size_t   blockIdx,
                                 const float*  moeInput,
                                 std::size_t   T,
                                 BlockBuffers& s) {
    namespace cmp = mimirmind::compute;
    const auto& w = _weights;

    const auto& routerW  = requireBlock(w, blockIdx, "ffn_gate_inp.weight");
    const auto& downExps = requireBlock(w, blockIdx, "ffn_down_exps.weight");

    // Routed experts ship EITHER a fused `ffn_gate_up_exps`
    // [n_embd, 2*n_ff, n_expert] OR separate `ffn_gate_exps` +
    // `ffn_up_exps` [n_embd, n_ff, n_expert] (llama.cpp
    // create_tensor_gate_up_exps). Support both; the recon target GGUF
    // uses the separate layout.
    const auto* gateUpFused = w.findBlock(blockIdx, "ffn_gate_up_exps.weight");
    const bool  fused       = (gateUpFused != nullptr);
    const auto* gateExpsP   = fused ? nullptr
                                    : &requireBlock(w, blockIdx, "ffn_gate_exps.weight");
    const auto* upExpsP     = fused ? nullptr
                                    : &requireBlock(w, blockIdx, "ffn_up_exps.weight");

    const std::size_t d_model  = s.d_model;
    const std::size_t nExperts = _config.expertCount;
    const std::size_t K        = _config.expertUsedCount;

    // Per-expert intermediate width from the tensor (ne0=n_embd contiguous;
    // ne1 = n_ff for separate, 2*n_ff for fused).
    const core::gguf::GgufTensor& gateSrc = fused ? *gateUpFused : *gateExpsP;
    if (gateSrc.dimensions.size() < 3) {
        throw std::runtime_error(
            "Qwen35MoeBackend: expert gate/gate_up tensor must be 3-D "
            "[n_embd, n_ff(*2), n_expert]");
    }
    if (fused && (gateSrc.dimensions[1] % 2) != 0) {
        throw std::runtime_error(
            "Qwen35MoeBackend: fused ffn_gate_up_exps ne1 must be even");
    }
    const std::size_t n_ff_exp =
        fused ? gateSrc.dimensions[1] / 2 : gateSrc.dimensions[1];

    float* const normBuf       = s.normBuf.as<float>();  // == moeInput
    float* const gateOutBuf    = s.gateOut.as<float>();
    float* const upOutBuf      = s.upOut.as<float>();
    float* const matmulScratch = s.matmulScratch.as<float>();
    float* const moeAccumBuf   = s.moeAccumBuf.as<float>();
    float* const expertOutBuf  = s.expertOutBuf.as<float>();
    (void)normBuf;

    // --- router: logits = ffn_gate_inp @ x, then top-K softmax -------
    _gmm.matmul(routerW.type, routerW.usmPtr, nExperts, d_model,
                moeInput, T, upOutBuf, matmulScratch);   // upOutBuf [T, nExperts]

    _topKIdx.resize(T * K);
    _topKWeight.resize(T * K);
    cmp::moeTopKRoute(upOutBuf, T, nExperts, K,
                      _topKIdx.data(), _topKWeight.data());

    // Optional router-weight scale (llama.cpp w_scale); 0 = unset = 1.0.
    const float wScale = (_config.expertWeightsScale != 0.0F)
                             ? _config.expertWeightsScale : 1.0F;

    // Per-expert byte strides from the QuantType registry. In the fused
    // layout gate and up share one weight block: the gate rows are the
    // first n_ff_exp, the up rows follow at `gateBytesHalf`. In the
    // separate layout each has its own per-expert block.
    const core::gguf::GgufTensor& upSrc = fused ? *gateUpFused : *upExpsP;
    const compute::QuantType* const qtGate = compute::quantType(gateSrc.type);
    const compute::QuantType* const qtUp   = compute::quantType(upSrc.type);
    const compute::QuantType* const qtDown = compute::quantType(downExps.type);
    if (qtGate == nullptr || qtUp == nullptr || qtDown == nullptr) {
        throw std::runtime_error(
            "Qwen35MoeBackend: expert weight type(s) not in QuantType registry");
    }
    const std::size_t rowBytesGate =
        (d_model / qtGate->blockElements()) * qtGate->blockBytes();
    const std::size_t rowBytesUp =
        (d_model / qtUp->blockElements()) * qtUp->blockBytes();
    const std::size_t gateBytesHalf = n_ff_exp * rowBytesGate;   // fused split
    // Per-expert block stride: fused holds 2*n_ff rows, separate holds n_ff.
    const std::size_t bytesGate = (fused ? 2 : 1) * n_ff_exp * rowBytesGate;
    const std::size_t bytesUp   = (fused ? 2 : 1) * n_ff_exp * rowBytesUp;
    const std::size_t bytesDown =
        d_model * (n_ff_exp / qtDown->blockElements()) * qtDown->blockBytes();

    const auto* const gateBase = static_cast<const std::uint8_t*>(gateSrc.usmPtr);
    const auto* const upBase   = static_cast<const std::uint8_t*>(upSrc.usmPtr);
    const auto* const downBase = static_cast<const std::uint8_t*>(downExps.usmPtr);
    const core::gguf::GgmlType gateType = gateSrc.type;
    const core::gguf::GgmlType upType   = upSrc.type;

    // --- zero the accumulator ----------------------------------------
    _ops.mulScalarAsync(moeAccumBuf, 0.0F, T * d_model);

    // --- routed experts: per-token top-K dispatch --------------------
    // Correctness-first (matches decode). Prefill expert-grouping is a
    // perf follow-up (M-Q3N.4).
    for (std::size_t t = 0; t < T; ++t) {
        const float* const xt     = moeInput    + t * d_model;
        float* const       accumT = moeAccumBuf + t * d_model;
        for (std::size_t k = 0; k < K; ++k) {
            const std::size_t e = static_cast<std::size_t>(_topKIdx[t * K + k]);
            const float routerWeight = _topKWeight[t * K + k];

            const void* Wg = gateBase + e * bytesGate;
            const void* Wu = fused
                ? static_cast<const void*>(gateBase + e * bytesGate + gateBytesHalf)
                : static_cast<const void*>(upBase + e * bytesUp);
            const void* Wd = downBase + e * bytesDown;

            _gmm.matmulAsync(gateType, Wg, n_ff_exp, d_model,
                             xt, 1, gateOutBuf, matmulScratch);
            _gmm.matmulAsync(upType, Wu, n_ff_exp, d_model,
                             xt, 1, upOutBuf, matmulScratch);
            _ops.siluMulAsync(gateOutBuf, upOutBuf, n_ff_exp);  // silu(gate)*up
            _gmm.matmulAsync(downExps.type, Wd, d_model, n_ff_exp,
                             gateOutBuf, 1, expertOutBuf, matmulScratch);

            _ops.scaledAddResidualAsync(accumT, expertOutBuf,
                                        routerWeight * wScale, d_model);
        }
    }

    // --- shared expert (always-on) + sigmoid gate --------------------
    // ffn_{gate,up}_shexp: [n_embd, n_ff_shexp]; ffn_down_shexp: [n_ff_shexp,
    // n_embd]; ffn_gate_inp_shexp: [n_embd] -> one scalar per token.
    const auto* upShexp = w.findBlock(blockIdx, "ffn_up_shexp.weight");
    if (upShexp != nullptr) {
        const auto& gateShexp  = requireBlock(w, blockIdx, "ffn_gate_shexp.weight");
        const auto& downShexp  = requireBlock(w, blockIdx, "ffn_down_shexp.weight");
        const auto& routerSh   = requireBlock(w, blockIdx, "ffn_gate_inp_shexp.weight");
        const std::size_t n_ff_shexp = gateShexp.dimensions.size() >= 2
                                           ? gateShexp.dimensions[1] : 0;
        if (n_ff_shexp == 0) {
            throw std::runtime_error(
                "Qwen35MoeBackend: ffn_gate_shexp has unexpected shape");
        }

        // gate/up over the full batch, silu-mul, down.
        {
            compute::UnorderedScope u{_ops};
            _gmm.matmulAsync(gateShexp.type, gateShexp.usmPtr, n_ff_shexp, d_model,
                             moeInput, T, gateOutBuf, matmulScratch);
            _gmm.matmulAsync(upShexp->type, upShexp->usmPtr, n_ff_shexp, d_model,
                             moeInput, T, upOutBuf, matmulScratch);
        }
        _ops.siluMulAsync(gateOutBuf, upOutBuf, T * n_ff_shexp);
        _gmm.matmulAsync(downShexp.type, downShexp.usmPtr, d_model, n_ff_shexp,
                         gateOutBuf, T, expertOutBuf, matmulScratch);

        // scalar gate per token -> sigmoid -> broadcast multiply.
        // scoreScratch is [maxSeq] fp32; T <= maxSeq, so it holds [T, 1].
        float* const gateScalar = s.scoreScratch.as<float>();
        _gmm.matmulAsync(routerSh.type, routerSh.usmPtr, 1, d_model,
                         moeInput, T, gateScalar, matmulScratch);
        _ops.sigmoidGateMulAsync(expertOutBuf, gateScalar, T, d_model,
                                 /*gateDim=*/1);

        _ops.addResidualAsync(moeAccumBuf, expertOutBuf, T * d_model);
    }
}

} // namespace mimirmind::runtime::arch