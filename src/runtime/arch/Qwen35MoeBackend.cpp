// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/arch/Qwen35MoeBackend.hpp"

#include "compute/ComputeMatmul.hpp"
#include "compute/ComputeOps.hpp"
#include "core/modelopt/BlockScaleSwizzle.hpp" // E-d.4b swizzledBlockScaleBytes

#include <algorithm>
#include "compute/Embedding.hpp"
#include "compute/MoeRouting.hpp"
#include "compute/QuantType.hpp"
#include "compute/QuantTypeRegistry.hpp"
#include "core/gguf/GgufTypes.hpp"
#include "core/gguf/WeightsMap.hpp"
#include "core/log/Log.hpp"
#include "model/LlmConfig.hpp"
#include "runtime/BlockBuffers.hpp"
#include "runtime/KvCache.hpp"
#include "runtime/serving/PagedKvPool.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>

namespace mimirmind::runtime::arch {

namespace {

// Block geometry (elements, bytes) for an expert weight type. NVFP4_BLK is a
// runtime-only blocked format (32-element super-blocks of 20 bytes) not in the
// QuantType registry, so handle it explicitly; everything else goes through the
// registry. Returns {0,0} for an unregistered non-NVFP4 type (caller checks).
inline std::pair<std::size_t, std::size_t>
moeBlockGeom(core::gguf::GgmlType type) {
    if (type == core::gguf::GgmlType::NVFP4_BLK) {
        return {32, 20};
    }
    const compute::QuantType* qt = compute::quantType(type);
    return qt != nullptr ? std::pair<std::size_t, std::size_t>{qt->blockElements(),
                                                               qt->blockBytes()}
                         : std::pair<std::size_t, std::size_t>{0, 0};
}

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
    if (const char* g = std::getenv("MIMIRMIND_GROUPED_MOE")) {
        // Opt-in only (default off). "1" = host-driven grouped (Option 1,
        // slower than fused-K batched on GB10 — per-layer expOffset D2H + host
        // per-expert loop). "2" = device-driven grouped (Option 2, Sub-Step
        // E): moe_group_tiles + a single moe_grouped_gemm_nvfp4blk launch per
        // projection, NO D2H — the shape that can actually beat fused-K.
        _moeGroupedPrefill      = (g[0] == '1' && g[1] == '\0');
        _moeGroupedDeviceDriven = (g[0] == '2' && g[1] == '\0');
        // "3" = FP4-tensor-core grouped (Sub-Step E-d): padded per-expert
        // act-quant + the CUTLASS block-scaled NVFP4 grouped GEMM. The only
        // path whose FP4-TC compute density can beat batched fused-K on GB10.
        _moeGroupedTc           = (g[0] == '3' && g[1] == '\0');
        if (_moeGroupedDeviceDriven || _moeGroupedTc) {
            _moeGroupedPrefill = true;   // route prefill through runMoeFfnGrouped
        }
    }
    _ssmTrace = (std::getenv("MIMIRMIND_SSM_TRACE") != nullptr);
    if (const char* d = std::getenv("MIMIRMIND_SSM_DUMP")) {
        _ssmDump    = true;
        _ssmDumpDir = d;
        if (const char* pp = std::getenv("MIMIRMIND_SSM_DUMP_POS")) {
            _ssmDumpPos = std::strtol(pp, nullptr, 10);
        }
        MM_LOG_INFO("qwen35moe",
                    "MIMIRMIND_SSM_DUMP active — dir='{}' pos={} (directional "
                    "per-block residual dump)",
                    _ssmDumpDir, _ssmDumpPos);
    }
    if (const char* d = std::getenv("MIMIRMIND_GDN_DUMP")) {
        _gdnDump    = true;
        _gdnDumpDir = d;
        if (const char* b = std::getenv("MIMIRMIND_GDN_DUMP_BLK")) {
            _gdnDumpBlk = static_cast<std::size_t>(std::strtol(b, nullptr, 10));
        }
        MM_LOG_INFO("qwen35moe",
                    "MIMIRMIND_GDN_DUMP active — dir='{}' blk={} (recurrence "
                    "in/out isolation dump)",
                    _gdnDumpDir, _gdnDumpBlk);
    }
    _q8Dp4a   = (std::getenv("MIMIRMIND_Q8_DP4A") != nullptr);
    _moeDeviceTopKEnabled = (std::getenv("MIMIRMIND_MOE_DEVICE_TOPK") != nullptr);
    // Chunked GatedDeltaNet prefill auto-gate (M-Q3N.4). See the header for the
    // precedence rules; MIN_T (the A/B sweep knob) overrides the coarse flag.
    _gdnChunkMinT = kGdnChunkMinTDefault;
    if (std::getenv("MIMIRMIND_GDN_CHUNK") != nullptr) {
        _gdnChunkMinT = 2;  // force chunk for every prefill (T > 1)
    }
    if (const char* mt = std::getenv("MIMIRMIND_GDN_CHUNK_MIN_T")) {
        const long v = std::strtol(mt, nullptr, 10);
        if (v >= 2) {
            _gdnChunkMinT = static_cast<std::size_t>(v);
        } else {
            MM_LOG_WARN("qwen35moe",
                        "MIMIRMIND_GDN_CHUNK_MIN_T='{}' ignored (need >= 2)", mt);
        }
    }
    for (std::size_t i = 0; i < 4; ++i) {
        _ropeSections[i] = i < _config.ropeSections.size()
                               ? _config.ropeSections[i]
                               : 0;
    }
    std::size_t recurrent = 0;
    _fullAttnDense.assign(_config.blockCount,
                          std::numeric_limits<std::size_t>::max());
    std::size_t denseFull = 0;
    for (std::size_t b = 0; b < _config.blockCount; ++b) {
        if (_config.isRecurrentLayer(b)) {
            ++recurrent;
        } else {
            _fullAttnDense[b] = denseFull++;   // dense PagedKvPool layer index
        }
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
    if (_gdnChunkMinT == kGdnChunkMinTDefault) {
        MM_LOG_INFO("qwen35moe", "GatedDeltaNet prefill: AR (chunk disabled)");
    } else {
        MM_LOG_INFO("qwen35moe",
                    "GatedDeltaNet prefill: chunked (C=64) for T>={}", _gdnChunkMinT);
    }
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

void Qwen35MoeBackend::traceDump(const char* tag, std::size_t blockIdx,
                                 std::size_t pos, const float* p,
                                 std::size_t n) const {
    if (_ssmDumpPos >= 0 && pos != static_cast<std::size_t>(_ssmDumpPos)) {
        return;
    }
    _gmm.sync();  // p is a unified-memory pointer; readable after sync.
    const std::string path = _ssmDumpDir + "/pos" + std::to_string(pos) +
                             "-blk" + std::to_string(blockIdx) + "-" + tag +
                             ".bin";
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) {
        MM_LOG_WARN("ssm-dump", "cannot open '{}' for write", path);
        return;
    }
    f.write(reinterpret_cast<const char*>(p),
            static_cast<std::streamsize>(n * sizeof(float)));
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

    if (_ssmTrace || _ssmDump) {
        const std::size_t base    = cache.length();
        const std::size_t last    = (T > 0 ? T - 1 : 0);
        const std::size_t lastPos = base + last;
        const char* kind = _config.isRecurrentLayer(blockIdx) ? "xout(lin)"
                                                              : "xout(full)";
        if (_ssmTrace) {
            traceNorm(kind, blockIdx, lastPos, x, T * s.d_model);
        }
        if (_ssmDump) {
            // Directional dump of the residual stream after this block — the
            // vector to diff for prefill-vs-decode localisation. With a target
            // position set (MIMIRMIND_SSM_DUMP_POS), dump that absolute
            // position's row whenever it falls inside this call's T-window (so
            // a T=N prefill dumps the same position a T=1 decode step does);
            // otherwise dump the last row every call (decode-trajectory mode).
            if (_ssmDumpPos >= 0) {
                const std::size_t tgt = static_cast<std::size_t>(_ssmDumpPos);
                if (tgt >= base && tgt < base + T) {
                    traceDump("xout", blockIdx, tgt,
                              x + (tgt - base) * s.d_model, s.d_model);
                }
            } else {
                // Dump every row at its absolute position, so a single T=N
                // prefill pass captures the same positions a T=1 decode
                // trajectory does — the basis for the prefill-vs-decode diff.
                for (std::size_t r = 0; r < T; ++r) {
                    traceDump("xout", blockIdx, base + r,
                              x + r * s.d_model, s.d_model);
                }
            }
        }
    }
}

void Qwen35MoeBackend::runFullAttentionBlock(std::size_t   blockIdx,
                                             float*        x,
                                             std::size_t   T,
                                             KvCache&      cache,
                                             BlockBuffers& s,
                                             bool          diag,
                                             std::size_t   kvLayerIdx) {
    // Weight-block index (blockIdx) vs KV-cache layer index (kvL) are
    // decoupled so the MTP module can read blk.<mtp> weights while writing
    // into a private 1-layer KvCache at layer 0.
    const std::size_t kvL =
        (kvLayerIdx == std::numeric_limits<std::size_t>::max()) ? blockIdx
                                                                : kvLayerIdx;
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

    void* const kSlot = cache.writeSlotK(kvL);
    void* const vSlot = cache.writeSlotV(kvL);
    void* const kBase = const_cast<void*>(cache.baseK(kvL));
    void* const vBase = const_cast<void*>(cache.baseV(kvL));

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
    _ops.attentionAsync(qBuf, cache.baseK(kvL), cache.baseV(kvL),
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
    // Prefill (T>1) routes the routed-MoE through the amortised batched
    // fused-K path when prefill routing scratch is set (setPrefillMoeScratch),
    // turning O(T) per-token expert-weight re-reads into one batched pass.
    // runMoeFfnBatched falls back internally to runMoeFfn for unsupported
    // expert layouts. nullptr scratch => the historical per-token path
    // (single-session generate is unaffected).
    if (_prefillMoeExpIdx != nullptr && T > 1) {
        if (_moeGroupedPrefill) {
            runMoeFfnGrouped(blockIdx, normBuf, T, _prefillMoeExpIdx,
                             _prefillMoeKw, s);
        } else {
            runMoeFfnBatched(blockIdx, normBuf, T, _prefillMoeExpIdx,
                             _prefillMoeKw, s);
        }
    } else {
        runMoeFfn(blockIdx, normBuf, T, s);
    }

    trace("ffn residual");
    _ops.addResidualAsync(x, s.moeAccumBuf.as<float>(), T * d_model);
}

void Qwen35MoeBackend::runMtpDraftStep(const float*  hidden,
                                       std::int32_t  prevTok,
                                       KvCache&      mtpCache,
                                       BlockBuffers& s,
                                       float*        embScratch,
                                       float*        catScratch,
                                       float*        ehScratch,
                                       float*        logitsOut,
                                       float*        logitsScratch) {
    const auto& w = _weights;
    const std::size_t mtpBlk = _config.blockCount;   // nextn module = blk.<blockCount>
    const std::size_t d      = s.d_model;
    const float       eps    = _config.rmsNormEps;

    const auto& enorm      = requireBlock(w, mtpBlk, "nextn.enorm.weight");
    const auto& hnorm      = requireBlock(w, mtpBlk, "nextn.hnorm.weight");
    const auto& ehProj     = requireBlock(w, mtpBlk, "nextn.eh_proj.weight");
    const auto& sharedNorm = requireBlock(w, mtpBlk, "nextn.shared_head_norm.weight");

    const auto* tokEmb = w.find("token_embd.weight");
    const auto* lmHead = w.find("output.weight");
    if (lmHead == nullptr) lmHead = tokEmb;
    if (tokEmb == nullptr || lmHead == nullptr) {
        throw std::runtime_error("runMtpDraftStep: shared embed/lm_head missing");
    }
    const std::size_t vocabEmb = tokEmb->dimensions.size() >= 2
                                     ? tokEmb->dimensions[1] : d;
    const std::size_t vocabLm  = lmHead->dimensions.size() >= 2
                                     ? lmHead->dimensions[1] : d;

    // 1. token embedding of the just-produced token (shared trunk embedding).
    //    qwen35moe does not scale embeddings (scalesEmbedding()==false).
    const std::int32_t tokArr[1] = {prevTok};
    compute::embeddingLookup(tokEmb->type, tokEmb->usmPtr, d, vocabEmb,
                             std::span<const std::int32_t>{tokArr, 1}, embScratch);

    // 2. h = eh_proj( concat( RMSNorm(embed, enorm), RMSNorm(hidden, hnorm) ) )
    //    llama.cpp's GGUF eh_proj expects the embedding-norm FIRST, then the
    //    hidden-norm (the reverse of the HF/vLLM fc(cat(hnorm, enorm)) order —
    //    the convert step swaps the concat halves). catScratch = [ enorm-of-
    //    embed (d) | hnorm-of-hidden (d) ]  (2*d).
    _ops.rmsNormAsync(embScratch, 1, d, static_cast<const float*>(enorm.usmPtr),
                      eps, catScratch);
    _ops.rmsNormAsync(hidden,     1, d, static_cast<const float*>(hnorm.usmPtr),
                      eps, catScratch + d);
    _gmm.matmul(ehProj.type, ehProj.usmPtr, d, 2 * d, catScratch, 1,
                ehScratch, s.matmulScratch.as<float>());

    // 3. the nextn transformer block (attn + MoE), in place on ehScratch,
    //    using the private MTP KV cache (layer 0). No commit here — the caller
    //    advances / rolls back mtpCache around the verify/accept loop.
    runFullAttentionBlock(mtpBlk, ehScratch, /*T=*/1, mtpCache, s,
                          /*diag=*/false, /*kvLayerIdx=*/0);

    // 4. shared head: RMSNorm(shared_head_norm) -> shared lm_head -> logits.
    _ops.rmsNormAsync(ehScratch, 1, d,
                      static_cast<const float*>(sharedNorm.usmPtr), eps,
                      s.normBuf.as<float>());
    _gmm.matmul(lmHead->type, lmHead->usmPtr, vocabLm, d,
                s.normBuf.as<float>(), 1, logitsOut, logitsScratch);
    // ehScratch now holds the block-<mtp> output = the next-step MTP hidden.
}

void Qwen35MoeBackend::runMtpDraftStepBatched(
        const float* hidden, const std::int32_t* prevTok, std::size_t nSeq,
        const BatchedDecodeCtx& ctx, std::size_t kvPoolLayer, BlockBuffers& s,
        float* embScratch, float* catScratch, float* ehScratch,
        float* tmpE, float* tmpH, float* logitsOut, float* logitsScratch,
        bool skipHead) {
    if (nSeq == 0) {
        return;
    }
    const auto&       w      = _weights;
    const std::size_t mtpBlk = _config.blockCount;   // nextn = blk.<blockCount>
    const std::size_t d      = s.d_model;
    const float       eps    = _config.rmsNormEps;

    const auto& enorm      = requireBlock(w, mtpBlk, "nextn.enorm.weight");
    const auto& hnorm      = requireBlock(w, mtpBlk, "nextn.hnorm.weight");
    const auto& ehProj     = requireBlock(w, mtpBlk, "nextn.eh_proj.weight");
    const auto& sharedNorm = requireBlock(w, mtpBlk, "nextn.shared_head_norm.weight");

    const auto* tokEmb = w.find("token_embd.weight");
    const auto* lmHead = w.find("output.weight");
    if (lmHead == nullptr) lmHead = tokEmb;
    if (tokEmb == nullptr || lmHead == nullptr) {
        throw std::runtime_error("runMtpDraftStepBatched: shared embed/lm_head missing");
    }
    const std::size_t vocabEmb = tokEmb->dimensions.size() >= 2 ? tokEmb->dimensions[1] : d;
    const std::size_t vocabLm  = lmHead->dimensions.size() >= 2 ? lmHead->dimensions[1] : d;

    // 1. batched token embedding of prevTok[nSeq] (shared trunk embedding).
    compute::embeddingLookup(tokEmb->type, tokEmb->usmPtr, d, vocabEmb,
                             std::span<const std::int32_t>{prevTok, nSeq}, embScratch);

    // 2. eh = eh_proj( concat( RMSNorm(embed, enorm), RMSNorm(hidden, hnorm) ) ).
    //    Normalise into temps, then interleave into the [nSeq, 2d] concat
    //    ([enorm-embed | hnorm-hidden] per row) since rmsnorm has no strided out.
    _ops.rmsNormAsync(embScratch, nSeq, d, static_cast<const float*>(enorm.usmPtr), eps, tmpE);
    _ops.rmsNormAsync(hidden,     nSeq, d, static_cast<const float*>(hnorm.usmPtr), eps, tmpH);
    for (std::size_t r = 0; r < nSeq; ++r) {
        _ops.appendMemoryCopy(catScratch + r * 2 * d,     tmpE + r * d, d * sizeof(float));
        _ops.appendMemoryCopy(catScratch + r * 2 * d + d, tmpH + r * d, d * sizeof(float));
    }
    _gmm.matmulAsync(ehProj.type, ehProj.usmPtr, d, 2 * d, catScratch, nSeq,
                     ehScratch, s.matmulScratch.as<float>());

    // 3. the nextn transformer block (attn + MoE), batched over nSeq slots,
    //    in place on ehScratch, using the paged nextn pool layer `kvPoolLayer`.
    runFullAttentionBlockBatched(mtpBlk, ehScratch, ctx, s, kvPoolLayer);

    // 4. shared head (skipped while seeding): RMSNorm(shared_head_norm) -> lm_head.
    if (!skipHead) {
        _ops.rmsNormAsync(ehScratch, nSeq, d,
                          static_cast<const float*>(sharedNorm.usmPtr), eps,
                          s.normBuf.as<float>());
        _gmm.matmulAsync(lmHead->type, lmHead->usmPtr, vocabLm, d,
                         s.normBuf.as<float>(), nSeq, logitsOut, logitsScratch);
    }
    // ehScratch now holds the per-slot block-<mtp> output = next MTP hidden.
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
    // reallocation). Slab-aware per-layer stride (BlockBuffers convention,
    // see BlockBuffers.hpp): the layer base steps by ssmSlabNSeq*elems so a
    // multi-sequence slab (serving) and the single-sequence path share one
    // indexing rule. ssmSlabNSeq defaults to 1, where this reduces to
    // blockIdx*elems (bit-identical to the historical single-session path).
    // ServingSession::prefillSlot pre-offsets ssmStatePtr/ssmConvStatePtr by
    // slot*elems so this T>1 path targets that slot's slab slice.
    const std::size_t slabNSeq = (s.ssmSlabNSeq != 0) ? s.ssmSlabNSeq : 1;
    float* const stateBuf  = s.ssmStatePtr     + blockIdx * (slabNSeq * stateElems);
    float* const convState = s.ssmConvStatePtr + blockIdx * (slabNSeq * convStateElems);
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
    if (T > 1 && T >= _gdnChunkMinT) {
        // Chunked prefill: K0 (cumgate) -> K1 (kkt inverse) -> K2 (forward).
        // Parity-equivalent to the AR recurrence (cuda_parity 10/10) but
        // parallel across the chunk instead of T sequential steps. gateBuf is
        // gLog; K0 turns it into gCum. Chunk size C = 64. Gated on T so short
        // prefills (where chunking is correct but not faster) keep the AR path.
        const std::size_t cChunk = 64;
        float* const gCum = s.ssmGCum.as<float>();
        float* const a0   = s.ssmA0.as<float>();
        _ops.deltanetChunkCumGateAsync(gateBuf, gCum, T, hV, cChunk);
        _ops.deltanetKktSolveInverseAsync(kBuf, betaBuf, a0, T, hV, S, cChunk);
        _ops.deltanetChunkForwardAsync(qBuf, kBuf, vBuf, gCum, betaBuf, a0,
                                       stateBuf, deltaOut, T, hV, S, cChunk);
    } else {
        _ops.gatedDeltaNetRecurrentAsync(qBuf, kBuf, vBuf, gateBuf, betaBuf,
                                         stateBuf, deltaOut, T, hV, S);
    }

    if (_ssmTrace) {
        // For T>1 (prefill) the state reflects the last token; for decode
        // (T==1) cache.length() is the position of the token just added.
        const std::size_t pos = cache.length() + (T > 0 ? T - 1 : 0);
        traceNorm("gate",  blockIdx, pos, gateBuf, T * hV);
        traceNorm("state", blockIdx, pos, stateBuf, stateElems);
        traceNorm("dnet",  blockIdx, pos, deltaOut, T * valueDim);
    }

    if (_gdnDump && blockIdx == _gdnDumpBlk) {
        // Isolate the recurrence: dump its exact in/out tensors so the fp64 HF
        // reference can be run on identical inputs. q,k are L2-normed here (as
        // the kernel consumes them); glog=gateBuf, beta=betaBuf. One prefill.
        _gmm.sync();
        auto dumpT = [&](const char* tag, const float* p, std::size_t n) {
            std::ofstream f(_gdnDumpDir + "/blk" + std::to_string(blockIdx) +
                            "-" + tag + ".bin",
                            std::ios::binary | std::ios::trunc);
            if (f) {
                f.write(reinterpret_cast<const char*>(p),
                        static_cast<std::streamsize>(n * sizeof(float)));
            }
        };
        dumpT("q",    qBuf,     T * hV * S);
        dumpT("k",    kBuf,     T * hV * S);
        dumpT("v",    vBuf,     T * hV * S);
        dumpT("glog", gateBuf,  T * hV);
        dumpT("beta", betaBuf,  T * hV);
        dumpT("dnet", deltaOut, T * valueDim);
        MM_LOG_INFO("gdn-dump", "blk{} dumped q/k/v/glog/beta/dnet T={} hV={} S={}",
                    blockIdx, T, hV, S);
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

    if (diag) {
        const std::size_t posPA = cache.length() + (T > 0 ? T - 1 : 0);
        traceNorm("postattn", blockIdx, posPA, x, T * d_model);
    }

    trace("post_attention_norm");
    _ops.rmsNormAsync(x, T, d_model,
                      static_cast<const float*>(attnPost.usmPtr), eps, normBuf);

    trace("MoE FFN");
    // Prefill (T>1) routes the routed-MoE through the amortised batched
    // fused-K path when prefill routing scratch is set (setPrefillMoeScratch),
    // turning O(T) per-token expert-weight re-reads into one batched pass.
    // runMoeFfnBatched falls back internally to runMoeFfn for unsupported
    // expert layouts. nullptr scratch => the historical per-token path
    // (single-session generate is unaffected).
    if (_prefillMoeExpIdx != nullptr && T > 1) {
        if (_moeGroupedPrefill) {
            runMoeFfnGrouped(blockIdx, normBuf, T, _prefillMoeExpIdx,
                             _prefillMoeKw, s);
        } else {
            runMoeFfnBatched(blockIdx, normBuf, T, _prefillMoeExpIdx,
                             _prefillMoeKw, s);
        }
    } else {
        runMoeFfn(blockIdx, normBuf, T, s);
    }

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

    // --- M-Q3N.5: decide device-side top-K up front. These predicates are
    // identical to useMoeFusedDown / useGateUpFused below (recomputed here
    // only because they are needed before the router to gate the host
    // moeTopKRoute). deviceTopK is true only on the fully-fused decode path,
    // where _topKIdx/_topKWeight go unused (the fused-K kernels read
    // expIdxSlot/kwSlot filled on-device instead) — so skipping the host
    // top-K there removes the per-layer host round trip.
    const core::gguf::GgufTensor& upSrcPre = fused ? *gateUpFused : *upExpsP;
    const bool useMoeFusedDownPre =
        _moeFusedDownEnabled && T == 1 &&
        _gmm.moeDownFusedKAvailable(downExps.type) && (n_ff_exp % 256 == 0) &&
        s.moeExpIdxScratch.get() != nullptr && s.moeKwScratch.get() != nullptr &&
        s.moeGateCompact.get() != nullptr;
    const bool useGateUpFusedPre =
        !fused && gateSrc.type == upSrcPre.type &&
        _gmm.moeGateUpFusedKAvailable(gateSrc.type) && (d_model % 256 == 0);
    const bool deviceTopK =
        _moeDeviceTopKEnabled && useMoeFusedDownPre && useGateUpFusedPre;

    // --- router: logits = ffn_gate_inp @ x, then top-K softmax -------
    // On the device-top-K path nothing reads upOutBuf on the host, so the
    // router runs async — this removes the last per-layer host sync on the
    // fused decode block (matmul() = matmulAsync + stream sync), letting the
    // whole trunk pipeline and unblocking graph capture. The host path keeps
    // the sync because cmp::moeTopKRoute reads upOutBuf on the CPU below.
    if (deviceTopK) {
        _gmm.matmulAsync(routerW.type, routerW.usmPtr, nExperts, d_model,
                         moeInput, T, upOutBuf, matmulScratch);
    } else {
        _gmm.matmul(routerW.type, routerW.usmPtr, nExperts, d_model,
                    moeInput, T, upOutBuf, matmulScratch);  // upOutBuf [T, nExperts]
    }

    _topKIdx.resize(T * K);
    _topKWeight.resize(T * K);
    if (!deviceTopK) {
        cmp::moeTopKRoute(upOutBuf, T, nExperts, K,
                          _topKIdx.data(), _topKWeight.data());
    }

    // Optional router-weight scale (llama.cpp w_scale); 0 = unset = 1.0.
    const float wScale = (_config.expertWeightsScale != 0.0F)
                             ? _config.expertWeightsScale : 1.0F;

    // Per-expert byte strides from the QuantType registry. In the fused
    // layout gate and up share one weight block: the gate rows are the
    // first n_ff_exp, the up rows follow at `gateBytesHalf`. In the
    // separate layout each has its own per-expert block.
    const core::gguf::GgufTensor& upSrc = fused ? *gateUpFused : *upExpsP;
    const auto [geGate, gbGate] = moeBlockGeom(gateSrc.type);
    const auto [geUp,   gbUp]   = moeBlockGeom(upSrc.type);
    const auto [geDown, gbDown] = moeBlockGeom(downExps.type);
    if (geGate == 0 || geUp == 0 || geDown == 0) {
        throw std::runtime_error(
            "Qwen35MoeBackend: expert weight type(s) not in QuantType registry");
    }
    const std::size_t rowBytesGate = (d_model / geGate) * gbGate;
    const std::size_t rowBytesUp   = (d_model / geUp)   * gbUp;
    const std::size_t gateBytesHalf = n_ff_exp * rowBytesGate;   // fused split
    // Per-expert block stride: fused holds 2*n_ff rows, separate holds n_ff.
    const std::size_t bytesGate = (fused ? 2 : 1) * n_ff_exp * rowBytesGate;
    const std::size_t bytesUp   = (fused ? 2 : 1) * n_ff_exp * rowBytesUp;
    const std::size_t bytesDown =
        d_model * (n_ff_exp / geDown) * gbDown;

    const auto* const gateBase = static_cast<const std::uint8_t*>(gateSrc.usmPtr);
    const auto* const upBase   = static_cast<const std::uint8_t*>(upSrc.usmPtr);
    const auto* const downBase = static_cast<const std::uint8_t*>(downExps.usmPtr);
    const core::gguf::GgmlType gateType = gateSrc.type;
    const core::gguf::GgmlType upType   = upSrc.type;

    // --- zero the accumulator ----------------------------------------
    _ops.mulScalarAsync(moeAccumBuf, 0.0F, T * d_model);

    // --- routed experts ----------------------------------------------
    // M-Q3N.4 MoE-Expert-Grouping (decode). At T==1 the down-projections
    // of all K experts fold into ONE fused-K launch (+ the router-weighted
    // residual add), replacing K separate down-matmuls + K scaledAdds. The
    // gate/up half stays per-expert (separate Q4_K experts + SILU). Prefill
    // (T>1) and any model whose down quant lacks a fused kernel keep the
    // sequential per-token/per-expert path.
    const bool useMoeFusedDown =
        _moeFusedDownEnabled &&
        T == 1 &&
        _gmm.moeDownFusedKAvailable(downExps.type) &&
        (n_ff_exp % 256 == 0) &&
        s.moeExpIdxScratch.get() != nullptr &&
        s.moeKwScratch.get()     != nullptr &&
        s.moeGateCompact.get()   != nullptr;

    if (useMoeFusedDown) {
        float* const gateActAll = s.moeGateCompact.as<float>();
        const float* const xt   = moeInput;   // T == 1

        // Per-layer routing scratch (host-visible USM on UMA); both fused
        // kernels read it on the stream after these writes.
        auto* const expIdxSlot = s.moeExpIdxScratch.as<std::int32_t>() + blockIdx * K;
        auto* const kwSlot     = s.moeKwScratch.as<float>()            + blockIdx * K;
        if (deviceTopK) {
            // top-K straight into the USM slots on the stream — no host round
            // trip. deviceTopK guarantees the fused gate/up path below reads
            // expIdxSlot (never the host _topKIdx fallback).
            _ops.moeTopKRouteDeviceAsync(upOutBuf, expIdxSlot, kwSlot,
                                         T, nExperts, K, wScale);
        } else {
            for (std::size_t k = 0; k < K; ++k) {
                expIdxSlot[k] = _topKIdx[k];
                kwSlot[k]     = _topKWeight[k] * wScale;
            }
        }

        // gate/up -> silu(gate)*up into the K-strided [K, n_ff_exp] slots.
        // Fused-K when the experts are separate Q4_K banks and the kernel
        // is loaded (one launch for all K×2 GEMVs + silu); otherwise the
        // per-expert matmul + siluMul path.
        const bool useGateUpFused =
            !fused &&
            gateType == upType &&
            _gmm.moeGateUpFusedKAvailable(gateType) &&
            (d_model % 256 == 0);

        if (useGateUpFused) {
            _gmm.moeGateUpFusedKAsync(gateType, xt, gateBase, upBase,
                                      expIdxSlot, gateActAll,
                                      d_model, n_ff_exp, K, bytesGate, bytesUp);
        } else {
            for (std::size_t k = 0; k < K; ++k) {
                const std::size_t e = static_cast<std::size_t>(_topKIdx[k]);
                const void* Wg = gateBase + e * bytesGate;
                const void* Wu = fused
                    ? static_cast<const void*>(gateBase + e * bytesGate + gateBytesHalf)
                    : static_cast<const void*>(upBase + e * bytesUp);
                _gmm.matmulAsync(gateType, Wg, n_ff_exp, d_model, xt, 1, gateOutBuf, matmulScratch);
                _gmm.matmulAsync(upType,   Wu, n_ff_exp, d_model, xt, 1, upOutBuf,   matmulScratch);
                _ops.siluMulAsync(gateOutBuf, upOutBuf, n_ff_exp);  // silu(gate)*up
                _ops.appendMemoryCopy(gateActAll + k * n_ff_exp, gateOutBuf,
                                      n_ff_exp * sizeof(float));
            }
        }

        // Fused-K down projection: accum += sum_k kw[k] * (W[e_k] @ gateAct[k]).
        _gmm.moeDownFusedKAsync(downExps.type, gateActAll, downBase,
                                expIdxSlot, kwSlot, moeAccumBuf,
                                n_ff_exp, d_model, K, bytesDown);
    } else {
        // Sequential per-token top-K dispatch (prefill / no fused kernel).
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

        // gate/up over the batch, silu-mul, down. At T=1 decode the Q8_0
        // GEMVs can go through the dp4a (int8) path (M-Q3N.4e): quantize the
        // activation once per matmul input, then int8 dot products. gate/up
        // share moeInput's quantization; down quantizes the silu-mul result.
        const bool shexpDp4a =
            _q8Dp4a && T == 1 &&
            gateShexp.type == core::gguf::GgmlType::Q8_0 &&
            upShexp->type  == core::gguf::GgmlType::Q8_0 &&
            downShexp.type == core::gguf::GgmlType::Q8_0 &&
            _gmm.dp4aAvailable(gateShexp.type) &&
            (d_model % 32 == 0) && (n_ff_shexp % 32 == 0);

        if (shexpDp4a) {
            std::int8_t* const xq = s.xqI8.as<std::int8_t>();
            float* const       xs = s.xScaleI8.as<float>();
            _ops.xQuantI8Async(moeInput, xq, xs, 1, d_model);
            _gmm.matmulDp4aAsync(gateShexp.type, xq, xs, gateShexp.usmPtr,
                                 n_ff_shexp, d_model, 1, gateOutBuf);
            _gmm.matmulDp4aAsync(upShexp->type, xq, xs, upShexp->usmPtr,
                                 n_ff_shexp, d_model, 1, upOutBuf);
            _ops.siluMulAsync(gateOutBuf, upOutBuf, n_ff_shexp);
            _ops.xQuantI8Async(gateOutBuf, xq, xs, 1, n_ff_shexp);
            _gmm.matmulDp4aAsync(downShexp.type, xq, xs, downShexp.usmPtr,
                                 d_model, n_ff_shexp, 1, expertOutBuf);
        } else {
            // gate/up -> silu(gate)*up into gateOutBuf. At T=1 decode a
            // single fused Q8_0 kernel does gate+up+silu (launch reduction);
            // otherwise the two-matmul + siluMul path.
            const bool shexpFusedGu =
                T == 1 &&
                gateShexp.type == core::gguf::GgmlType::Q8_0 &&
                upShexp->type  == core::gguf::GgmlType::Q8_0 &&
                _gmm.ffnGateUpFusedQ8Available() &&
                (d_model % 32 == 0);

            if (shexpFusedGu) {
                _gmm.ffnGateUpFusedQ8Async(moeInput, gateShexp.usmPtr,
                                           upShexp->usmPtr, gateOutBuf,
                                           d_model, n_ff_shexp);
            } else {
                {
                    compute::UnorderedScope u{_ops};
                    _gmm.matmulAsync(gateShexp.type, gateShexp.usmPtr, n_ff_shexp, d_model,
                                     moeInput, T, gateOutBuf, matmulScratch);
                    _gmm.matmulAsync(upShexp->type, upShexp->usmPtr, n_ff_shexp, d_model,
                                     moeInput, T, upOutBuf, matmulScratch);
                }
                _ops.siluMulAsync(gateOutBuf, upOutBuf, T * n_ff_shexp);
            }
            _gmm.matmulAsync(downShexp.type, downShexp.usmPtr, d_model, n_ff_shexp,
                             gateOutBuf, T, expertOutBuf, matmulScratch);
        }

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

void Qwen35MoeBackend::runMoeFfnBatched(std::size_t    blockIdx,
                                        const float*   moeInput,
                                        std::size_t    nSeq,
                                        std::int32_t*  expIdxSlot,
                                        float*         kwSlot,
                                        BlockBuffers&  s) {
    namespace cmp = mimirmind::compute;

    // The batched fused-K fast path only exists for the separate Q4_K
    // gate/up + Q5_K down decode layout. Any other layout/quant (notably
    // the NVFP4->BF16 materialised experts of the Bragi target checkpoint)
    // has no batched fused-K kernel — delegate to the generic runMoeFfn,
    // whose sequential per-token dispatch already routes each of the nSeq
    // rows independently when it is handed nSeq as its T. (expIdxSlot /
    // kwSlot are used only by the fused-K path below, so they are unused
    // here — the generic path routes through _topKIdx / _topKWeight.)
    {
        const auto* gu = _weights.findBlock(blockIdx, "ffn_gate_up_exps.weight");
        const auto* g  = _weights.findBlock(blockIdx, "ffn_gate_exps.weight");
        const auto* u  = _weights.findBlock(blockIdx, "ffn_up_exps.weight");
        const auto* d  = _weights.findBlock(blockIdx, "ffn_down_exps.weight");
        bool fusedKOk = (gu == nullptr) && g != nullptr && u != nullptr &&
                        d != nullptr && g->dimensions.size() >= 3;
        if (fusedKOk) {
            const std::size_t nff = g->dimensions[1];
            // BATCHED-specific support: the *_Batched kernels handle Q4_K/BF16
            // gate-up + Q5_K/Q6_K/BF16 down. Types without a batched variant
            // (e.g. the Q5_K gate-up of one Q4_K_XL dynamic-quant layer, blk 39)
            // delegate to the generic per-token runMoeFfn, which routes each of
            // the nSeq rows independently. (moeDownFusedKAvailable is single-seq
            // scope and admits Q8_0 which has no batched down kernel.)
            const bool downBatchedOk =
                d->type == core::gguf::GgmlType::Q5_K ||
                d->type == core::gguf::GgmlType::Q6_K ||
                d->type == core::gguf::GgmlType::NVFP4_BLK ||
                d->type == core::gguf::GgmlType::BF16;
            // NVFP4_BLK is 32-element super-blocks (not 256); the other batched
            // types are 256-block. Require the matching alignment per type.
            const std::size_t blkAlign =
                (g->type == core::gguf::GgmlType::NVFP4_BLK) ? 32 : 256;
            fusedKOk = _gmm.moeGateUpFusedKAvailable(g->type) &&
                       downBatchedOk &&
                       g->type == u->type &&
                       (s.d_model % blkAlign == 0) && (nff % blkAlign == 0);
        }
        if (!fusedKOk) {
            (void)expIdxSlot; (void)kwSlot;
            runMoeFfn(blockIdx, moeInput, nSeq, s);
            return;
        }
    }

    const auto& w = _weights;

    const auto& routerW  = requireBlock(w, blockIdx, "ffn_gate_inp.weight");
    const auto& downExps = requireBlock(w, blockIdx, "ffn_down_exps.weight");

    // Serving decode requires the SEPARATE-bank fused-K layout: Q4_K gate/up
    // banks + a Q5_K down bank (the only layout the *_Batched kernels port
    // today). The single-session runMoeFfn falls back to per-expert matmuls
    // for other layouts; there is no such fallback here — a batched decode on
    // an unsupported model is a configuration error, not a slow path.
    const auto* gateUpFused = w.findBlock(blockIdx, "ffn_gate_up_exps.weight");
    if (gateUpFused != nullptr) {
        throw std::runtime_error(
            "Qwen35MoeBackend::runMoeFfnBatched: fused ffn_gate_up_exps layout "
            "is not supported on the batched decode path (need separate Q4_K "
            "gate/up + Q5_K down banks)");
    }
    const auto& gateExps = requireBlock(w, blockIdx, "ffn_gate_exps.weight");
    const auto& upExps   = requireBlock(w, blockIdx, "ffn_up_exps.weight");

    const std::size_t d_model  = s.d_model;
    const std::size_t nExperts = _config.expertCount;
    const std::size_t K        = _config.expertUsedCount;

    if (gateExps.dimensions.size() < 3) {
        throw std::runtime_error(
            "Qwen35MoeBackend::runMoeFfnBatched: expert gate tensor must be 3-D "
            "[n_embd, n_ff, n_expert]");
    }
    const std::size_t n_ff_exp = gateExps.dimensions[1];

    const bool downBatchedOk =
        downExps.type == core::gguf::GgmlType::Q5_K ||
        downExps.type == core::gguf::GgmlType::Q6_K ||
        downExps.type == core::gguf::GgmlType::NVFP4_BLK ||
        downExps.type == core::gguf::GgmlType::BF16;
    const std::size_t blkAlign =
        (gateExps.type == core::gguf::GgmlType::NVFP4_BLK) ? 32 : 256;
    if (!_gmm.moeGateUpFusedKAvailable(gateExps.type) ||
        !downBatchedOk ||
        gateExps.type != upExps.type ||
        (d_model % blkAlign != 0) || (n_ff_exp % blkAlign != 0)) {
        throw std::runtime_error(
            "Qwen35MoeBackend::runMoeFfnBatched: batched fused-K MoE kernels "
            "unavailable for this model (need Q4_K/NVFP4_BLK/BF16 gate/up + "
            "Q5_K/Q6_K/NVFP4_BLK/BF16 down, d_model/n_ff_exp block-aligned)");
    }

    float* const gateOutBuf    = s.gateOut.as<float>();
    float* const upOutBuf      = s.upOut.as<float>();
    float* const matmulScratch = s.matmulScratch.as<float>();
    float* const moeAccumBuf   = s.moeAccumBuf.as<float>();
    float* const expertOutBuf  = s.expertOutBuf.as<float>();
    float* const gateActAll    = s.moeGateCompact.as<float>();  // [nSeq*K, n_ff_exp]

    // --- router: logits[nSeq, nExperts] = ffn_gate_inp @ moeInput ----------
    // Device top-K straight into the USM routing slots — the batched fused-K
    // kernels below consume expIdxSlot/kwSlot on the same stream, so there is
    // NO host round trip and NO per-MoE-block GPU sync. moe_topk grids one
    // block per token (grid.x = nSeq), so a single async launch routes the
    // whole batch. (Removes ~1 sync per MoE block × 40 blocks per decode
    // step — the dominant host-stall in the serving decode loop.)
    const float wScale = (_config.expertWeightsScale != 0.0F)
                             ? _config.expertWeightsScale : 1.0F;
    _gmm.matmulAsync(routerW.type, routerW.usmPtr, nExperts, d_model,
                     moeInput, nSeq, upOutBuf, matmulScratch);
    _ops.moeTopKRouteDeviceAsync(upOutBuf, expIdxSlot, kwSlot,
                                 nSeq, nExperts, K, wScale);

    // Per-expert byte strides (separate banks: one block per expert). For
    // BF16 experts (NVFP4->BF16 checkpoints) the *_bf16_batched kernels index
    // by dims, so a dense 2-byte stride is used; the K-quant banks take the
    // QuantType block byte layout.
    std::size_t bytesGate = 0, bytesUp = 0, bytesDown = 0;
    if (gateExps.type == core::gguf::GgmlType::BF16) {
        bytesGate = n_ff_exp * d_model * sizeof(std::uint16_t);
        bytesUp   = n_ff_exp * d_model * sizeof(std::uint16_t);
        bytesDown = d_model * n_ff_exp * sizeof(std::uint16_t);
    } else {
        const auto [geGate, gbGate] = moeBlockGeom(gateExps.type);
        const auto [geUp,   gbUp]   = moeBlockGeom(upExps.type);
        const auto [geDown, gbDown] = moeBlockGeom(downExps.type);
        if (geGate == 0 || geUp == 0 || geDown == 0) {
            throw std::runtime_error(
                "Qwen35MoeBackend::runMoeFfnBatched: expert weight type(s) not "
                "in QuantType registry");
        }
        bytesGate = n_ff_exp * ((d_model / geGate) * gbGate);
        bytesUp   = n_ff_exp * ((d_model / geUp)   * gbUp);
        bytesDown = d_model * ((n_ff_exp / geDown) * gbDown);
    }

    const auto* const gateBase = static_cast<const std::uint8_t*>(gateExps.usmPtr);
    const auto* const upBase   = static_cast<const std::uint8_t*>(upExps.usmPtr);
    const auto* const downBase = static_cast<const std::uint8_t*>(downExps.usmPtr);

    // --- routed experts: fused gate/up -> silu*up -> fused down ------------
    // gateActAll[seq, k, f] laid out as [nSeq, K*n_ff_exp] (seq stride
    // K*n_ff_exp), exactly the batched kernels' contract.
    _gmm.moeGateUpFusedKBatchedAsync(gateExps.type, moeInput, gateBase, upBase,
                                     expIdxSlot, gateActAll, nSeq,
                                     d_model, n_ff_exp, K, bytesGate, bytesUp);

    _ops.mulScalarAsync(moeAccumBuf, 0.0F, nSeq * d_model);

    _gmm.moeDownFusedKBatchedAsync(downExps.type, gateActAll, downBase,
                                   expIdxSlot, kwSlot, moeAccumBuf, nSeq,
                                   n_ff_exp, d_model, K, bytesDown);

    // --- shared expert (always-on) + sigmoid gate -------------------------
    // Row-parallel over the nSeq tokens: identical to runMoeFfn's shared
    // expert with T := nSeq (the two-matmul + siluMul + down path; the T==1
    // fused-Q8 / dp4a shortcuts are single-token-only and intentionally
    // skipped here).
    const auto* upShexp = w.findBlock(blockIdx, "ffn_up_shexp.weight");
    if (upShexp != nullptr) {
        const auto& gateShexp = requireBlock(w, blockIdx, "ffn_gate_shexp.weight");
        const auto& downShexp = requireBlock(w, blockIdx, "ffn_down_shexp.weight");
        const auto& routerSh  = requireBlock(w, blockIdx, "ffn_gate_inp_shexp.weight");
        const std::size_t n_ff_shexp = gateShexp.dimensions.size() >= 2
                                           ? gateShexp.dimensions[1] : 0;
        if (n_ff_shexp == 0) {
            throw std::runtime_error(
                "Qwen35MoeBackend::runMoeFfnBatched: ffn_gate_shexp has "
                "unexpected shape");
        }

        {
            compute::UnorderedScope u{_ops};
            _gmm.matmulAsync(gateShexp.type, gateShexp.usmPtr, n_ff_shexp, d_model,
                             moeInput, nSeq, gateOutBuf, matmulScratch);
            _gmm.matmulAsync(upShexp->type, upShexp->usmPtr, n_ff_shexp, d_model,
                             moeInput, nSeq, upOutBuf, matmulScratch);
        }
        _ops.siluMulAsync(gateOutBuf, upOutBuf, nSeq * n_ff_shexp);
        _gmm.matmulAsync(downShexp.type, downShexp.usmPtr, d_model, n_ff_shexp,
                         gateOutBuf, nSeq, expertOutBuf, matmulScratch);

        // Scalar gate per token: [nSeq, 1] -> sigmoid -> broadcast multiply.
        float* const gateScalar = s.scoreScratch.as<float>();
        _gmm.matmulAsync(routerSh.type, routerSh.usmPtr, 1, d_model,
                         moeInput, nSeq, gateScalar, matmulScratch);
        _ops.sigmoidGateMulAsync(expertOutBuf, gateScalar, nSeq, d_model,
                                 /*gateDim=*/1);

        _ops.addResidualAsync(moeAccumBuf, expertOutBuf, nSeq * d_model);
    }
}

void Qwen35MoeBackend::runMoeFfnGrouped(std::size_t    blockIdx,
                                        const float*   moeInput,
                                        std::size_t    nSeq,
                                        std::int32_t*  expIdxSlot,
                                        float*         kwSlot,
                                        BlockBuffers&  s) {
    const auto& w = _weights;

    // Same layout gate as runMoeFfnBatched: SEPARATE gate/up + down banks the
    // fused-K kernels support. Anything else -> fall back to the generic
    // per-token runMoeFfn (which routes via _topKIdx, so expIdxSlot/kwSlot are
    // unused there).
    const auto* gateUpFused = w.findBlock(blockIdx, "ffn_gate_up_exps.weight");
    const auto* gExps = w.findBlock(blockIdx, "ffn_gate_exps.weight");
    const auto* uExps = w.findBlock(blockIdx, "ffn_up_exps.weight");
    const auto* dExps = w.findBlock(blockIdx, "ffn_down_exps.weight");
    if (gateUpFused != nullptr || gExps == nullptr || uExps == nullptr ||
        dExps == nullptr || gExps->dimensions.size() < 3) {
        (void)expIdxSlot; (void)kwSlot;
        runMoeFfn(blockIdx, moeInput, nSeq, s);
        return;
    }

    const auto& routerW  = requireBlock(w, blockIdx, "ffn_gate_inp.weight");
    const auto& gateExps = *gExps;
    const auto& upExps   = *uExps;
    const auto& downExps = *dExps;

    const std::size_t d_model  = s.d_model;
    const std::size_t nExperts = _config.expertCount;
    const std::size_t K        = _config.expertUsedCount;
    const std::size_t n_ff_exp = gateExps.dimensions[1];
    const std::size_t R        = nSeq * K;

    // Block alignment / type support identical to the batched path; on a
    // mismatch fall back to the generic per-token path rather than throw.
    const bool downOk =
        downExps.type == core::gguf::GgmlType::Q5_K ||
        downExps.type == core::gguf::GgmlType::Q6_K ||
        downExps.type == core::gguf::GgmlType::NVFP4_BLK ||
        downExps.type == core::gguf::GgmlType::BF16;
    const std::size_t blkAlign =
        (gateExps.type == core::gguf::GgmlType::NVFP4_BLK) ? 32 : 256;
    if (!_gmm.moeGateUpFusedKAvailable(gateExps.type) || !downOk ||
        gateExps.type != upExps.type ||
        (d_model % blkAlign != 0) || (n_ff_exp % blkAlign != 0)) {
        runMoeFfn(blockIdx, moeInput, nSeq, s);
        return;
    }

    float* const upOutBuf      = s.upOut.as<float>();
    float* const matmulScratch = s.matmulScratch.as<float>();
    float* const moeAccumBuf   = s.moeAccumBuf.as<float>();
    float* const gateOutBuf    = s.gateOut.as<float>();

    float* const xComp    = s.moeXCompact.as<float>();
    float* const gateComp = s.moeGateCompact.as<float>();
    float* const upComp   = s.moeUpCompact.as<float>();
    float* const downComp = s.moeDownCompact.as<float>();

    auto* const expOffset = s.moeGroupOffset.as<std::int32_t>();
    auto* const rowSrcTok = s.moeGroupRowTok.as<std::int32_t>();
    float* const rowKw    = s.moeGroupRowKw.as<float>();
    auto* const asnToRow  = s.moeGroupAsnRow.as<std::int32_t>();

    // --- router + device top-K straight into the caller's USM slots --------
    const float wScale = (_config.expertWeightsScale != 0.0F)
                             ? _config.expertWeightsScale : 1.0F;
    _gmm.matmulAsync(routerW.type, routerW.usmPtr, nExperts, d_model,
                     moeInput, nSeq, upOutBuf, matmulScratch);
    _ops.moeTopKRouteDeviceAsync(upOutBuf, expIdxSlot, kwSlot,
                                 nSeq, nExperts, K, wScale);

    // --- group the T*K assignments by expert (device counting sort) --------
    _ops.moeGroupBuildAsync(expIdxSlot, kwSlot, expOffset, rowSrcTok, rowKw,
                            asnToRow, R, nExperts, K);

    // --- gather activations into per-expert-contiguous rows (both paths) ---
    _ops.moeGatherRowsAsync(moeInput, rowSrcTok, xComp, d_model, R);

    // FP4-tensor-core grouped path needs the E-d.2b load-time side banks.
    const bool tcGrouped =
        _moeGroupedTc && _ops.moeGroupedGemmNvfp4TcAvailable() &&
        gateExps.tcNibblePtr != nullptr && upExps.tcNibblePtr != nullptr &&
        downExps.tcNibblePtr != nullptr;

    const bool deviceDrivenGrouped =
        !tcGrouped && _moeGroupedDeviceDriven &&
        gateExps.type == core::gguf::GgmlType::NVFP4_BLK &&
        upExps.type   == core::gguf::GgmlType::NVFP4_BLK &&
        downExps.type == core::gguf::GgmlType::NVFP4_BLK;

    if (tcGrouped) {
        // --- Sub-Step E-d: FP4-tensor-core grouped GEMM (F32 out) ----------
        // Each expert padded to 128 rows so its SFA sub-tensor is tile-aligned
        // in one big act-quant; every per-group pointer built on device from
        // expOffset/padOffset — nothing crosses to the host.
        namespace mo = core::modelopt;
        const std::size_t maxPad = R + nExperts * 128;
        const std::size_t nAsn   = R;                    // nSeq * K assignments
        // Per-slot scratch lives in BlockBuffers (concurrent-prefill safe);
        // lazily grown to the current maxPad on first / larger use.
        auto grow = [&](compute::ComputeBuffer& buf, std::size_t bytes) {
            if (buf.bytes() < bytes) buf = _ops.allocate(bytes);
        };
        grow(s.moeTcPadOffset,   (nExperts + 1) * sizeof(std::int32_t));
        grow(s.moeTcContigToPad, R * sizeof(std::int32_t));
        grow(s.moeTcPadAsn,      nAsn * sizeof(std::int32_t));
        grow(s.moeTcXPad,        maxPad * d_model * sizeof(float));
        grow(s.moeTcGatePad,     maxPad * n_ff_exp * sizeof(float));
        grow(s.moeTcUpPad,       maxPad * n_ff_exp * sizeof(float));
        grow(s.moeTcDownPad,     maxPad * d_model * sizeof(float));
        grow(s.moeTcABank,       maxPad * (d_model / 2));
        grow(s.moeTcSfaBank,     mo::swizzledBlockScaleBytes(maxPad, d_model / 16));
        grow(s.moeTcABank2,      maxPad * (n_ff_exp / 2));
        grow(s.moeTcSfaBank2,    mo::swizzledBlockScaleBytes(maxPad, n_ff_exp / 16));
        grow(s.moeTcBanksScratch,
             _ops.moeGroupedGemmNvfp4TcBanksScratchBytes(nExperts));

        auto* const padOffset   = s.moeTcPadOffset.as<std::int32_t>();
        auto* const contigToPad = s.moeTcContigToPad.as<std::int32_t>();
        auto* const padAsn      = s.moeTcPadAsn.as<std::int32_t>();
        float* const xPad    = s.moeTcXPad.as<float>();
        float* const gatePad = s.moeTcGatePad.as<float>();
        float* const upPad   = s.moeTcUpPad.as<float>();
        float* const downPad = s.moeTcDownPad.as<float>();
        auto* const aBank    = s.moeTcABank.as<unsigned char>();
        auto* const sfaBank  = s.moeTcSfaBank.as<unsigned char>();
        auto* const aBank2   = s.moeTcABank2.as<unsigned char>();
        auto* const sfaBank2 = s.moeTcSfaBank2.as<unsigned char>();
        void* const banksScratch     = s.moeTcBanksScratch.get();
        const std::size_t banksBytes = s.moeTcBanksScratch.bytes();

        // padded row maps (device only)
        _ops.moePadOffsetsAsync(expOffset, padOffset, nExperts);
        _ops.moeContigToPadAsync(expOffset, padOffset, contigToPad, nExperts, R);
        _ops.moeIndexGatherI32Async(asnToRow, contigToPad, padAsn, nAsn);

        // spread gathered rows -> padded slots; act-quant (SF pre-zeroed).
        _ops.moeRowsScatterF32Async(xComp, contigToPad, xPad, R, d_model);
        _ops.moeZeroBytesAsync(sfaBank, mo::swizzledBlockScaleBytes(maxPad, d_model / 16));
        _ops.moeActQuantNvfp4Async(xPad, aBank, sfaBank, 1.0F, maxPad, d_model);

        // gate + up: N=n_ff_exp, K=d_model. alpha[e] = weight global (folds the
        // per-expert global back in; act gscale=1).
        _ops.moeGroupedGemmNvfp4TcBanksAsync(
            nExperts, n_ff_exp, d_model, expOffset, padOffset, aBank, sfaBank,
            gateExps.tcNibblePtr, gateExps.tcSfbPtr,
            static_cast<const float*>(gateExps.tcGlobalsPtr), gatePad,
            banksScratch, banksBytes);
        _ops.moeGroupedGemmNvfp4TcBanksAsync(
            nExperts, n_ff_exp, d_model, expOffset, padOffset, aBank, sfaBank,
            upExps.tcNibblePtr, upExps.tcSfbPtr,
            static_cast<const float*>(upExps.tcGlobalsPtr), upPad,
            banksScratch, banksBytes);

        _ops.siluMulAsync(gatePad, upPad, maxPad * n_ff_exp);  // silu(gate)*up

        // act-quant the intermediate -> down GEMM: N=d_model, K=n_ff_exp.
        _ops.moeZeroBytesAsync(sfaBank2, mo::swizzledBlockScaleBytes(maxPad, n_ff_exp / 16));
        _ops.moeActQuantNvfp4Async(gatePad, aBank2, sfaBank2, 1.0F, maxPad, n_ff_exp);
        _ops.moeGroupedGemmNvfp4TcBanksAsync(
            nExperts, d_model, n_ff_exp, expOffset, padOffset, aBank2, sfaBank2,
            downExps.tcNibblePtr, downExps.tcSfbPtr,
            static_cast<const float*>(downExps.tcGlobalsPtr), downPad,
            banksScratch, banksBytes);

        // scatter padded expert output back to token order (routed sum).
        _ops.moeScatterExpertOutAsync(downPad, padAsn, kwSlot, moeAccumBuf,
                                      d_model, nSeq, K);
    } else if (deviceDrivenGrouped) {
        // --- Option 2: fully device-driven grouped GEMM (Sub-Step E) -------
        // moe_group_tiles builds a compact per-tile (expert, row-range)
        // schedule on the device; ONE moe_grouped_gemm_nvfp4blk launch per
        // projection consumes it, reading the tile assignment on the device.
        // No expOffset D2H, no per-expert host loop — nothing crosses to the
        // host, so the stream is never drained mid-layer (the killer that
        // made the host-driven path lose to fused-K on GB10).
        const std::size_t tileM    = 16;                 // == GEMM_MAX_M
        const std::size_t maxTiles = (R + tileM - 1) / tileM + nExperts;
        auto* const tileExpert = s.moeGroupTileExpert.as<std::int32_t>();
        auto* const tileRow0   = s.moeGroupTileRow0.as<std::int32_t>();
        auto* const tileRows   = s.moeGroupTileRows.as<std::int32_t>();
        auto* const tileCount  = s.moeGroupTileCount.as<std::int32_t>();

        _ops.moeGroupTilesAsync(expOffset, tileExpert, tileRow0, tileRows,
                                tileCount, nExperts, maxTiles, tileM);

        const auto* const gateBase =
            static_cast<const unsigned char*>(gateExps.usmPtr);
        const auto* const upBase =
            static_cast<const unsigned char*>(upExps.usmPtr);
        const auto* const downBase =
            static_cast<const unsigned char*>(downExps.usmPtr);

        // gate/up: weight [nExperts][n_ff_exp][d_model] (N=n_ff_exp, K=d_model)
        _ops.moeGroupedGemmNvfp4Async(xComp, gateBase, gateComp,
                                      tileExpert, tileRow0, tileRows,
                                      d_model, n_ff_exp, maxTiles);
        _ops.moeGroupedGemmNvfp4Async(xComp, upBase, upComp,
                                      tileExpert, tileRow0, tileRows,
                                      d_model, n_ff_exp, maxTiles);
        _ops.siluMulAsync(gateComp, upComp, R * n_ff_exp);      // silu(gate)*up
        // down: weight [nExperts][d_model][n_ff_exp] (N=d_model, K=n_ff_exp)
        _ops.moeGroupedGemmNvfp4Async(gateComp, downBase, downComp,
                                      tileExpert, tileRow0, tileRows,
                                      n_ff_exp, d_model, maxTiles);
    } else {
        // --- Option 1: host-driven grouped (correct but slower on GB10) ----
        // The per-expert launch bounds are the only thing that must cross to
        // the host — one small D2H (nExperts+1 ints) per MoE layer. This
        // stream drain is exactly why Option 1 loses; the device-driven
        // branch above avoids it.
        _groupOffsetHost.resize(nExperts + 1);
        _ops.flush();
        _ops.readbackToHost(_groupOffsetHost.data(), expOffset,
                            (nExperts + 1) * sizeof(std::int32_t));

        // Per-expert byte strides (separate banks, one block per expert).
        std::size_t bytesGate = 0, bytesUp = 0, bytesDown = 0;
        if (gateExps.type == core::gguf::GgmlType::BF16) {
            bytesGate = n_ff_exp * d_model * sizeof(std::uint16_t);
            bytesUp   = n_ff_exp * d_model * sizeof(std::uint16_t);
            bytesDown = d_model * n_ff_exp * sizeof(std::uint16_t);
        } else {
            const auto [geGate, gbGate] = moeBlockGeom(gateExps.type);
            const auto [geUp,   gbUp]   = moeBlockGeom(upExps.type);
            const auto [geDown, gbDown] = moeBlockGeom(downExps.type);
            if (geGate == 0 || geUp == 0 || geDown == 0) {
                throw std::runtime_error(
                    "Qwen35MoeBackend::runMoeFfnGrouped: expert weight type(s) "
                    "not in QuantType registry");
            }
            bytesGate = n_ff_exp * ((d_model / geGate) * gbGate);
            bytesUp   = n_ff_exp * ((d_model / geUp)   * gbUp);
            bytesDown = d_model * ((n_ff_exp / geDown) * gbDown);
        }
        const auto* const gateBase = static_cast<const std::uint8_t*>(gateExps.usmPtr);
        const auto* const upBase   = static_cast<const std::uint8_t*>(upExps.usmPtr);
        const auto* const downBase = static_cast<const std::uint8_t*>(downExps.usmPtr);

        // --- one dense GEMM per expert over its M=count[e] grouped rows ----
        // Each expert weight is read once per 16-row GEMM chunk. Experts with
        // no routed tokens are skipped (an M=0 launch is pure overhead).
        for (std::size_t e = 0; e < nExperts; ++e) {
            const std::int32_t off = _groupOffsetHost[e];
            const std::int32_t end = _groupOffsetHost[e + 1];
            const std::size_t  Me  = static_cast<std::size_t>(end - off);
            if (Me == 0) {
                continue;
            }
            const float* xE    = xComp    + static_cast<std::size_t>(off) * d_model;
            float*       gateE = gateComp + static_cast<std::size_t>(off) * n_ff_exp;
            float*       upE   = upComp   + static_cast<std::size_t>(off) * n_ff_exp;
            float*       downE = downComp + static_cast<std::size_t>(off) * d_model;

            _gmm.matmulAsync(gateExps.type, gateBase + e * bytesGate,
                             n_ff_exp, d_model, xE, Me, gateE, matmulScratch);
            _gmm.matmulAsync(upExps.type, upBase + e * bytesUp,
                             n_ff_exp, d_model, xE, Me, upE, matmulScratch);
            _ops.siluMulAsync(gateE, upE, Me * n_ff_exp);      // silu(gate)*up
            _gmm.matmulAsync(downExps.type, downBase + e * bytesDown,
                             d_model, n_ff_exp, gateE, Me, downE, matmulScratch);
        }
    }

    // --- scatter the grouped output back to token order (routed sum) -------
    // Overwrites moeAccumBuf (no pre-zero needed); the shared expert is added
    // after, exactly as runMoeFfnBatched does. The FP4-TC path already
    // scattered its padded output above.
    if (!tcGrouped) {
        _ops.moeScatterExpertOutAsync(downComp, asnToRow, kwSlot, moeAccumBuf,
                                      d_model, nSeq, K);
    }

    // --- shared expert (always-on) + sigmoid gate — identical to batched ---
    const auto* upShexp = w.findBlock(blockIdx, "ffn_up_shexp.weight");
    if (upShexp != nullptr) {
        const auto& gateShexp = requireBlock(w, blockIdx, "ffn_gate_shexp.weight");
        const auto& downShexp = requireBlock(w, blockIdx, "ffn_down_shexp.weight");
        const auto& routerSh  = requireBlock(w, blockIdx, "ffn_gate_inp_shexp.weight");
        const std::size_t n_ff_shexp = gateShexp.dimensions.size() >= 2
                                           ? gateShexp.dimensions[1] : 0;
        if (n_ff_shexp == 0) {
            throw std::runtime_error(
                "Qwen35MoeBackend::runMoeFfnGrouped: ffn_gate_shexp has "
                "unexpected shape");
        }
        float* const expertOutBuf = s.expertOutBuf.as<float>();
        {
            compute::UnorderedScope u{_ops};
            _gmm.matmulAsync(gateShexp.type, gateShexp.usmPtr, n_ff_shexp, d_model,
                             moeInput, nSeq, gateOutBuf, matmulScratch);
            _gmm.matmulAsync(upShexp->type, upShexp->usmPtr, n_ff_shexp, d_model,
                             moeInput, nSeq, upOutBuf, matmulScratch);
        }
        _ops.siluMulAsync(gateOutBuf, upOutBuf, nSeq * n_ff_shexp);
        _gmm.matmulAsync(downShexp.type, downShexp.usmPtr, d_model, n_ff_shexp,
                         gateOutBuf, nSeq, expertOutBuf, matmulScratch);

        float* const gateScalar = s.scoreScratch.as<float>();
        _gmm.matmulAsync(routerSh.type, routerSh.usmPtr, 1, d_model,
                         moeInput, nSeq, gateScalar, matmulScratch);
        _ops.sigmoidGateMulAsync(expertOutBuf, gateScalar, nSeq, d_model,
                                 /*gateDim=*/1);

        _ops.addResidualAsync(moeAccumBuf, expertOutBuf, nSeq * d_model);
    }
}

void Qwen35MoeBackend::runBlockBatched(std::size_t             blockIdx,
                                       float*                  x,
                                       const BatchedDecodeCtx& ctx,
                                       BlockBuffers&           s) {
    if (_config.isRecurrentLayer(blockIdx)) {
        runLinearBlockBatched(blockIdx, x, ctx, s);
    } else {
        runFullAttentionBlockBatched(blockIdx, x, ctx, s);
    }
}

void Qwen35MoeBackend::runFullAttentionBlockBatched(
        std::size_t blockIdx, float* x, const BatchedDecodeCtx& ctx,
        BlockBuffers& s, std::size_t kvPoolLayer) {
    const std::size_t nSeq = ctx.nSeq;
    if (nSeq == 0) {
        return;
    }
    if (ctx.pool == nullptr) {
        throw std::runtime_error(
            "runFullAttentionBlockBatched: null PagedKvPool in context");
    }
    const std::size_t denseLayer =
        (kvPoolLayer == std::numeric_limits<std::size_t>::max())
            ? _fullAttnDense[blockIdx]
            : kvPoolLayer;

    const auto& w        = _weights;
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
    const float       eps      = _config.rmsNormEps;

    float* const normBuf    = s.normBuf.as<float>();
    float* const qGateFused = s.qGateFused.as<float>();
    float* const qBuf       = s.qBuf.as<float>();
    float* const gateBuf    = s.gateScratch.as<float>();
    float* const kProj      = s.kvKFp32Scratch.as<float>();   // [nSeq, kv_dim]
    float* const vProj      = s.kvVFp32Scratch.as<float>();   // [nSeq, kv_dim]
    float* const attnOut    = s.attnOut.as<float>();
    float* const projOut    = s.projOut.as<float>();
    float* const mmScratch  = s.matmulScratch.as<float>();

    // --- pre-attention RMSNorm (nSeq rows) ---------------------------
    _ops.rmsNormAsync(x, nSeq, d_model,
                      static_cast<const float*>(attnNorm.usmPtr), eps, normBuf);

    // --- Q(+gate) / K / V projections (M = nSeq) ---------------------
    {
        compute::UnorderedScope u{_ops};
        _gmm.matmulAsync(qW.type, qW.usmPtr, 2 * q_dim, d_model,
                         normBuf, nSeq, qGateFused, mmScratch);
        _gmm.matmulAsync(kW.type, kW.usmPtr, kv_dim, d_model,
                         normBuf, nSeq, kProj, mmScratch);
        _gmm.matmulAsync(vW.type, vW.usmPtr, kv_dim, d_model,
                         normBuf, nSeq, vProj, mmScratch);
    }

    _ops.splitHeadPairAsync(qGateFused, qBuf, gateBuf, nSeq, nHeads, head_dim);

    // --- QK-norm (per-head RMS over head_dim) ------------------------
    // In place on the compact q/k buffers — the paged path writes K into
    // the pool below (not a contiguous cache), so unlike the single-seq
    // rmsNormQkvAsync there is no fused cache write here. Per-row rmsnorm
    // is in-place safe (each row is independent).
    _ops.rmsNormAsync(qBuf,  nSeq * nHeads,   head_dim,
                      static_cast<const float*>(qNorm.usmPtr), eps, qBuf);
    _ops.rmsNormAsync(kProj, nSeq * nKvHeads, head_dim,
                      static_cast<const float*>(kNorm.usmPtr), eps, kProj);
    // The single-session rmsNormQkv also RMS-normalises V per head (over
    // head_dim, with NO learned weight — kernels/cuda/llm/rmsnorm_qkv.cu V
    // branch). The batched path builds q/k/v separately, so replicate the
    // V normalisation here; without it V enters attention un-normalised and
    // the whole full-attention output is off by V's per-head 1/rms factor.
    _ops.rmsNormNoWeightAsync(vProj, nSeq * nKvHeads, head_dim, eps, vProj);

    // --- IMRoPE on Q and K (per-seq startPos, one token per seq) ------
    // writeOffsetStride MUST be 0 here: the batched mrope writes at
    // x_base + seq*xSeqStride + startPos*writeOffsetStride, which is the
    // KV-cache-absolute-position pattern. Our q/k live in COMPACT per-seq
    // buffers (one row per sequence), so the token position must not shift
    // the write — only the rotation angle depends on startPos (pos =
    // startPos + p inside the kernel). A non-zero stride here walks off the
    // compact buffer as startPos grows (the pos=2 OOB).
    {
        compute::UnorderedScope u{_ops};
        _ops.mropeInPlaceBatchedAsync(qBuf, nSeq, q_dim, /*seqLen=*/1,
                                      nHeads, head_dim, ctx.startPosDev,
                                      _config.ropeFreqBase, _ropeSections,
                                      /*writeOffsetStride=*/0,
                                      runtime::KvDtype::F32);
        _ops.mropeInPlaceBatchedAsync(kProj, nSeq, kv_dim, /*seqLen=*/1,
                                      nKvHeads, head_dim, ctx.startPosDev,
                                      _config.ropeFreqBase, _ropeSections,
                                      /*writeOffsetStride=*/0,
                                      runtime::KvDtype::F32);
    }

    // --- scatter this step's K/V into the paged pool -----------------
    for (std::size_t seq = 0; seq < nSeq; ++seq) {
        ctx.pool->writeToken(_ops, denseLayer,
                             ctx.writeBlockId[seq],
                             static_cast<std::size_t>(ctx.writeSlot[seq]),
                             kProj + seq * kv_dim,
                             vProj + seq * kv_dim);
    }

    // --- paged GQA attention (block-table indirection) ---------------
    const float attnScale = _config.attentionScaleFor(head_dim);
    _ops.pagedAttentionDecodeV1Async(
        attnOut, qBuf, ctx.pool->keyPool(denseLayer),
        ctx.pool->valuePool(denseLayer), ctx.blockTablesDev, ctx.seqLensDev,
        nSeq, nHeads, nKvHeads, head_dim, ctx.pool->blockSize(),
        ctx.maxBlocksPerSeq, attnScale, /*softcap=*/0.0f);

    // --- output gate + O projection + attn residual ------------------
    _ops.sigmoidGateMulAsync(attnOut, gateBuf, nSeq, q_dim, /*gateDim=*/q_dim);
    _gmm.matmulAsync(oW.type, oW.usmPtr, d_model, q_dim,
                     attnOut, nSeq, projOut, mmScratch);
    _ops.addResidualAsync(x, projOut, nSeq * d_model);

    // --- post-attention norm -> batched MoE -> FFN residual ----------
    _ops.rmsNormAsync(x, nSeq, d_model,
                      static_cast<const float*>(attnPost.usmPtr), eps, normBuf);
    runMoeFfnBatched(blockIdx, normBuf, nSeq, ctx.expIdxSlot, ctx.kwSlot, s);
    _ops.addResidualAsync(x, s.moeAccumBuf.as<float>(), nSeq * d_model);
}

void Qwen35MoeBackend::runLinearBlockBatched(
        std::size_t blockIdx, float* x, const BatchedDecodeCtx& ctx,
        BlockBuffers& s) {
    const std::size_t nSeq = ctx.nSeq;
    if (nSeq == 0) {
        return;
    }

    const auto& w        = _weights;
    const auto& attnNorm = requireBlock(w, blockIdx, "attn_norm.weight");
    const auto& qkvW     = requireBlock(w, blockIdx, "attn_qkv.weight");
    const auto& gateW    = requireBlock(w, blockIdx, "attn_gate.weight");
    const auto& betaW    = requireBlock(w, blockIdx, "ssm_beta.weight");
    const auto& alphaW   = requireBlock(w, blockIdx, "ssm_alpha.weight");
    const auto& ssmA     = requireBlock(w, blockIdx, "ssm_a");
    const auto& ssmDt    = requireBlock(w, blockIdx, "ssm_dt.bias");
    const auto& convW    = requireBlock(w, blockIdx, "ssm_conv1d.weight");
    const auto& ssmNormW = requireBlock(w, blockIdx, "ssm_norm.weight");
    const auto& ssmOutW  = requireBlock(w, blockIdx, "ssm_out.weight");
    const auto& attnPost = requireBlock(w, blockIdx, "post_attention_norm.weight");

    const std::size_t d_model        = s.d_model;
    const std::size_t S              = _config.ssmStateSize;
    const std::size_t hK             = _config.ssmNumKHeads();
    const std::size_t hV             = _config.ssmNumVHeads();
    const std::size_t valueDim       = _config.ssmInnerSize;      // hV * S
    const std::size_t convDim        = _config.ssmConvDim();
    const std::size_t dConv          = _config.ssmConvKernel;
    const std::size_t keyDim         = S * hK;
    const std::size_t stateElems     = _config.ssmStateElemsPerLayer();
    const std::size_t convStateElems = _config.ssmConvStateElemsPerLayer();
    const std::size_t stateRows      = (dConv > 0 ? dConv - 1 : 0);
    const float       eps            = _config.rmsNormEps;

    float* const normBuf   = s.normBuf.as<float>();
    float* const qkvMixed  = s.ssmQkvMixed.as<float>();   // [nSeq, convDim] (also conv out)
    float* const convInput = s.ssmConvInput.as<float>();  // [nSeq, dConv, convDim] (serving-sized)
    float* const zBuf      = s.ssmZ.as<float>();
    float* const qBuf      = s.ssmQ.as<float>();
    float* const kBuf      = s.ssmK.as<float>();
    float* const vBuf      = s.ssmV.as<float>();
    float* const deltaOut  = s.ssmDeltaOut.as<float>();
    float* const alphaBuf  = s.ssmAlpha.as<float>();
    float* const betaBuf   = s.ssmBeta.as<float>();
    float* const gateBuf   = s.ssmGate.as<float>();
    float* const projOut   = s.projOut.as<float>();
    float* const mmScratch = s.matmulScratch.as<float>();

    // Per-sequence recurrent + conv state (SsmState[slabNSeq]): the per-layer
    // slab is [slabNSeq, stateElems] / [slabNSeq, convStateElems], so the
    // layer base steps by slabNSeq*elems (== SsmState::stateLayerStride) and
    // sequence s sits at + s*elems. The layer stride MUST use the slab's
    // ALLOCATED nSeq, not the runtime batch `nSeq` — under continuous
    // batching the active count varies below the allocation, and using the
    // runtime nSeq would offset every layer into the wrong slice
    // (M-Cuda.Batch D2e.2). Falls back to nSeq when unset (== generateBatch,
    // where slab nSeq == runtime nSeq, so the two agree).
    const std::size_t slabNSeq = (s.ssmSlabNSeq != 0) ? s.ssmSlabNSeq : nSeq;
    float* const stateBase = s.ssmStatePtr     + blockIdx * (slabNSeq * stateElems);
    float* const convBase  = s.ssmConvStatePtr + blockIdx * (slabNSeq * convStateElems);

    // --- pre-attention RMSNorm (nSeq rows) ---------------------------
    _ops.rmsNormAsync(x, nSeq, d_model,
                      static_cast<const float*>(attnNorm.usmPtr), eps, normBuf);

    // --- projections (M = nSeq) --------------------------------------
    {
        compute::UnorderedScope u{_ops};
        _gmm.matmulAsync(qkvW.type,  qkvW.usmPtr,  convDim,  d_model, normBuf, nSeq, qkvMixed, mmScratch);
        _gmm.matmulAsync(gateW.type, gateW.usmPtr, valueDim, d_model, normBuf, nSeq, zBuf,     mmScratch);
        _gmm.matmulAsync(betaW.type, betaW.usmPtr, hV,       d_model, normBuf, nSeq, betaBuf,  mmScratch);
        _gmm.matmulAsync(alphaW.type,alphaW.usmPtr,hV,       d_model, normBuf, nSeq, alphaBuf, mmScratch);
    }

    // beta = sigmoid(beta); gLog = ssm_a * softplus(alpha + ssm_dt).
    _ops.sigmoidInPlaceAsync(betaBuf, nSeq * hV);
    _ops.deltanetGateAsync(alphaBuf,
                           static_cast<const float*>(ssmA.usmPtr),
                           static_cast<const float*>(ssmDt.usmPtr),
                           gateBuf, nSeq, hV);

    // --- causal conv1d + silu (per-seq rolling state) ----------------
    // convInput[seq] = [convState[seq] (stateRows) | qkvMixed[seq] (1 row)].
    // Serving BlockBuffers must size ssmConvInput to nSeq*dConv*convDim.
    const std::size_t convInBytes  = convStateElems * sizeof(float);
    const std::size_t qkvRowBytes  = convDim * sizeof(float);
    for (std::size_t seq = 0; seq < nSeq; ++seq) {
        float* const cvState = convBase + seq * convStateElems;
        float* const cvIn    = convInput + seq * dConv * convDim;
        if (ctx.isSeqStart != nullptr && ctx.isSeqStart[seq] != 0) {
            _ops.mulScalarAsync(cvState, 0.0F, convStateElems);
        }
        _ops.appendMemoryCopy(cvIn, cvState, convInBytes);
        _ops.appendMemoryCopy(cvIn + stateRows * convDim,
                              qkvMixed + seq * convDim, qkvRowBytes);
    }
    _ops.causalConv1dSiluBatchedAsync(convInput,
                                      static_cast<const float*>(convW.usmPtr),
                                      qkvMixed, nSeq, /*T=*/1, convDim, dConv);
    // Save each sequence's trailing stateRows rows as the next conv state.
    for (std::size_t seq = 0; seq < nSeq; ++seq) {
        float* const cvState = convBase + seq * convStateElems;
        float* const cvIn    = convInput + seq * dConv * convDim;
        _ops.appendMemoryCopy(cvState, cvIn + /*T=*/1 * convDim, convInBytes);
    }

    // --- split conv into q/k/v (+ GQA repeat H_k -> H_v) -------------
    _ops.gatherHeadsFromChannelsAsync(qkvMixed, qBuf, nSeq, 0,          hK, hV, S, convDim);
    _ops.gatherHeadsFromChannelsAsync(qkvMixed, kBuf, nSeq, keyDim,     hK, hV, S, convDim);
    _ops.gatherHeadsFromChannelsAsync(qkvMixed, vBuf, nSeq, 2 * keyDim, hV, hV, S, convDim);

    // --- L2-norm q,k over head_dim -----------------------------------
    _ops.l2NormInPlaceAsync(qBuf, nSeq * hV, S, eps);
    _ops.l2NormInPlaceAsync(kBuf, nSeq * hV, S, eps);

    // --- gated delta-rule recurrence (persistent per-seq state) ------
    for (std::size_t seq = 0; seq < nSeq; ++seq) {
        if (ctx.isSeqStart != nullptr && ctx.isSeqStart[seq] != 0) {
            _ops.mulScalarAsync(stateBase + seq * stateElems, 0.0F, stateElems);
        }
    }
    _ops.gatedDeltaNetRecurrentBatchedAsync(qBuf, kBuf, vBuf, gateBuf, betaBuf,
                                            stateBase, deltaOut, nSeq,
                                            /*T=*/1, hV, S);

    // --- gated output norm: ssm_norm(out) * silu(z) ------------------
    _ops.rmsNormAsync(deltaOut, nSeq * hV, S,
                      static_cast<const float*>(ssmNormW.usmPtr), eps, qBuf);
    _ops.siluMulAsync(zBuf, qBuf, nSeq * valueDim);

    // --- output projection ssm_out -----------------------------------
    _gmm.matmulAsync(ssmOutW.type, ssmOutW.usmPtr, d_model, valueDim,
                     zBuf, nSeq, projOut, mmScratch);
    _ops.addResidualAsync(x, projOut, nSeq * d_model);

    // --- post-attn norm -> batched MoE -> FFN residual ---------------
    _ops.rmsNormAsync(x, nSeq, d_model,
                      static_cast<const float*>(attnPost.usmPtr), eps, normBuf);
    runMoeFfnBatched(blockIdx, normBuf, nSeq, ctx.expIdxSlot, ctx.kwSlot, s);
    _ops.addResidualAsync(x, s.moeAccumBuf.as<float>(), nSeq * d_model);
}

} // namespace mimirmind::runtime::arch