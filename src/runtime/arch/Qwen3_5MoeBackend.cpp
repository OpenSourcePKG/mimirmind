// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/arch/Qwen3_5MoeBackend.hpp"

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
#include <vector>
#include <cstdlib>
#include <fstream>
#include <span>
#include <stdexcept>
#include <string>

namespace mimirmind::runtime::arch {

namespace {

// Track B: token threshold at/above which a shared-expert projection is routed
// through the FP4-TC grouped GEMM (nExp=1) instead of the dense blocked kernel.
// Below it (decode, small chunks) the blocked kGemmMaxM=16 kernel wins on
// launch overhead; the TC path only pays off once the M/16 weight re-reads it
// avoids dominate. Matches the routed-MoE "prefill takes TC" M>=64 crossover.
constexpr std::size_t kShexpTcMinM = 64;

// Block geometry (elements, bytes) for an expert weight type. NVFP4_BLK is a
// runtime-only blocked format (32-element super-blocks of 20 bytes) not in the
// QuantType registry, so handle it explicitly; everything else goes through the
// registry. Returns {0,0} for an unregistered non-NVFP4 type (caller checks).
inline std::pair<std::size_t, std::size_t>
moeBlockGeom(core::gguf::GgmlType type) {
    if (type == core::gguf::GgmlType::NVFP4_BLK) {
        return {32, 20};
    }
    if (type == core::gguf::GgmlType::NVFP4_TC) {
        return {32, 16};   // plain E2M1 nibbles: 16 B / 32 elems (decode uses TC banks)
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
            "Qwen3_5MoeBackend: block " + std::to_string(blockIdx) +
            " missing tensor '" + std::string(suffix) + "'");
    }
    return *t;
}

// M-dependent dense-weight pick (M-Cuda dense-FP8-lowM). At low batch
// (seqCount <= maxT) prefer the blocked-FP8 E4M3 ".fp8" variant — decode is
// memory-bound there and FP8 halves the always-read dense traffic (+~15%
// single-request, bit-coherent). At high batch use the BF16 copy — compute-
// bound, where the TF32 tensor-core GEMM wins. Falls back to BF16 when maxT==0
// or the model carries no ".fp8" variant (dual-copy not loaded).
const core::gguf::GgufTensor&
pickDense(const core::gguf::WeightsMap& w, std::size_t blockIdx,
          std::string_view suffix, std::size_t seqCount, std::size_t maxT) {
    if (maxT > 0 && seqCount <= maxT) {
        std::string fp8Suffix(suffix);
        fp8Suffix += ".fp8";
        if (const auto* v = w.findBlock(blockIdx, fp8Suffix)) {
            return *v;
        }
    }
    return requireBlock(w, blockIdx, suffix);
}

} // namespace

Qwen3_5MoeBackend::Qwen3_5MoeBackend(const model::LlmConfig&       config,
                                     const core::gguf::WeightsMap& weights,
                                     const model::FusedQkvWeights* fusedQkv,
                                     compute::ComputeOps&          ops,
                                     compute::ComputeMatmul&       gmm,
                                     runtime::OpProfiler&          opProfiler,
                                     bool                          moeGroupEnabled,
                                     bool                          moeFusedDownEnabled)
    : Qwen3_5Backend(config, weights, fusedQkv, ops, gmm, opProfiler) {
    _moeGroupEnabled     = moeGroupEnabled;
    _moeFusedDownEnabled = moeFusedDownEnabled;
    // Routed-MoE env flags (moved out of the base ctor — they only drive the
    // routed-expert / shared-expert FFN path this subclass owns).
    if (const char* g = std::getenv("MIMIRMIND_GROUPED_MOE")) {
        // "1" host-driven grouped, "2" device-driven grouped, "3" FP4-TC grouped.
        _moeGroupedPrefill      = (g[0] == '1' && g[1] == '\0');
        _moeGroupedDeviceDriven = (g[0] == '2' && g[1] == '\0');
        _moeGroupedTc           = (g[0] == '3' && g[1] == '\0');
        if (_moeGroupedDeviceDriven || _moeGroupedTc) {
            _moeGroupedPrefill = true;
        }
    }
    if (const char* d = std::getenv("MIMIRMIND_GROUPED_MOE_DECODE")) {
        _moeGroupedDecode = !(d[0] == '0' && d[1] == '\0');
    }
    if (const char* t = std::getenv("MIMIRMIND_GROUPED_MOE_DECODE_TC")) {
        _moeGroupedDecodeTc = (t[0] == '1' && t[1] == '\0');
    }
    if (const char* st = std::getenv("MIMIRMIND_SHEXP_TC")) {
        _shexpTc = !(st[0] == '0' && st[1] == '\0');
    }
    if (const char* dm = std::getenv("MIMIRMIND_NVFP4_DEINT")) {
        _useDeintMoe = (dm[0] == '1' && dm[1] == '\0');
    }
    if (const char* nb = std::getenv("MIMIRMIND_MOE_M1NB")) {
        _useM1nb = (nb[0] == '1' && nb[1] == '\0');
    }
    _q8Dp4a               = (std::getenv("MIMIRMIND_Q8_DP4A") != nullptr);
    _moeDeviceTopKEnabled = (std::getenv("MIMIRMIND_MOE_DEVICE_TOPK") != nullptr);
    // 5.22 OEA — batch-aware routing (default OFF, lossy). MINSHARE = an expert
    // shared by >= this many batch tokens is kept; singletons drop. MAXBATCH
    // caps OEA to decode-sized batches (prefill chunks are much larger and must
    // NOT be OEA-rerouted — the union there is legitimately near-full).
    _moeOeaEnabled = (std::getenv("MIMIRMIND_MOE_OEA") != nullptr);
    if (const char* ms = std::getenv("MIMIRMIND_MOE_OEA_MINSHARE")) {
        const int v = std::atoi(ms);
        if (v >= 1) {
            _moeOeaMinShare = v;
        }
    }
    if (const char* mb = std::getenv("MIMIRMIND_MOE_OEA_MAXBATCH")) {
        const int v = std::atoi(mb);
        if (v >= 2) {
            _moeOeaMaxBatch = static_cast<std::size_t>(v);
        }
    }
}

// Polymorphic FFN seam (5.20): routed top-K experts + gated shared expert. When
// the prefill routing hook is set (setPrefillMoeScratch) and T>1, route through
// the amortised batched / grouped fused-K path; otherwise the per-token path.
void Qwen3_5MoeBackend::runFfn(std::size_t   blockIdx,
                               const float*  moeInput,
                               std::size_t   T,
                               BlockBuffers& s) {
    if (_prefillMoeExpIdx != nullptr && T > 1) {
        const auto* gExpsRoute =
            _weights.findBlock(blockIdx, "ffn_gate_exps.weight");
        const bool tcRoute = gExpsRoute != nullptr
            && gExpsRoute->tcNibblePtr != nullptr;
        if (tcRoute || _moeGroupedPrefill) {
            runMoeFfnGrouped(blockIdx, moeInput, T, _prefillMoeExpIdx,
                             _prefillMoeKw, s);
        } else {
            runMoeFfnBatched(blockIdx, moeInput, T, _prefillMoeExpIdx,
                             _prefillMoeKw, s);
        }
    } else {
        runMoeFfn(blockIdx, moeInput, T, s);
    }
}

void Qwen3_5MoeBackend::runMtpDraftStep(const float*  hidden,
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

void Qwen3_5MoeBackend::runMtpDraftStepBatched(
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

void Qwen3_5MoeBackend::sharedExpertTcGemm(std::size_t N, std::size_t K,
                                          const float* X, std::size_t M,
                                          const void* wNib, const void* wSfb,
                                          const float* wGlob, float* Y,
                                          BlockBuffers& s) {
    namespace mo = core::modelopt;
    // One group padded to a 128-row tile so its SFA sub-tensor is tile-aligned.
    const std::size_t padM = ((M + 127) / 128) * 128;
    auto grow = [&](compute::ComputeBuffer& buf, std::size_t bytes) {
        if (buf.bytes() < bytes) buf = _ops.allocate(bytes);
    };
    grow(s.shexpTcOffsets,      4 * sizeof(std::int32_t));
    grow(s.shexpTcABank,        padM * (K / 2));
    grow(s.shexpTcSfaBank,      mo::swizzledBlockScaleBytes(padM, K / 16));
    grow(s.shexpTcOutPad,       padM * N * sizeof(float));
    grow(s.shexpTcBanksScratch, _ops.moeGroupedGemmNvfp4TcBanksScratchBytes(1));

    auto* const offsets = s.shexpTcOffsets.as<std::int32_t>();
    auto* const aBank   = s.shexpTcABank.as<unsigned char>();
    auto* const sfaBank = s.shexpTcSfaBank.as<unsigned char>();
    float* const outPad = s.shexpTcOutPad.as<float>();

    // offsets = {expOffset[0..1]=0,M ; padOffset[0..1]=0,padM}. Pure function of
    // M, which is constant across a forward's blocks -> upload only when M
    // changes (once per forward), keeping the sync H2D off the per-block path.
    if (s.shexpTcOffM != static_cast<std::int32_t>(M)) {
        const std::int32_t off[4] = {0, static_cast<std::int32_t>(M),
                                     0, static_cast<std::int32_t>(padM)};
        _ops.uploadHostBytes(offsets, off, sizeof(off));
        s.shexpTcOffM = static_cast<std::int32_t>(M);
    }
    const std::int32_t* const expOffset = offsets;
    const std::int32_t* const padOffset = offsets + 2;

    // Quantise the M real rows into the swizzled aBank/sfaBank. The pad tail
    // [M,padM) keeps its zeroed SFA (scale 0 -> act 0), so those GEMM rows are 0
    // and discarded; gscale=1 since the weight global folds in via globalsBank.
    _ops.moeZeroBytesAsync(sfaBank, mo::swizzledBlockScaleBytes(padM, K / 16));
    _ops.moeActQuantNvfp4Async(X, aBank, sfaBank, 1.0F, M, K);

    _ops.moeGroupedGemmNvfp4TcBanksAsync(
        1, N, K, expOffset, padOffset, aBank, sfaBank, wNib, wSfb, wGlob,
        outPad, s.shexpTcBanksScratch.get(), s.shexpTcBanksScratch.bytes());

    // The single group's real rows are the first M rows of outPad ([padM, N],
    // row-major) -> the contiguous leading [M*N] block. Copy them to Y[M,N].
    _ops.appendMemoryCopy(Y, outPad, M * N * sizeof(float));
}

void Qwen3_5MoeBackend::runMoeFfn(std::size_t   blockIdx,
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
            "Qwen3_5MoeBackend: expert gate/gate_up tensor must be 3-D "
            "[n_embd, n_ff(*2), n_expert]");
    }
    if (fused && (gateSrc.dimensions[1] % 2) != 0) {
        throw std::runtime_error(
            "Qwen3_5MoeBackend: fused ffn_gate_up_exps ne1 must be even");
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
            "Qwen3_5MoeBackend: expert weight type(s) not in QuantType registry");
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
        if (gateType == core::gguf::GgmlType::NVFP4_TC) {
            // E-d.5 FP4-TC single-token decode = the batched TC kernels, nSeq=1.
            namespace mo = core::modelopt;
            const std::size_t sfbGu = mo::moeSwizzledScaleStride(n_ff_exp, d_model / 16);
            const std::size_t sfbDn = mo::moeSwizzledScaleStride(d_model, n_ff_exp / 16);
            _gmm.moeGateUpFusedKTcBatchedAsync(
                xt, gateSrc.tcNibblePtr, upSrc.tcNibblePtr,
                gateSrc.tcSfbPtr, upSrc.tcSfbPtr,
                static_cast<const float*>(gateSrc.tcGlobalsPtr),
                static_cast<const float*>(upSrc.tcGlobalsPtr),
                expIdxSlot, gateActAll, 1, d_model, n_ff_exp, K, sfbGu);
            _ops.mulScalarAsync(moeAccumBuf, 0.0F, d_model);
            _gmm.moeDownFusedKTcBatchedAsync(
                gateActAll, downExps.tcNibblePtr, downExps.tcSfbPtr,
                static_cast<const float*>(downExps.tcGlobalsPtr),
                expIdxSlot, kwSlot, moeAccumBuf, 1, n_ff_exp, d_model, K, sfbDn);
        } else {
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
        }
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
                "Qwen3_5MoeBackend: ffn_gate_shexp has unexpected shape");
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
        } else if (_shexpTc && T >= kShexpTcMinM &&
                   _ops.moeGroupedGemmNvfp4TcAvailable() &&
                   gateShexp.tcNibblePtr != nullptr &&
                   upShexp->tcNibblePtr  != nullptr &&
                   downShexp.tcNibblePtr != nullptr) {
            // Track B: prefill shared expert via the FP4-TC grouped GEMM
            // (nExp=1) -> no kGemmMaxM=16 weight re-read. gate/up: N=n_ff_shexp,
            // K=d_model; down: N=d_model, K=n_ff_shexp. Sequential (shared
            // shexpTc* scratch); the T==1 dp4a/fused shortcuts above are
            // decode-only and never reached at this M.
            sharedExpertTcGemm(n_ff_shexp, d_model, moeInput, T,
                               gateShexp.tcNibblePtr, gateShexp.tcSfbPtr,
                               static_cast<const float*>(gateShexp.tcGlobalsPtr),
                               gateOutBuf, s);
            sharedExpertTcGemm(n_ff_shexp, d_model, moeInput, T,
                               upShexp->tcNibblePtr, upShexp->tcSfbPtr,
                               static_cast<const float*>(upShexp->tcGlobalsPtr),
                               upOutBuf, s);
            _ops.siluMulAsync(gateOutBuf, upOutBuf, T * n_ff_shexp);
            sharedExpertTcGemm(d_model, n_ff_shexp, gateOutBuf, T,
                               downShexp.tcNibblePtr, downShexp.tcSfbPtr,
                               static_cast<const float*>(downShexp.tcGlobalsPtr),
                               expertOutBuf, s);
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

void Qwen3_5MoeBackend::runMoeFfnBatched(std::size_t    blockIdx,
                                        const float*   moeInput,
                                        std::size_t    nSeq,
                                        std::int32_t*  expIdxSlot,
                                        float*         kwSlot,
                                        BlockBuffers&  s) {
    namespace cmp = mimirmind::compute;
    _ops.profileSection("moe");

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
                d->type == core::gguf::GgmlType::NVFP4_TC ||
                d->type == core::gguf::GgmlType::BF16;
            // NVFP4_BLK is 32-element super-blocks (not 256); the other batched
            // types are 256-block. Require the matching alignment per type.
            const std::size_t blkAlign =
                (g->type == core::gguf::GgmlType::NVFP4_BLK || g->type == core::gguf::GgmlType::NVFP4_TC) ? 32 : 256;
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
            "Qwen3_5MoeBackend::runMoeFfnBatched: fused ffn_gate_up_exps layout "
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
            "Qwen3_5MoeBackend::runMoeFfnBatched: expert gate tensor must be 3-D "
            "[n_embd, n_ff, n_expert]");
    }
    const std::size_t n_ff_exp = gateExps.dimensions[1];

    const bool downBatchedOk =
        downExps.type == core::gguf::GgmlType::Q5_K ||
        downExps.type == core::gguf::GgmlType::Q6_K ||
        downExps.type == core::gguf::GgmlType::NVFP4_BLK ||
        downExps.type == core::gguf::GgmlType::NVFP4_TC ||
        downExps.type == core::gguf::GgmlType::BF16;
    const std::size_t blkAlign =
        (gateExps.type == core::gguf::GgmlType::NVFP4_BLK || gateExps.type == core::gguf::GgmlType::NVFP4_TC) ? 32 : 256;
    if (!_gmm.moeGateUpFusedKAvailable(gateExps.type) ||
        !downBatchedOk ||
        gateExps.type != upExps.type ||
        (d_model % blkAlign != 0) || (n_ff_exp % blkAlign != 0)) {
        throw std::runtime_error(
            "Qwen3_5MoeBackend::runMoeFfnBatched: batched fused-K MoE kernels "
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
                "Qwen3_5MoeBackend::runMoeFfnBatched: expert weight type(s) not "
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
    if (gateExps.type == core::gguf::GgmlType::NVFP4_TC) {
        // E-d.5 FP4-TC decode: read the plain nibbles (usmPtr) + swizzled SFB +
        // per-expert globals, no blocked bank.
        namespace mo = core::modelopt;
        const std::size_t sfbGu = mo::moeSwizzledScaleStride(n_ff_exp, d_model / 16);
        const std::size_t sfbDn = mo::moeSwizzledScaleStride(d_model, n_ff_exp / 16);
        _gmm.moeGateUpFusedKTcBatchedAsync(
            moeInput, gateExps.tcNibblePtr, upExps.tcNibblePtr,
            gateExps.tcSfbPtr, upExps.tcSfbPtr,
            static_cast<const float*>(gateExps.tcGlobalsPtr),
            static_cast<const float*>(upExps.tcGlobalsPtr),
            expIdxSlot, gateActAll, nSeq, d_model, n_ff_exp, K, sfbGu);
        _ops.mulScalarAsync(moeAccumBuf, 0.0F, nSeq * d_model);
        _gmm.moeDownFusedKTcBatchedAsync(
            gateActAll, downExps.tcNibblePtr, downExps.tcSfbPtr,
            static_cast<const float*>(downExps.tcGlobalsPtr),
            expIdxSlot, kwSlot, moeAccumBuf, nSeq, n_ff_exp, d_model, K, sfbDn);
    } else {
        _gmm.moeGateUpFusedKBatchedAsync(gateExps.type, moeInput, gateBase, upBase,
                                         expIdxSlot, gateActAll, nSeq,
                                         d_model, n_ff_exp, K, bytesGate, bytesUp);
        _ops.mulScalarAsync(moeAccumBuf, 0.0F, nSeq * d_model);
        _gmm.moeDownFusedKBatchedAsync(downExps.type, gateActAll, downBase,
                                       expIdxSlot, kwSlot, moeAccumBuf, nSeq,
                                       n_ff_exp, d_model, K, bytesDown);
    }

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
                "Qwen3_5MoeBackend::runMoeFfnBatched: ffn_gate_shexp has "
                "unexpected shape");
        }

        if (_shexpTc && nSeq >= kShexpTcMinM &&
            _ops.moeGroupedGemmNvfp4TcAvailable() &&
            gateShexp.tcNibblePtr != nullptr &&
            upShexp->tcNibblePtr  != nullptr &&
            downShexp.tcNibblePtr != nullptr) {
            // Track B: prefill shared expert via the FP4-TC grouped GEMM
            // (nExp=1) -> no kGemmMaxM=16 weight re-read. Sequential across the
            // three projections (they share the shexpTc* scratch).
            sharedExpertTcGemm(n_ff_shexp, d_model, moeInput, nSeq,
                               gateShexp.tcNibblePtr, gateShexp.tcSfbPtr,
                               static_cast<const float*>(gateShexp.tcGlobalsPtr),
                               gateOutBuf, s);
            sharedExpertTcGemm(n_ff_shexp, d_model, moeInput, nSeq,
                               upShexp->tcNibblePtr, upShexp->tcSfbPtr,
                               static_cast<const float*>(upShexp->tcGlobalsPtr),
                               upOutBuf, s);
            _ops.siluMulAsync(gateOutBuf, upOutBuf, nSeq * n_ff_shexp);
            sharedExpertTcGemm(d_model, n_ff_shexp, gateOutBuf, nSeq,
                               downShexp.tcNibblePtr, downShexp.tcSfbPtr,
                               static_cast<const float*>(downShexp.tcGlobalsPtr),
                               expertOutBuf, s);
        } else {
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
        }

        // Scalar gate per token: [nSeq, 1] -> sigmoid -> broadcast multiply.
        float* const gateScalar = s.scoreScratch.as<float>();
        _gmm.matmulAsync(routerSh.type, routerSh.usmPtr, 1, d_model,
                         moeInput, nSeq, gateScalar, matmulScratch);
        _ops.sigmoidGateMulAsync(expertOutBuf, gateScalar, nSeq, d_model,
                                 /*gateDim=*/1);

        _ops.addResidualAsync(moeAccumBuf, expertOutBuf, nSeq * d_model);
    }
}

void Qwen3_5MoeBackend::runMoeFfnGrouped(std::size_t    blockIdx,
                                        const float*   moeInput,
                                        std::size_t    nSeq,
                                        std::int32_t*  expIdxSlot,
                                        float*         kwSlot,
                                        BlockBuffers&  s,
                                        bool           preferBlocked) {
    const auto& w = _weights;
    _ops.profileSection("moe");

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
        downExps.type == core::gguf::GgmlType::NVFP4_TC ||
        downExps.type == core::gguf::GgmlType::BF16;
    const std::size_t blkAlign =
        (gateExps.type == core::gguf::GgmlType::NVFP4_BLK || gateExps.type == core::gguf::GgmlType::NVFP4_TC) ? 32 : 256;
    if (!_gmm.moeGateUpFusedKAvailable(gateExps.type) || !downOk ||
        gateExps.type != upExps.type ||
        (d_model % blkAlign != 0) || (n_ff_exp % blkAlign != 0)) {
        runMoeFfn(blockIdx, moeInput, nSeq, s);
        return;
    }

    // GD-a: the decode grouped path only supports the blocked NVFP4 bank (the
    // scalar device-driven grouped GEMM reads gateExps.usmPtr as a blocked
    // super-block bank). For any other layout (e.g. tc-only NVFP4_TC, where
    // usmPtr is the plain-nibble bank, or Q4_K/BF16) stay on the fused-K batched
    // path rather than dropping into the slow host-driven per-expert loop.
    if (preferBlocked &&
        (gateExps.type != core::gguf::GgmlType::NVFP4_BLK ||
         upExps.type   != core::gguf::GgmlType::NVFP4_BLK ||
         downExps.type != core::gguf::GgmlType::NVFP4_BLK)) {
        runMoeFfnBatched(blockIdx, moeInput, nSeq, expIdxSlot, kwSlot, s);
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
    _ops.profileSection("rt.gemm");   // route sub-breakdown (DECODE_PROFILE only)
    _gmm.matmulAsync(routerW.type, routerW.usmPtr, nExperts, d_model,
                     moeInput, nSeq, upOutBuf, matmulScratch);
    _ops.profileSection("rt.topk");
    _ops.moeTopKRouteDeviceAsync(upOutBuf, expIdxSlot, kwSlot,
                                 nSeq, nExperts, K, wScale);

    // --- 5.22 OEA: batch-aware reroute to cut the unique-expert union -------
    // Rewrites expIdxSlot/kwSlot in place (top-K among the batch's active-expert
    // set) so the grouped GEMM below loads fewer distinct expert weights. Decode
    // batches only: T==1 has no batch to piggyback on, and prefill chunks (T
    // large) must NOT be rerouted (their near-full union is legitimate), so cap
    // at _moeOeaMaxBatch. Lossy → env-gated, default OFF (MIMIRMIND_MOE_OEA).
    if (_moeOeaEnabled && nSeq > 1 && nSeq <= _moeOeaMaxBatch) {
        _ops.profileSection("rt.oea");
        _ops.moeOeaRerouteAsync(upOutBuf, expIdxSlot, kwSlot,
                                s.moeOeaActive.as<std::int32_t>(),
                                nSeq, nExperts, K, _moeOeaMinShare, wScale);
    }

    // --- group the T*K assignments by expert (device counting sort) --------
    _ops.profileSection("rt.build");
    _ops.moeGroupBuildAsync(expIdxSlot, kwSlot, expOffset, rowSrcTok, rowKw,
                            asnToRow, R, nExperts, K);

    // --- 5.22 OEA measure-first: unique-expert union at batched decode ------
    // Env-gated diagnostic (MIMIRMIND_MOE_UNION_PROFILE). Only meaningful at
    // nSeq>1 — a single-token decode has no batch to piggyback on. Reads
    // expOffset back and counts non-empty expert groups = how many distinct
    // expert weight matrices moe.gemm must load this step; that union is the
    // ceiling OEA (batch-aware piggyback routing) can cut. Sync D2H (~1 KB),
    // profile-run only; when the env is unset this whole block is skipped so
    // the prod device-driven path is untouched (no D2H).
    static const bool kOeaUnionProfile =
        std::getenv("MIMIRMIND_MOE_UNION_PROFILE") != nullptr;
    if (kOeaUnionProfile && nSeq > 1) {
        std::vector<std::int32_t> hostOff(nExperts + 1);
        _ops.readbackToHost(hostOff.data(), expOffset,
                            (nExperts + 1) * sizeof(std::int32_t));
        std::size_t unique = 0;
        for (std::size_t e = 0; e < nExperts; ++e) {
            if (hostOff[e + 1] > hostOff[e]) {
                ++unique;
            }
        }
        static long sumUnique = 0;
        static long sumT      = 0;
        static long nCalls    = 0;
        sumUnique += static_cast<long>(unique);
        sumT      += static_cast<long>(nSeq);
        ++nCalls;
        if ((nCalls % 256) == 0) {
            MM_LOG_INFO(
                "moe-oea",
                "union-profile: avg unique={}/{} ({:.1f}%) @avg T={} over {} "
                "MoE-layer-steps (this: layer={} T={} unique={} R={})",
                sumUnique / nCalls, nExperts,
                100.0 * static_cast<double>(sumUnique) /
                    (static_cast<double>(nCalls) * static_cast<double>(nExperts)),
                sumT / nCalls, nCalls, blockIdx, nSeq, unique, R);
        }
    }

    // --- gather activations into per-expert-contiguous rows (both paths) ---
    _ops.profileSection("rt.gather");
    _ops.moeGatherRowsAsync(moeInput, rowSrcTok, xComp, d_model, R);

    // FP4-tensor-core grouped path: the routed experts are the NVFP4_TC format
    // (loader built the nibble + swizzled-SFB + globals banks) and CUTLASS is
    // linked. Type-driven, so it is the default whenever those banks exist.
    // Prefill (preferBlocked=false) always takes TC when the banks exist.
    // GD-c: decode (preferBlocked=true) takes TC only when _moeGroupedDecodeTc
    // is set — real-row act-quant makes it a net decode win (+28% @nSeq64,
    // 2026-08-04 A/B), default-on. When the TC sidecar banks are absent
    // (blocked-only =0 load) tcNibblePtr is null and decode falls through to
    // the device-driven blocked branch below — never the host-driven loop.
    const bool tcGrouped =
        (!preferBlocked || _moeGroupedDecodeTc) &&
        _ops.moeGroupedGemmNvfp4TcAvailable() &&
        gateExps.tcNibblePtr != nullptr &&
        upExps.tcNibblePtr   != nullptr &&
        downExps.tcNibblePtr != nullptr;

    const bool deviceDrivenGrouped =
        !tcGrouped && (_moeGroupedDeviceDriven || preferBlocked) &&
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

        _ops.profileSection("moe.prep");   // row maps + act-quant (prefill sub-split)
        // padded row maps (device only)
        _ops.moePadOffsetsAsync(expOffset, padOffset, nExperts);
        _ops.moeContigToPadAsync(expOffset, padOffset, contigToPad, nExperts, R);
        _ops.moeIndexGatherI32Async(asnToRow, contigToPad, padAsn, nAsn);

        // spread gathered rows -> padded slots; act-quant (SF pre-zeroed).
        // Only the R real rows are quantised (each at its padded slot via
        // contigToPad); the padding rows keep the zeroed SF (scale 0 -> act 0)
        // and their GEMM output is discarded, so quantising them is pure waste
        // (~64x the real rows at decode M). Bit-identical for the real rows.
        _ops.moeRowsScatterF32Async(xComp, contigToPad, xPad, R, d_model);
        _ops.moeZeroBytesAsync(sfaBank, mo::swizzledBlockScaleBytes(maxPad, d_model / 16));
        _ops.moeActQuantNvfp4RowsAsync(xPad, aBank, sfaBank, 1.0F, contigToPad, R, d_model);

        // gate + up: N=n_ff_exp, K=d_model. alpha[e] = weight global (folds the
        // per-expert global back in; act gscale=1).
        _ops.profileSection("moe.gemm");   // gate+up TC GEMM (prefill sub-split)
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

        _ops.profileSection("moe.silu");   // silu + intermediate act-quant (sub-split)
        _ops.siluMulAsync(gatePad, upPad, maxPad * n_ff_exp);  // silu(gate)*up

        // act-quant the intermediate -> down GEMM: N=d_model, K=n_ff_exp.
        // Row-mapped: the GEMM wrote gate/up outputs at the same padded slots
        // (contigToPad), so the intermediate's real rows live there too.
        _ops.moeZeroBytesAsync(sfaBank2, mo::swizzledBlockScaleBytes(maxPad, n_ff_exp / 16));
        _ops.moeActQuantNvfp4RowsAsync(gatePad, aBank2, sfaBank2, 1.0F, contigToPad, R, n_ff_exp);

        _ops.profileSection("moe.dgemm");   // down TC GEMM (prefill sub-split)
        _ops.moeGroupedGemmNvfp4TcBanksAsync(
            nExperts, d_model, n_ff_exp, expOffset, padOffset, aBank2, sfaBank2,
            downExps.tcNibblePtr, downExps.tcSfbPtr,
            static_cast<const float*>(downExps.tcGlobalsPtr), downPad,
            banksScratch, banksBytes);

        // scatter padded expert output back to token order (routed sum).
        _ops.profileSection("moe.sc");   // scatter expert out (prefill sub-split)
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
        // GD-b: decode (preferBlocked) uses tileM=4 + the small-M kernel so the
        // GEMM's shared/register footprint drops and SM occupancy rises (decode
        // M is ~1-2 rows/expert). Prefill keeps tileM=16.
        const std::size_t tileM    = preferBlocked ? 4 : 16;
        const bool        smallM   = preferBlocked;
        const std::size_t maxTiles = (R + tileM - 1) / tileM + nExperts;
        auto* const tileExpert = s.moeGroupTileExpert.as<std::int32_t>();
        auto* const tileRow0   = s.moeGroupTileRow0.as<std::int32_t>();
        auto* const tileRows   = s.moeGroupTileRows.as<std::int32_t>();
        auto* const tileCount  = s.moeGroupTileCount.as<std::int32_t>();

        _ops.profileSection("rt.tiles");
        _ops.moeGroupTilesAsync(expOffset, tileExpert, tileRow0, tileRows,
                                tileCount, nExperts, maxTiles, tileM);

        const auto* const gateBase =
            static_cast<const unsigned char*>(gateExps.usmPtr);
        const auto* const upBase =
            static_cast<const unsigned char*>(upExps.usmPtr);
        const auto* const downBase =
            static_cast<const unsigned char*>(downExps.usmPtr);

        _ops.profileSection("moe.gemm");
        // gate/up: weight [nExperts][n_ff_exp][d_model] (N=n_ff_exp, K=d_model)
        // Single-user decode (nSeq==1 => <=1 row/tile) can take the de-inter-
        // leaved uint4-coalesced kernel (~2x DRAM bandwidth); everything else
        // stays on the interleaved GD-b path.
        const bool deint = _useDeintMoe && nSeq == 1;
        // M1-REG: single-user (nSeq==1) register-staged decode kernel — the
        // activation lives in registers instead of shared memory, removing the
        // ncu-measured MIO/short-scoreboard stall without spending occupancy
        // (+2-4% vs m4). Interleaved layout like GD-b (no de-interleave cache);
        // default-on, mutually exclusive with deint.
        const bool m1nb = _useM1nb && nSeq == 1 && !deint;
        // 5.18.8: fuse gate+up into ONE stacked-w13 deint GEMM (N=2*n_ff). One
        // launch instead of two AND double N -> better SM fill on the tileM=4
        // M=1 kernel. Per-block stacked bank [nExp][2*n_ff][d_model] built once
        // (gate[e] rows then up[e] rows), mirroring the GDN in_proj concat-cache.
        // Bit-identical: same weights, same silu*up math (fused split kernel).
        const bool w13Fused = deint && _moeW13Fuse;
        if (w13Fused) {
            const auto [ge, gb] = moeBlockGeom(gateExps.type);   // {32,20} NVFP4_BLK
            const std::size_t perExpertBytes = n_ff_exp * ((d_model / ge) * gb);
            compute::ComputeBuffer& w13Buf = _moeW13W[blockIdx];
            if (w13Buf.bytes() == 0) {
                w13Buf = _ops.allocate(nExperts * 2 * perExpertBytes);
                auto* const w13 = static_cast<unsigned char*>(
                    static_cast<void*>(w13Buf.as<float>()));
                for (std::size_t e = 0; e < nExperts; ++e) {
                    _ops.appendMemoryCopy(w13 + e * 2 * perExpertBytes,
                        gateBase + e * perExpertBytes, perExpertBytes);
                    _ops.appendMemoryCopy(w13 + e * 2 * perExpertBytes + perExpertBytes,
                        upBase + e * perExpertBytes, perExpertBytes);
                }
            }
            float* const w13Comp = s.moeW13Compact.as<float>();
            const auto* const w13Base = static_cast<const unsigned char*>(
                static_cast<const void*>(w13Buf.as<float>()));
            _ops.moeGroupedGemmNvfp4DeintAsync(xComp, w13Base, w13Comp,
                tileExpert, tileRow0, tileRows, d_model, 2 * n_ff_exp, nExperts,
                maxTiles, smallM);
            _ops.siluMulSplitAsync(w13Comp, gateComp, R, n_ff_exp);
        } else if (deint) {
            _ops.moeGroupedGemmNvfp4DeintAsync(xComp, gateBase, gateComp,
                tileExpert, tileRow0, tileRows, d_model, n_ff_exp, nExperts,
                maxTiles, smallM);
            _ops.moeGroupedGemmNvfp4DeintAsync(xComp, upBase, upComp,
                tileExpert, tileRow0, tileRows, d_model, n_ff_exp, nExperts,
                maxTiles, smallM);
        } else if (m1nb) {
            _ops.moeGroupedGemmNvfp4M1NBAsync(xComp, gateBase, gateComp,
                tileExpert, tileRow0, tileRows, d_model, n_ff_exp, maxTiles);
            _ops.moeGroupedGemmNvfp4M1NBAsync(xComp, upBase, upComp,
                tileExpert, tileRow0, tileRows, d_model, n_ff_exp, maxTiles);
        } else {
            _ops.moeGroupedGemmNvfp4Async(xComp, gateBase, gateComp,
                tileExpert, tileRow0, tileRows, d_model, n_ff_exp, maxTiles, smallM);
            _ops.moeGroupedGemmNvfp4Async(xComp, upBase, upComp,
                tileExpert, tileRow0, tileRows, d_model, n_ff_exp, maxTiles, smallM);
        }
        if (!w13Fused) {
            _ops.siluMulAsync(gateComp, upComp, R * n_ff_exp);  // silu(gate)*up
        }
        // down: weight [nExperts][d_model][n_ff_exp] (N=d_model, K=n_ff_exp)
        if (deint) {
            _ops.moeGroupedGemmNvfp4DeintAsync(gateComp, downBase, downComp,
                tileExpert, tileRow0, tileRows, n_ff_exp, d_model, nExperts,
                maxTiles, smallM);
        } else if (m1nb) {
            _ops.moeGroupedGemmNvfp4M1NBAsync(gateComp, downBase, downComp,
                tileExpert, tileRow0, tileRows, n_ff_exp, d_model, maxTiles);
        } else {
            _ops.moeGroupedGemmNvfp4Async(gateComp, downBase, downComp,
                tileExpert, tileRow0, tileRows, n_ff_exp, d_model, maxTiles, smallM);
        }
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
                    "Qwen3_5MoeBackend::runMoeFfnGrouped: expert weight type(s) "
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
                "Qwen3_5MoeBackend::runMoeFfnGrouped: ffn_gate_shexp has "
                "unexpected shape");
        }
        float* const expertOutBuf = s.expertOutBuf.as<float>();
        if (_shexpTc && nSeq >= kShexpTcMinM &&
            _ops.moeGroupedGemmNvfp4TcAvailable() &&
            gateShexp.tcNibblePtr != nullptr &&
            upShexp->tcNibblePtr  != nullptr &&
            downShexp.tcNibblePtr != nullptr) {
            // Track B: prefill shared expert via the FP4-TC grouped GEMM
            // (nExp=1) -> no kGemmMaxM=16 weight re-read. Sequential across the
            // three projections (they share the shexpTc* scratch).
            sharedExpertTcGemm(n_ff_shexp, d_model, moeInput, nSeq,
                               gateShexp.tcNibblePtr, gateShexp.tcSfbPtr,
                               static_cast<const float*>(gateShexp.tcGlobalsPtr),
                               gateOutBuf, s);
            sharedExpertTcGemm(n_ff_shexp, d_model, moeInput, nSeq,
                               upShexp->tcNibblePtr, upShexp->tcSfbPtr,
                               static_cast<const float*>(upShexp->tcGlobalsPtr),
                               upOutBuf, s);
            _ops.siluMulAsync(gateOutBuf, upOutBuf, nSeq * n_ff_shexp);
            sharedExpertTcGemm(d_model, n_ff_shexp, gateOutBuf, nSeq,
                               downShexp.tcNibblePtr, downShexp.tcSfbPtr,
                               static_cast<const float*>(downShexp.tcGlobalsPtr),
                               expertOutBuf, s);
        } else {
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
        }

        float* const gateScalar = s.scoreScratch.as<float>();
        _gmm.matmulAsync(routerSh.type, routerSh.usmPtr, 1, d_model,
                         moeInput, nSeq, gateScalar, matmulScratch);
        _ops.sigmoidGateMulAsync(expertOutBuf, gateScalar, nSeq, d_model,
                                 /*gateDim=*/1);

        _ops.addResidualAsync(moeAccumBuf, expertOutBuf, nSeq * d_model);
    }
}

void Qwen3_5MoeBackend::runBlockBatched(std::size_t             blockIdx,
                                       float*                  x,
                                       const BatchedDecodeCtx& ctx,
                                       BlockBuffers&           s) {
    if (_config.isRecurrentLayer(blockIdx)) {
        runLinearBlockBatched(blockIdx, x, ctx, s);
    } else {
        runFullAttentionBlockBatched(blockIdx, x, ctx, s);
    }
}

void Qwen3_5MoeBackend::runFullAttentionBlockBatched(
        std::size_t blockIdx, float* x, const BatchedDecodeCtx& ctx,
        BlockBuffers& s, std::size_t kvPoolLayer) {
    const std::size_t nSeq = ctx.nSeq;
    if (nSeq == 0) {
        return;
    }
    // 5.21-III ragged: nRow activation rows (= nSeq decode / sum(seqT) prefill);
    // ropePos = per-token rope position (= startPosDev for T=1 decode). Decode
    // (seqTDev==nullptr) => nRow==nSeq, ropePos==startPosDev => bit-identical.
    const bool                 ragged  = (ctx.seqTDev != nullptr);
    const std::size_t          nRow    = ragged ? ctx.nRow : nSeq;
    const std::int32_t* const  ropePos =
        (ctx.ropePosDev != nullptr) ? ctx.ropePosDev : ctx.startPosDev;
    _ops.profileSection("attn");
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
    const auto& qW       = pickDense(w, blockIdx, "attn_q.weight", nRow, _denseFp8MaxT);
    const auto& kW       = pickDense(w, blockIdx, "attn_k.weight", nRow, _denseFp8MaxT);
    const auto& vW       = pickDense(w, blockIdx, "attn_v.weight", nRow, _denseFp8MaxT);
    const auto& qNorm    = requireBlock(w, blockIdx, "attn_q_norm.weight");
    const auto& kNorm    = requireBlock(w, blockIdx, "attn_k_norm.weight");
    const auto& oW       = pickDense(w, blockIdx, "attn_output.weight", nRow, _denseFp8MaxT);
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

    // --- pre-attention RMSNorm (nRow rows) ---------------------------
    _ops.rmsNormAsync(x, nRow, d_model,
                      static_cast<const float*>(attnNorm.usmPtr), eps, normBuf);

    // --- Q(+gate) / K / V projections (M = nRow) ---------------------
    {
        compute::UnorderedScope u{_ops};
        _gmm.matmulAsync(qW.type, qW.usmPtr, 2 * q_dim, d_model,
                         normBuf, nRow, qGateFused, mmScratch);
        _gmm.matmulAsync(kW.type, kW.usmPtr, kv_dim, d_model,
                         normBuf, nRow, kProj, mmScratch);
        _gmm.matmulAsync(vW.type, vW.usmPtr, kv_dim, d_model,
                         normBuf, nRow, vProj, mmScratch);
    }

    _ops.splitHeadPairAsync(qGateFused, qBuf, gateBuf, nRow, nHeads, head_dim);

    // --- QK-norm (per-head RMS over head_dim) ------------------------
    // In place on the compact q/k buffers — the paged path writes K into
    // the pool below (not a contiguous cache), so unlike the single-seq
    // rmsNormQkvAsync there is no fused cache write here. Per-row rmsnorm
    // is in-place safe (each row is independent).
    _ops.rmsNormAsync(qBuf,  nRow * nHeads,   head_dim,
                      static_cast<const float*>(qNorm.usmPtr), eps, qBuf);
    _ops.rmsNormAsync(kProj, nRow * nKvHeads, head_dim,
                      static_cast<const float*>(kNorm.usmPtr), eps, kProj);
    // The single-session rmsNormQkv also RMS-normalises V per head (over
    // head_dim, with NO learned weight — kernels/cuda/llm/rmsnorm_qkv.cu V
    // branch). The batched path builds q/k/v separately, so replicate the
    // V normalisation here; without it V enters attention un-normalised and
    // the whole full-attention output is off by V's per-head 1/rms factor.
    _ops.rmsNormNoWeightAsync(vProj, nRow * nKvHeads, head_dim, eps, vProj);

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
        _ops.mropeInPlaceBatchedAsync(qBuf, nRow, q_dim, /*seqLen=*/1,
                                      nHeads, head_dim, ropePos,
                                      _config.ropeFreqBase, _ropeSections,
                                      /*writeOffsetStride=*/0,
                                      runtime::KvDtype::F32);
        _ops.mropeInPlaceBatchedAsync(kProj, nRow, kv_dim, /*seqLen=*/1,
                                      nKvHeads, head_dim, ropePos,
                                      _config.ropeFreqBase, _ropeSections,
                                      /*writeOffsetStride=*/0,
                                      runtime::KvDtype::F32);
    }

    // --- scatter this step's K/V into the paged pool -----------------
    // Device path (graph-capturable) when the device index arrays are provided;
    // otherwise the per-seq host loop (host-computed slot address).
    if (ctx.writeBlockIdDev != nullptr && ctx.writeSlotDev != nullptr) {
        // Per-ROW scatter (nRow tokens); for ragged prefill writeBlockId/Slot are
        // per-token [nRow]. Decode nRow==nSeq (unchanged).
        ctx.pool->writeTokensBatched(_ops, denseLayer, kProj, vProj,
                                     ctx.writeBlockIdDev, ctx.writeSlotDev, nRow,
                                     ctx.activeMask);
    } else {
        for (std::size_t seq = 0; seq < nSeq; ++seq) {
            ctx.pool->writeToken(_ops, denseLayer,
                                 ctx.writeBlockId[seq],
                                 static_cast<std::size_t>(ctx.writeSlot[seq]),
                                 kProj + seq * kv_dim,
                                 vProj + seq * kv_dim);
        }
    }

    // --- paged GQA attention (block-table indirection) ---------------
    const float attnScale = _config.attentionScaleFor(head_dim);
    // Split-K paged decode (V2) when the context is long enough to split;
    // V2 internally falls back to the single-pass V1 for short/unknown
    // (maxSeqLen<=partition) contexts so short-prompt decode keeps V1's speed.
    // MIMIRMIND_PAGED_V1=1 forces the old single-pass path (A/B + rollback).
    _ops.profileSection("attn.paged");   // decode attn sub-split (paged kernel)
    // Pool base pointers are void* (element width follows dtype()); pass them as
    // addresses — the fp16 V2 kernel variant reinterprets K/V as __half (5.14 I1).
    const auto  kvDtype = ctx.pool->dtype();
    const auto* keyBase = static_cast<const float*>(ctx.pool->keyPool(denseLayer));
    const auto* valBase =
        static_cast<const float*>(ctx.pool->valuePool(denseLayer));
    if (ragged) {
        // 5.21-III HYBRID: a TRUE mixed forward routes seqT==1 (decode) rows
        // through split-K paged decode-V2 (fast for long context) and leaves the
        // seqT>1 (prefill) rows on the O(seq_len) prefill-causal path. Decode rows
        // are gathered to a compact [D] set (query rows = hybDecodeRowMapDev),
        // run through decode-V2, and scattered back into attnOut; prefill-causal
        // then writes ONLY the prefill rows (hybSeqTPrefillDev zeroes the decode
        // slots so its `pq >= seqT` guard skips them). hybDecodeCount==0 => the
        // plain all-rows prefill-causal path (pure prefill / hybrid disabled).
        if (ctx.hybDecodeCount > 0) {
            const std::size_t D = ctx.hybDecodeCount;
            _ops.moeGatherRowsAsync(qBuf, ctx.hybDecodeRowMapDev,
                                    ctx.hybQDecodeScratch, q_dim, D);
            _ops.pagedAttentionDecodeV2Async(
                ctx.hybAttnDecodeScratch, ctx.hybQDecodeScratch, keyBase, valBase,
                ctx.hybDecodeBlockTablesDev, ctx.hybDecodeSeqLensDev,
                D, nHeads, nKvHeads, head_dim, ctx.pool->blockSize(),
                ctx.maxBlocksPerSeq, ctx.hybMaxDecodeSeqLen, attnScale,
                /*softcap=*/0.0f, kvDtype);
            _ops.moeRowsScatterF32Async(ctx.hybAttnDecodeScratch,
                                        ctx.hybDecodeRowMapDev, attnOut, D, q_dim);
        }
        // prefill/varlen rows attend CAUSALLY over their chunk + prior KV (query
        // pq -> keys [0, startPos[seq]+pq]) via the paged pool. When hybrid is on,
        // seqTPrefill zeroes the decode slots so only prefill rows are written.
        _ops.pagedAttentionPrefillCausalAsync(
            attnOut, qBuf, keyBase, valBase, ctx.blockTablesDev,
            (ctx.hybDecodeCount > 0) ? ctx.hybSeqTPrefillDev : ctx.seqTDev,
            ctx.seqOffDev, ctx.startPosDev,
            nSeq, nHeads, nKvHeads, head_dim, ctx.pool->blockSize(),
            ctx.maxBlocksPerSeq, ctx.maxSeqT, attnScale, /*softcap=*/0.0f,
            kvDtype);   // 5.16: fp16/fp8 pool → matching prefill-causal read
    } else if (_forcePagedV1 && kvDtype == runtime::KvDtype::F32) {
        // V1 is F32-only; a non-F32 (fp16/fp8) pool always routes through V2
        // (which reinterprets the pool bytes per dtype). The dtype guard makes
        // the "fp16 pool always routes through V2" invariant real — a bare
        // MIMIRMIND_PAGED_V1=1 with a non-F32 pool would otherwise misread the
        // packed bytes as F32 (5.16; pre-existing latent for fp16).
        _ops.pagedAttentionDecodeV1Async(
            attnOut, qBuf, keyBase, valBase, ctx.blockTablesDev, ctx.seqLensDev,
            nSeq, nHeads, nKvHeads, head_dim, ctx.pool->blockSize(),
            ctx.maxBlocksPerSeq, attnScale, /*softcap=*/0.0f);
    } else {
        _ops.pagedAttentionDecodeV2Async(
            attnOut, qBuf, keyBase, valBase, ctx.blockTablesDev, ctx.seqLensDev,
            nSeq, nHeads, nKvHeads, head_dim, ctx.pool->blockSize(),
            ctx.maxBlocksPerSeq, static_cast<std::size_t>(ctx.maxSeqLen),
            attnScale, /*softcap=*/0.0f, kvDtype);
    }

    // --- output gate + O projection + attn residual ------------------
    _ops.profileSection("attn.out");   // decode attn sub-split (gate + O proj)
    _ops.sigmoidGateMulAsync(attnOut, gateBuf, nRow, q_dim, /*gateDim=*/q_dim);
    _gmm.matmulAsync(oW.type, oW.usmPtr, d_model, q_dim,
                     attnOut, nRow, projOut, mmScratch);
    _ops.addResidualAsync(x, projOut, nRow * d_model);

    // --- post-attention norm -> batched MoE -> FFN residual ----------
    _ops.rmsNormAsync(x, nRow, d_model,
                      static_cast<const float*>(attnPost.usmPtr), eps, normBuf);
    if (_moeGroupedDecode) {
        // GD-a: expert-grouped decode — amortise routed expert-weight reads
        // across the batch. preferBlocked marks decode; ragged prefill (nRow>nSeq)
        // uses the batched path (M>1) via preferBlocked=false.
        runMoeFfnGrouped(blockIdx, normBuf, nRow, ctx.expIdxSlot, ctx.kwSlot, s,
                         /*preferBlocked=*/!ragged);
    } else {
        runMoeFfnBatched(blockIdx, normBuf, nRow, ctx.expIdxSlot, ctx.kwSlot, s);
    }
    _ops.addResidualAsync(x, s.moeAccumBuf.as<float>(), nRow * d_model);
}

void Qwen3_5MoeBackend::runLinearBlockBatched(
        std::size_t blockIdx, float* x, const BatchedDecodeCtx& ctx,
        BlockBuffers& s) {
    const std::size_t nSeq = ctx.nSeq;
    // GDN sub-instrumentation (MIMIRMIND_DECODE_PROFILE only): split the single
    // "gdn" bucket into proj / conv / recur / tail to locate the residual cost.
    _ops.profileSection("gdn.proj");
    if (nSeq == 0) {
        return;
    }
    // 5.21-III ragged: nRow activation rows (= nSeq decode / sum(seqT) prefill).
    // Decode (seqTDev==nullptr) => nRow==nSeq => bit-identical. GDN uses no RoPE.
    const bool        ragged = (ctx.seqTDev != nullptr);
    const std::size_t nRow   = ragged ? ctx.nRow : nSeq;

    const auto& w        = _weights;
    const auto& attnNorm = requireBlock(w, blockIdx, "attn_norm.weight");
    const auto& qkvW     = pickDense(w, blockIdx, "attn_qkv.weight", nRow, _denseFp8MaxT);
    const auto& gateW    = pickDense(w, blockIdx, "attn_gate.weight", nRow, _denseFp8MaxT);
    const auto& betaW    = requireBlock(w, blockIdx, "ssm_beta.weight");
    const auto& alphaW   = requireBlock(w, blockIdx, "ssm_alpha.weight");
    const auto& ssmA     = requireBlock(w, blockIdx, "ssm_a");
    const auto& ssmDt    = requireBlock(w, blockIdx, "ssm_dt.bias");
    const auto& convW    = requireBlock(w, blockIdx, "ssm_conv1d.weight");
    const auto& ssmNormW = requireBlock(w, blockIdx, "ssm_norm.weight");
    const auto& ssmOutW  = pickDense(w, blockIdx, "ssm_out.weight", nRow, _denseFp8MaxT);
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
    float*       qkvMixed  = s.ssmQkvMixed.as<float>();   // [nSeq, convDim] (also conv out; fused proj -> _gdnQkvzOut)
    float* const convInput = s.ssmConvInput.as<float>();  // [nSeq, dConv, convDim] (serving-sized)
    float*       zBuf      = s.ssmZ.as<float>();
    float* const qBuf      = s.ssmQ.as<float>();
    float* const kBuf      = s.ssmK.as<float>();
    float* const vBuf      = s.ssmV.as<float>();
    float* const deltaOut  = s.ssmDeltaOut.as<float>();
    float*       alphaBuf  = s.ssmAlpha.as<float>();
    float*       betaBuf   = s.ssmBeta.as<float>();
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

    // --- pre-attention RMSNorm (nRow rows) ---------------------------
    _ops.rmsNormAsync(x, nRow, d_model,
                      static_cast<const float*>(attnNorm.usmPtr), eps, normBuf);

    // --- projections (M = nRow; fused proj is nSeq==1-only => decode) -
    // GDN-Inc 1: at nSeq==1 decode, fuse qkv+gate -> qkvz and beta+alpha -> ba
    // into ONE fp8 GEMV each (vLLM in_proj_qkvz / in_proj_ba). The fused BF16
    // weight is concatenated once per block and cached; matmulAsync quantises it
    // to E4M3 with a SINGLE per-tensor scale, matching vLLM's fused-tensor
    // granularity. The output is contiguous [qkv | gate] / [beta | alpha], so
    // the split is pure pointer arithmetic at nSeq==1. Falls back to 4 matmuls.
    if (_gdnProjFuse && nSeq == 1 && !ragged) {   // 5.21-III: fusion is nRow==1-only
        auto buildFused = [&](compute::ComputeBuffer& dst,
                              const core::gguf::GgufTensor& wa,
                              const core::gguf::GgufTensor& wb) {
            if (dst.bytes() == 0) {
                dst = _ops.allocate(wa.nbytes + wb.nbytes);
                char* const base =
                    static_cast<char*>(static_cast<void*>(dst.as<float>()));
                _ops.appendMemoryCopy(base,             wa.usmPtr, wa.nbytes);
                _ops.appendMemoryCopy(base + wa.nbytes, wb.usmPtr, wb.nbytes);
            }
        };
        auto grow = [&](compute::ComputeBuffer& b, std::size_t bytes) {
            if (b.bytes() < bytes) { b = _ops.allocate(bytes); }
        };
        buildFused(_gdnQkvzW[blockIdx], qkvW,  gateW);
        buildFused(_gdnBaW[blockIdx],   betaW, alphaW);
        grow(_gdnQkvzOut, (convDim + valueDim) * nSeq * sizeof(float));
        grow(_gdnBaOut,   (std::size_t{2} * hV) * nSeq * sizeof(float));
        float* const qkvzOut = _gdnQkvzOut.as<float>();
        float* const baOut   = _gdnBaOut.as<float>();
        {
            compute::UnorderedScope u{_ops};
            _gmm.matmulAsync(qkvW.type, _gdnQkvzW[blockIdx].as<float>(),
                             convDim + valueDim, d_model, normBuf, nSeq,
                             qkvzOut, mmScratch);
            _gmm.matmulAsync(betaW.type, _gdnBaW[blockIdx].as<float>(),
                             2 * hV, d_model, normBuf, nSeq, baOut, mmScratch);
        }
        qkvMixed = qkvzOut;               // [0, convDim)
        zBuf     = qkvzOut + convDim;     // [convDim, convDim+valueDim)
        betaBuf  = baOut;                 // [0, hV)
        alphaBuf = baOut + hV;            // [hV, 2*hV)
    } else {
        compute::UnorderedScope u{_ops};
        _gmm.matmulAsync(qkvW.type,  qkvW.usmPtr,  convDim,  d_model, normBuf, nRow, qkvMixed, mmScratch);
        _gmm.matmulAsync(gateW.type, gateW.usmPtr, valueDim, d_model, normBuf, nRow, zBuf,     mmScratch);
        _gmm.matmulAsync(betaW.type, betaW.usmPtr, hV,       d_model, normBuf, nRow, betaBuf,  mmScratch);
        _gmm.matmulAsync(alphaW.type,alphaW.usmPtr,hV,       d_model, normBuf, nRow, alphaBuf, mmScratch);
    }

    // beta = sigmoid(beta); gLog = ssm_a * softplus(alpha + ssm_dt).
    // GDN-Inc 2: when the gate is folded into the recurrence kernel these two
    // launches are skipped and beta/alpha stay RAW (consumed by the fused
    // recurrence below). MIMIRMIND_GDN_GATE_FUSE=1.
    if (!_gdnGateFuse) {
        _ops.sigmoidInPlaceAsync(betaBuf, nRow * hV);
        _ops.deltanetGateAsync(alphaBuf,
                               static_cast<const float*>(ssmA.usmPtr),
                               static_cast<const float*>(ssmDt.usmPtr),
                               gateBuf, nRow, hV);
    }

    // --- causal conv1d + silu (per-seq rolling state) ----------------
    // convInput[seq] = [convState[seq] (stateRows) | qkvMixed[seq] (1 row)].
    // Serving BlockBuffers must size ssmConvInput to nSeq*dConv*convDim.
    _ops.profileSection("gdn.conv");
    const std::size_t convInBytes  = convStateElems * sizeof(float);
    const std::size_t qkvRowBytes  = convDim * sizeof(float);
    // 5.21-III ragged: slot seq's conv input = [state tail (stateRows) | its
    // seqT[seq] tokens] at convInOff[seq]; its tokens come from qkvMixed at token
    // offset seqOff[seq]. Decode (ragged=false): Tslot=1, inRowOff=seq*dConv,
    // tokOff=seq — exactly the pre-varlen layout.
    std::size_t inRowRun = 0, tokRun = 0;
    for (std::size_t seq = 0; seq < nSeq; ++seq) {
        const std::size_t Tslot =
            ragged ? static_cast<std::size_t>(ctx.seqTHost[seq]) : 1;
        const std::size_t inRowOff = ragged ? inRowRun : seq * dConv;
        const std::size_t tokOff   = ragged ? tokRun   : seq;
        float* const cvState = convBase + seq * convStateElems;
        float* const cvIn    = convInput + inRowOff * convDim;
        const bool frozen = ctx.activeMaskHost != nullptr
                            && ctx.activeMaskHost[seq] == 0;
        if (!frozen && ctx.isSeqStart != nullptr && ctx.isSeqStart[seq] != 0) {
            _ops.mulScalarAsync(cvState, 0.0F, convStateElems);
        }
        _ops.appendMemoryCopy(cvIn, cvState, convInBytes);
        _ops.appendMemoryCopy(cvIn + stateRows * convDim,
                              qkvMixed + tokOff * convDim, Tslot * qkvRowBytes);
        inRowRun += stateRows + Tslot;
        tokRun   += Tslot;
    }
    _ops.causalConv1dSiluBatchedAsync(
        convInput, static_cast<const float*>(convW.usmPtr), qkvMixed,
        nSeq, ragged ? ctx.maxSeqT : 1, convDim, dConv,
        ragged ? ctx.seqTDev : nullptr,
        ragged ? ctx.convInOffDev : nullptr,
        ragged ? ctx.seqOffDev : nullptr);
    // Save each sequence's trailing stateRows rows as the next conv state (the
    // last stateRows of [state | Tslot tokens] start at row Tslot).
    // 5.21-I: a frozen slot keeps its conv tail byte-identical (skip the save).
    std::size_t inRowRun2 = 0;
    for (std::size_t seq = 0; seq < nSeq; ++seq) {
        const std::size_t Tslot =
            ragged ? static_cast<std::size_t>(ctx.seqTHost[seq]) : 1;
        const std::size_t inRowOff = ragged ? inRowRun2 : seq * dConv;
        inRowRun2 += stateRows + Tslot;
        if (ctx.activeMaskHost != nullptr && ctx.activeMaskHost[seq] == 0) {
            continue;
        }
        float* const cvState = convBase + seq * convStateElems;
        float* const cvIn    = convInput + inRowOff * convDim;
        _ops.appendMemoryCopy(cvState, cvIn + Tslot * convDim, convInBytes);
    }

    // --- split conv into q/k/v (+ GQA repeat H_k -> H_v) + q/k L2-norm ---
    // GDN-Inc 2b: one fused launch (gather q/k/v + norm q/k) vs 3 gathers + 2 norms.
    if (_gdnPrepFuse) {
        _ops.fusedPostConvPrepAsync(qkvMixed, qBuf, kBuf, vBuf, nRow, hK, hV, S,
                                    convDim, keyDim, eps);
    } else {
        _ops.gatherHeadsFromChannelsAsync(qkvMixed, qBuf, nRow, 0,          hK, hV, S, convDim);
        _ops.gatherHeadsFromChannelsAsync(qkvMixed, kBuf, nRow, keyDim,     hK, hV, S, convDim);
        _ops.gatherHeadsFromChannelsAsync(qkvMixed, vBuf, nRow, 2 * keyDim, hV, hV, S, convDim);
        _ops.l2NormInPlaceAsync(qBuf, nRow * hV, S, eps);
        _ops.l2NormInPlaceAsync(kBuf, nRow * hV, S, eps);
    }

    // --- gated delta-rule recurrence (persistent per-seq state) ------
    _ops.profileSection("gdn.recur");
    for (std::size_t seq = 0; seq < nSeq; ++seq) {
        // 5.21-I: a masked (frozen) slot must NOT be zeroed even if seqStart —
        // freeze dominates; its state stays byte-identical.
        const bool frozen = ctx.activeMaskHost != nullptr
                            && ctx.activeMaskHost[seq] == 0;
        if (!frozen && ctx.isSeqStart != nullptr && ctx.isSeqStart[seq] != 0) {
            _ops.mulScalarAsync(stateBase + seq * stateElems, 0.0F, stateElems);
        }
    }
    // 5.21-III ragged: per-slot seqT/seqOff drive the recurrence (T fallback =
    // maxSeqT). Decode (ragged=false) => seqT/seqOff nullptr, T=1 => bit-identical.
    const compute::GdnBatchedShape gdnShape{
        nSeq, ragged ? ctx.maxSeqT : 1, hV, S, ctx.activeMask,
        ragged ? ctx.seqTDev : nullptr, ragged ? ctx.seqOffDev : nullptr};
    if (_gdnGateFuse) {
        // GDN-Inc 2: gate folded in — pass RAW alpha/beta + per-head ssm_a/ssm_dt.
        _ops.gatedDeltaNetRecurrentGateFusedBatchedAsync(
            qBuf, kBuf, vBuf, alphaBuf, betaBuf,
            static_cast<const float*>(ssmA.usmPtr),
            static_cast<const float*>(ssmDt.usmPtr),
            stateBase, deltaOut, gdnShape);
    } else {
        _ops.gatedDeltaNetRecurrentBatchedAsync(qBuf, kBuf, vBuf, gateBuf, betaBuf,
                                                stateBase, deltaOut, gdnShape);
    }

    // --- gated output norm: ssm_norm(out) * silu(z) ------------------
    _ops.profileSection("gdn.tail");
    _ops.rmsNormAsync(deltaOut, nRow * hV, S,
                      static_cast<const float*>(ssmNormW.usmPtr), eps, qBuf);
    _ops.siluMulAsync(zBuf, qBuf, nRow * valueDim);

    // --- output projection ssm_out -----------------------------------
    _gmm.matmulAsync(ssmOutW.type, ssmOutW.usmPtr, d_model, valueDim,
                     zBuf, nRow, projOut, mmScratch);
    _ops.addResidualAsync(x, projOut, nRow * d_model);

    // --- post-attn norm -> batched MoE -> FFN residual ---------------
    _ops.rmsNormAsync(x, nRow, d_model,
                      static_cast<const float*>(attnPost.usmPtr), eps, normBuf);
    if (_moeGroupedDecode) {
        // preferBlocked marks decode; ragged prefill (nRow>nSeq) uses the batched
        // path (M>1) via preferBlocked=false.
        runMoeFfnGrouped(blockIdx, normBuf, nRow, ctx.expIdxSlot, ctx.kwSlot, s,
                         /*preferBlocked=*/!ragged);
    } else {
        runMoeFfnBatched(blockIdx, normBuf, nRow, ctx.expIdxSlot, ctx.kwSlot, s);
    }
    _ops.addResidualAsync(x, s.moeAccumBuf.as<float>(), nRow * d_model);
}

void Qwen3_5MoeBackend::runLinearBlockVerify(
        std::size_t blockIdx, float* x, std::size_t N, std::size_t Kp1,
        std::int32_t* expIdxSlot, float* kwSlot, const std::uint8_t* gdnSeqStart,
        std::size_t maxBatch, float* ssmExport, float* const* convSnap,
        BlockBuffers& s) {
    if (N == 0 || Kp1 == 0) {
        return;
    }
    const std::size_t M = N * Kp1;

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
    const std::size_t valueDim       = _config.ssmInnerSize;
    const std::size_t convDim        = _config.ssmConvDim();
    const std::size_t dConv          = _config.ssmConvKernel;
    const std::size_t keyDim         = S * hK;
    const std::size_t stateElems     = _config.ssmStateElemsPerLayer();
    const std::size_t convStateElems = _config.ssmConvStateElemsPerLayer();
    const std::size_t stateRows      = (dConv > 0 ? dConv - 1 : 0);
    const float       eps            = _config.rmsNormEps;

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
    float* const projOut   = s.projOut.as<float>();
    float* const mmScratch = s.matmulScratch.as<float>();

    const std::size_t slabNSeq = (s.ssmSlabNSeq != 0) ? s.ssmSlabNSeq : N;
    float* const stateBase = s.ssmStatePtr     + blockIdx * (slabNSeq * stateElems);
    float* const convBase  = s.ssmConvStatePtr + blockIdx * (slabNSeq * convStateElems);
    const std::size_t cvStride = slabNSeq * convStateElems;

    // MV-d: this layer's compact per-position state-export slab. Layout
    // [Kp1, maxBatch, stateElems] (time-major on the verify position), written
    // by the fused verify kernel with the runtime slot stride N (positions
    // packed contiguously in the first Kp1*N*stateElems). The per-layer stride
    // uses maxBatch so the buffer size is constant across runs.
    float* const ssmExportBase = ssmExport + blockIdx * (Kp1 * maxBatch * stateElems);

    // === Phase 1: weight-heavy projections, BATCHED over M (read once) =====
    _ops.profileSection("verify.proj");   // MTP-verify breakdown (DECODE_PROFILE)
    _ops.rmsNormAsync(x, M, d_model,
                      static_cast<const float*>(attnNorm.usmPtr), eps, normBuf);
    {
        compute::UnorderedScope u{_ops};
        _gmm.matmulAsync(qkvW.type,  qkvW.usmPtr,  convDim,  d_model, normBuf, M, qkvMixed, mmScratch);
        _gmm.matmulAsync(gateW.type, gateW.usmPtr, valueDim, d_model, normBuf, M, zBuf,     mmScratch);
        _gmm.matmulAsync(betaW.type, betaW.usmPtr, hV,       d_model, normBuf, M, betaBuf,  mmScratch);
        _gmm.matmulAsync(alphaW.type,alphaW.usmPtr,hV,       d_model, normBuf, M, alphaBuf, mmScratch);
    }
    _ops.sigmoidInPlaceAsync(betaBuf, M * hV);
    _ops.deltanetGateAsync(alphaBuf,
                           static_cast<const float*>(ssmA.usmPtr),
                           static_cast<const float*>(ssmDt.usmPtr),
                           gateBuf, M, hV);

    // === Phase 2: conv1d (per position) + ONE fused gated-delta verify =====
    // Position j's N slots are the contiguous row block [j*N, j*N+N). conv1d
    // stays per-position (cheap rolling causal state) and its full slab is
    // snapshotted into convSnap[j] for partial-accept restore. The gated-delta
    // recurrence, by contrast, runs as ONE fused kernel over the whole K+1
    // window: the [S,S] state stays resident across the positions (no per-token
    // global round-trip, no K+1 separate launches) and every position's
    // post-step state is exported to ssmExportBase — replacing the K+1
    // full-slab recurrent snapshots. The kernel reads stateBase (S_0) but does
    // NOT advance it; the caller commits from ssmExport[a].
    _ops.profileSection("verify.conv");
    const std::size_t convInBytes = convStateElems * sizeof(float);
    const std::size_t qkvRowBytes = convDim * sizeof(float);
    for (std::size_t j = 0; j < Kp1; ++j) {
        float* const qkvMixedJ = qkvMixed + j * N * convDim;
        const std::uint8_t* const seqStartJ = gdnSeqStart + j * maxBatch;

        // causal conv1d + silu (per-seq rolling state) over the N slots.
        for (std::size_t seq = 0; seq < N; ++seq) {
            float* const cvState = convBase + seq * convStateElems;
            float* const cvIn    = convInput + seq * dConv * convDim;
            if (seqStartJ[seq] != 0) {
                _ops.mulScalarAsync(cvState, 0.0F, convStateElems);
            }
            _ops.appendMemoryCopy(cvIn, cvState, convInBytes);
            _ops.appendMemoryCopy(cvIn + stateRows * convDim,
                                  qkvMixedJ + seq * convDim, qkvRowBytes);
        }
        _ops.causalConv1dSiluBatchedAsync(convInput,
                                          static_cast<const float*>(convW.usmPtr),
                                          qkvMixedJ, N, /*T=*/1, convDim, dConv);
        for (std::size_t seq = 0; seq < N; ++seq) {
            float* const cvState = convBase + seq * convStateElems;
            float* const cvIn    = convInput + seq * dConv * convDim;
            _ops.appendMemoryCopy(cvState, cvIn + /*T=*/1 * convDim, convInBytes);
        }

        // Gather q/k/v for this position into its time-major slice so the fused
        // kernel sees all Kp1 positions at once ([Kp1, N, hV, S]).
        float* const qJ = qBuf + j * N * valueDim;
        float* const kJ = kBuf + j * N * valueDim;
        float* const vJ = vBuf + j * N * valueDim;
        _ops.gatherHeadsFromChannelsAsync(qkvMixedJ, qJ, N, 0,          hK, hV, S, convDim);
        _ops.gatherHeadsFromChannelsAsync(qkvMixedJ, kJ, N, keyDim,     hK, hV, S, convDim);
        _ops.gatherHeadsFromChannelsAsync(qkvMixedJ, vJ, N, 2 * keyDim, hV, hV, S, convDim);
        _ops.l2NormInPlaceAsync(qJ, N * hV, S, eps);
        _ops.l2NormInPlaceAsync(kJ, N * hV, S, eps);

        // Conv-state snapshot after verify position j (recurrent state is
        // exported by the fused kernel below, not snapshotted here).
        _ops.appendMemoryCopy(convSnap[j] + blockIdx * cvStride,
                              convBase, cvStride * sizeof(float));
    }

    // Verify never re-starts a sequence mid-window (all slots are ongoing
    // committed sequences); a fresh start can only appear at position 0, which
    // maps to the fused kernel's stateIn. Reset S_0 for any such slot.
    for (std::size_t seq = 0; seq < N; ++seq) {
        if (gdnSeqStart[seq] != 0) {
            _ops.mulScalarAsync(stateBase + seq * stateElems, 0.0F, stateElems);
        }
    }

    // ONE fused verify kernel: q/k/v/out time-major [Kp1, N, hV, S], gate/beta
    // [Kp1, N, hV], stateIn [N, stateElems], stateOut [Kp1, N, stateElems]
    // packed with stride N into this layer's export slab.
    _ops.profileSection("verify.gdn");
    _ops.gatedDeltaNetVerifyBatchedAsync(qBuf, kBuf, vBuf, gateBuf, betaBuf,
                                         stateBase, ssmExportBase, deltaOut,
                                         N, Kp1, hV, S);

    // === Phase 3: gated output norm + out-proj + MoE, BATCHED over M =======
    _ops.profileSection("verify.tail");
    _ops.rmsNormAsync(deltaOut, M * hV, S,
                      static_cast<const float*>(ssmNormW.usmPtr), eps, qBuf);
    _ops.siluMulAsync(zBuf, qBuf, M * valueDim);
    _gmm.matmulAsync(ssmOutW.type, ssmOutW.usmPtr, d_model, valueDim,
                     zBuf, M, projOut, mmScratch);
    _ops.addResidualAsync(x, projOut, M * d_model);

    _ops.rmsNormAsync(x, M, d_model,
                      static_cast<const float*>(attnPost.usmPtr), eps, normBuf);
    if (_moeGroupedDecode) {
        runMoeFfnGrouped(blockIdx, normBuf, M, expIdxSlot, kwSlot, s,
                         /*preferBlocked=*/true);
    } else {
        runMoeFfnBatched(blockIdx, normBuf, M, expIdxSlot, kwSlot, s);
    }
    _ops.addResidualAsync(x, s.moeAccumBuf.as<float>(), M * d_model);
}

} // namespace mimirmind::runtime::arch