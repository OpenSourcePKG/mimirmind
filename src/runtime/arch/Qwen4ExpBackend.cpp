// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/arch/Qwen4ExpBackend.hpp"

#include "compute/ComputeMatmul.hpp"
#include "compute/ComputeOps.hpp"
#include "core/gguf/WeightsMap.hpp"
#include "core/log/Log.hpp"
#include "model/LlmConfig.hpp"
#include "runtime/BlockBuffers.hpp"

#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace mimirmind::runtime::arch {

namespace {

// Sentinel block index selecting the top-level hyper_connection_mixer tensors
// (vs the per-layer blk.N.* tensors) inside hcGatedResidual.
constexpr std::size_t kMixerBlock = static_cast<std::size_t>(-1);

const core::gguf::GgufTensor& requireBlockT(const core::gguf::WeightsMap& w,
                                            std::size_t blockIdx,
                                            const std::string& suffix) {
    const std::string name = "blk." + std::to_string(blockIdx) + "." + suffix;
    const auto* t = w.find(name);
    if (t == nullptr) {
        throw std::runtime_error("Qwen4ExpBackend: missing tensor '" + name + "'");
    }
    return *t;
}

const core::gguf::GgufTensor& requireTopT(const core::gguf::WeightsMap& w,
                                          const std::string& name) {
    const auto* t = w.find(name);
    if (t == nullptr) {
        throw std::runtime_error("Qwen4ExpBackend: missing tensor '" + name + "'");
    }
    return *t;
}

// I-6 diagnostic (MIMIRMIND_Q4E_DIAG): readback + log L2/max/nan of a device
// buffer to localise where the qwen4_exp forward diverges. Temporary.
void dumpNorm(compute::ComputeOps& ops, compute::ComputeMatmul& gmm,
              const char* tag, const float* dev, std::size_t n,
              std::size_t nRows = 1) {
    static const bool diag = std::getenv("MIMIRMIND_Q4E_DIAG") != nullptr;
    if (!diag || dev == nullptr || n == 0) return;
    gmm.sync();
    std::vector<float> h(n);
    ops.readbackToHost(h.data(), dev, n * sizeof(float));
    double ss = 0.0; float mx = 0.0F; std::size_t bad = 0;
    for (float v : h) {
        ss += static_cast<double>(v) * v;
        if (std::fabs(v) > mx) mx = std::fabs(v);
        if (std::isnan(v) || std::isinf(v)) ++bad;
    }
    // 5.27.11.2 localization: when the buffer holds nRows independent rows and
    // those rows SHOULD be identical (e.g. batched decode of identical prompts),
    // maxRowDiff pinpoints the first op that makes identical rows diverge.
    if (nRows > 1 && n % nRows == 0) {
        const std::size_t rl = n / nRows;
        float maxRowDiff = 0.0F;
        for (std::size_t r = 1; r < nRows; ++r) {
            for (std::size_t j = 0; j < rl; ++j) {
                const float d = std::fabs(h[r * rl + j] - h[j]);
                if (d > maxRowDiff) maxRowDiff = d;
            }
        }
        MM_LOG_INFO("q4ediag",
                    "{} l2={:.4g} max={:.4g} nan/inf={} rows={} maxRowDiff={:.4g}",
                    tag, std::sqrt(ss), mx, bad, nRows, maxRowDiff);
        return;
    }
    MM_LOG_INFO("q4ediag", "{} l2={:.4g} max={:.4g} nan/inf={}", tag,
                std::sqrt(ss), mx, bad);
}

// --- I-4 PLE n-gram hashing (host; verified in tools/microbench/q4e_ngram_*) --
bool isPrime(std::int64_t v) {
    if (v < 2) return false;
    if (v % 2 == 0) return v == 2;
    for (std::int64_t d = 3; d * d <= v; d += 2)
        if (v % d == 0) return false;
    return true;
}
std::int64_t findNthPrimeAfter(std::int64_t start, std::int64_t count) {
    std::int64_t p = start;
    for (std::int64_t i = 0; i < count; ++i) { ++p; while (!isPrime(p)) ++p; }
    return p;
}
// EOS-aware right shift over a token row (B=1), mirroring the reference.
std::vector<std::int32_t> shiftRightIgnoreEos(const std::vector<std::int32_t>& tok,
                                              int shift, std::int32_t eos) {
    const int L = static_cast<int>(tok.size());
    if (shift == 0) return tok;
    std::vector<std::int32_t> out(L);
    std::int64_t run = -1, prevMaxPrev = -1;
    for (int i = 0; i < L; ++i) {
        const std::int64_t eospos = (tok[i] == eos) ? i : -1;
        const std::int64_t previousEos = (i == 0) ? -1 : prevMaxPrev;
        if (eospos > run) run = eospos;
        const std::int64_t segStart = previousEos + 1;
        const std::int64_t posInSeg = i - segStart;
        const std::int64_t src = static_cast<std::int64_t>(i) - shift;
        const std::int32_t shifted = tok[src < 0 ? 0 : src];
        out[i] = (posInSeg >= shift && src >= 0) ? shifted : eos;
        prevMaxPrev = run;
    }
    return out;
}

} // namespace

// I-1 seam: forward everything to the Qwen3_5MoeBackend backbone. I-3 overrides
// the residual-stream seams (blockInputNorm / blockResidualAdd) with the 4-stream
// Hyper-Connections GatedResidual; the heavy attn/GDN/MoE compute is unchanged.
Qwen4ExpBackend::Qwen4ExpBackend(const model::LlmConfig&       config,
                                 const core::gguf::WeightsMap& weights,
                                 const model::FusedQkvWeights* fusedQkv,
                                 compute::ComputeOps&          ops,
                                 compute::ComputeMatmul&       gmm,
                                 runtime::OpProfiler&          opProfiler,
                                 bool                          moeGroupEnabled,
                                 bool                          moeFusedDownEnabled)
    : Qwen3_5MoeBackend(config, weights, fusedQkv, ops, gmm, opProfiler,
                        moeGroupEnabled, moeFusedDownEnabled) {}

void Qwen4ExpBackend::growHcScratch(std::size_t T) {
    if (T <= _hcCapT) {
        return;
    }
    const std::size_t d       = _config.embeddingLength;
    const std::size_t hc      = _config.hcCount;
    const std::size_t lowrank = _config.hcLowrank;
    const std::size_t hcd     = hc * d;
    _hcStreams = _ops.allocate(T * hcd * sizeof(float));
    _hcNormed  = _ops.allocate(T * hcd * sizeof(float));
    _hcW1      = _ops.allocate(T * lowrank * sizeof(float));
    _hcW2      = _ops.allocate(T * hcd * sizeof(float));
    _hcInj     = _ops.allocate(T * hc * sizeof(float));
    _hcCapT    = T;
}

// GatedResidual: mixed = weighted stream-mean of grouped-RMSNorm(streams), with
// a low-rank input-mix gate; combine also builds the injection weights _hcInj.
void Qwen4ExpBackend::hcGatedResidual(std::size_t blockIdx, std::size_t T,
                                      const float* normWeight,
                                      const std::string& modulePrefix, bool combine,
                                      BlockBuffers& s, float* mixed) {
    const std::size_t d       = _config.embeddingLength;
    const std::size_t hc      = _config.hcCount;
    const std::size_t lowrank = _config.hcLowrank;
    const std::size_t hcd     = hc * d;
    const float       eps     = _config.rmsNormEps;
    const float       invHc   = 1.0F / static_cast<float>(hc);

    float* const streams = _hcStreams.as<float>();
    float* const normed  = _hcNormed.as<float>();
    float* const w1      = _hcW1.as<float>();
    float* const w2      = _hcW2.as<float>();
    float* const scratch = s.matmulScratch.as<float>();

    // normed = groupedRMSNorm(streams) * (1+w) hc_norm
    _ops.hcGroupedRmsNormAsync(streams, normWeight, normed, T, hc, d, eps);

    // input-mix: w1 = silu(down(normed)/hc); w2 = sigmoid(up(w1))
    const auto& downW = (blockIdx == kMixerBlock)
        ? requireTopT(_weights, modulePrefix + "input_mix_weight_down.weight")
        : requireBlockT(_weights, blockIdx, modulePrefix + "input_mix_weight_down.weight");
    const auto& upW = (blockIdx == kMixerBlock)
        ? requireTopT(_weights, modulePrefix + "input_mix_weight_up.weight")
        : requireBlockT(_weights, blockIdx, modulePrefix + "input_mix_weight_up.weight");
    _gmm.matmulAsync(downW.type, downW.usmPtr, lowrank, hcd, normed, T, w1, scratch);
    _ops.hcSiluScaleAsync(w1, T * lowrank, invHc);
    _gmm.matmulAsync(upW.type, upW.usmPtr, hcd, lowrank, w1, T, w2, scratch);
    _ops.sigmoidInPlaceAsync(w2, T * hcd);

    // mixed = mean_g(w2 * normed)
    _ops.hcWeightedMeanStreamsAsync(w2, normed, mixed, T, hc, d);

    if (!combine) {
        return;
    }
    // inj = 2*sigmoid(block_inject(normed)/hc)
    const auto& biW = requireBlockT(_weights, blockIdx,
                                    modulePrefix + "block_inject_weight.weight");
    float* const inj = _hcInj.as<float>();
    _gmm.matmulAsync(biW.type, biW.usmPtr, hc, hcd, normed, T, inj, scratch);
    _ops.mulScalarAsync(inj, invHc, T * hc);
    _ops.sigmoidInPlaceAsync(inj, T * hc);
    _ops.mulScalarAsync(inj, 2.0F, T * hc);
}

void Qwen4ExpBackend::blockInputNorm(std::size_t blockIdx, const float* x,
                                     std::size_t T, const float* normWeight,
                                     BlockBuffers& s, float* normBuf, bool isAttn) {
    growHcScratch(T);
    const std::size_t d  = _config.embeddingLength;
    const std::size_t hc = _config.hcCount;

    // Forward start: broadcast this step's embedding into the hc streams. (PLE
    // layer-1 injection is deferred to I-4; without it the streams start as the
    // plain repeated embedding.)
    if (blockIdx == 0 && isAttn) {
        _ops.hcStreamBroadcastAsync(x, _hcStreams.as<float>(), T, hc, d);
    }

    const std::string prefix =
        isAttn ? "attn_hyper_connection." : "mlp_hyper_connection.";
    hcGatedResidual(blockIdx, T, normWeight, prefix, /*combine=*/true, s, normBuf);

    if (blockIdx == 0) {
        dumpNorm(_ops, _gmm, "blk0 stream-in", _hcStreams.as<float>(), T * hc * d,
                 /*nRows=*/T);
        dumpNorm(_ops, _gmm, isAttn ? "blk0 attn-mixed" : "blk0 mlp-mixed",
                 normBuf, T * d);
    }
}

void Qwen4ExpBackend::blockResidualAdd(std::size_t blockIdx, float* /*x*/,
                                       const float* moduleOut, std::size_t T,
                                       BlockBuffers& /*s*/, bool isAttn) {
    const std::size_t d  = _config.embeddingLength;
    const std::size_t hc = _config.hcCount;
    // Scatter the module output into every stream: H_g += inj_g * out.
    _ops.hcInjectScatterAsync(_hcStreams.as<float>(), moduleOut,
                              _hcInj.as<float>(), T, hc, d);
    if (!isAttn) {  // end of the block — trajectory of the stream-state norm.
        dumpNorm(_ops, _gmm, ("blk" + std::to_string(blockIdx) + " out").c_str(),
                 _hcStreams.as<float>(), T * hc * d, /*nRows=*/T);
    }
}

void Qwen4ExpBackend::setPleTable(nvfp4::PleNgramTable&& table, int pleGgufLayer,
                                  std::int64_t eosTokenId) {
    _pleTable        = std::move(table);
    _pleGgufLayer    = pleGgufLayer;
    _pleEos          = eosTokenId;
    _pleNgramSize    = static_cast<int>(_config.ngramSize);
    _pleHeadsPerNgram = static_cast<int>(_config.headsPerNgram);
    _pleNgramHeads   = (_pleNgramSize - 1) * _pleHeadsPerNgram;
    _pleEmbedDim     = static_cast<int>(_config.pleEmbedDim);

    // Multipliers: read verbatim from the checkpoint (vocab_size-dependent).
    _pleMult = _pleTable.layerMultipliers();
    // Per-head prime vocab sizes + offsets (ple_layer_index = 0; deterministic).
    const std::int64_t base = static_cast<std::int64_t>(_config.ngramVocabBase);
    _pleHeadVocab.assign(_pleNgramHeads, 0);
    _pleHeadOff.assign(_pleNgramHeads, 0);
    std::int64_t total = 0;
    for (int h = 0; h < _pleNgramHeads; ++h) {
        _pleHeadVocab[h] = findNthPrimeAfter(base - 1, h + 1);
        _pleHeadOff[h] = total;
        total += _pleHeadVocab[h];
    }
    _pleCtx = {static_cast<std::int32_t>(eosTokenId), static_cast<std::int32_t>(eosTokenId)};
    _pleCtxInit = true;
}

void Qwen4ExpBackend::prepareForward(std::span<const std::int32_t> tokIds,
                                     const float* /*hiddenStates*/, std::size_t T) {
    _pleBatchedNSeq = 0;   // single-sequence PLE path
    _pleSlotStore   = -1;  // no per-slot store-back
    _pleTokens.assign(tokIds.begin(), tokIds.end());
    // Prefill (T>1) starts a new sequence → reset the rolling n-gram context to
    // EOS (matches the reference's initial eos-pad). Decode (T=1) carries it.
    if (T > 1) {
        _pleCtx = {static_cast<std::int32_t>(_pleEos), static_cast<std::int32_t>(_pleEos)};
    }
}

void Qwen4ExpBackend::resetForwardContext() {
    // Sequence start: the rolling 2-token n-gram context starts as the
    // reference's initial eos-pad (mirrors setPleTable / prepareForward T>1).
    _pleCtx = {static_cast<std::int32_t>(_pleEos),
               static_cast<std::int32_t>(_pleEos)};
    _pleCtxInit = true;
    _pleBatchedNSeq = 0;
    for (auto& c : _pleCtxSlot) {
        c = {static_cast<std::int32_t>(_pleEos),
             static_cast<std::int32_t>(_pleEos)};
    }
}

// I-9b: per-slot batched PLE context. One token per active slot this step; slots
// flagged isSeqStart reset their own rolling n-gram context to the eos-pad.
void Qwen4ExpBackend::prepareForwardBatched(std::span<const std::int32_t> tokIds,
                                            std::span<const std::uint8_t> isSeqStart,
                                            std::size_t nSeq) {
    _pleBatchedNSeq = nSeq;
    _pleSlotStore   = -1;
    _pleTokens.assign(tokIds.begin(),
                      tokIds.begin() + static_cast<std::ptrdiff_t>(nSeq));
    if (_pleCtxSlot.size() < nSeq) {
        _pleCtxSlot.resize(nSeq, {static_cast<std::int32_t>(_pleEos),
                                  static_cast<std::int32_t>(_pleEos)});
    }
    for (std::size_t s = 0; s < nSeq; ++s) {
        if (!isSeqStart.empty() && isSeqStart[s] != 0) {
            _pleCtxSlot[s] = {static_cast<std::int32_t>(_pleEos),
                              static_cast<std::int32_t>(_pleEos)};
        }
    }
}

// I-9b: single-slot prefill. Prefill goes through the single-session runBlock, so
// pleForward uses the single-session n-gram path over _pleCtx; load the slot's
// own rolling context here and mark it for store-back after the roll.
void Qwen4ExpBackend::prepareForwardSlot(std::size_t slot,
                                         std::span<const std::int32_t> tokens,
                                         bool seqStart) {
    _pleBatchedNSeq = 0;
    _pleTokens.assign(tokens.begin(), tokens.end());
    if (_pleCtxSlot.size() <= slot) {
        _pleCtxSlot.resize(slot + 1, {static_cast<std::int32_t>(_pleEos),
                                      static_cast<std::int32_t>(_pleEos)});
    }
    if (seqStart) {
        _pleCtxSlot[slot] = {static_cast<std::int32_t>(_pleEos),
                             static_cast<std::int32_t>(_pleEos)};
    }
    _pleCtx       = _pleCtxSlot[slot];
    _pleSlotStore = static_cast<std::ptrdiff_t>(slot);
}

void Qwen4ExpBackend::blockEnter(std::size_t blockIdx, float* /*x*/, std::size_t T,
                                 BlockBuffers& s) {
    if (static_cast<int>(blockIdx) == _pleGgufLayer && _pleTable.isOpen()) {
        pleForward(T, s);
    }
}

void Qwen4ExpBackend::computeNgramIds(std::size_t T,
                                      std::vector<std::int64_t>& outIds) const {
    // history = [ctx(2) | tokens(T)]
    std::vector<std::int32_t> history;
    history.reserve(2 + T);
    history.push_back(_pleCtx[0]);
    history.push_back(_pleCtx[1]);
    history.insert(history.end(), _pleTokens.begin(), _pleTokens.begin() + T);
    const int Lh = static_cast<int>(history.size());
    const int nh = _pleNgramHeads;

    std::vector<std::vector<std::int32_t>> shifted(_pleNgramSize);
    for (int s = 0; s < _pleNgramSize; ++s)
        shifted[s] = shiftRightIgnoreEos(history, s, static_cast<std::int32_t>(_pleEos));

    std::vector<std::int64_t> full(static_cast<std::size_t>(Lh) * nh);
    for (int ngram = 2; ngram <= _pleNgramSize; ++ngram) {
        const int start = (ngram - 2) * _pleHeadsPerNgram;
        for (int i = 0; i < Lh; ++i) {
            std::uint64_t mixed = static_cast<std::uint64_t>(shifted[0][i])
                                * static_cast<std::uint64_t>(_pleMult[0]);
            for (int pos = 1; pos < ngram; ++pos)
                mixed ^= static_cast<std::uint64_t>(shifted[pos][i])
                       * static_cast<std::uint64_t>(_pleMult[pos]);
            const std::int64_t m = static_cast<std::int64_t>(mixed);
            for (int j = 0; j < _pleHeadsPerNgram; ++j) {
                const std::int64_t hv = _pleHeadVocab[start + j];
                const std::int64_t r = ((m % hv) + hv) % hv;
                full[static_cast<std::size_t>(i) * nh + start + j] = r + _pleHeadOff[start + j];
            }
        }
    }
    outIds.resize(static_cast<std::size_t>(T) * nh);
    for (std::size_t t = 0; t < T; ++t)
        for (int j = 0; j < nh; ++j)
            outIds[t * nh + j] = full[static_cast<std::size_t>(Lh - T + t) * nh + j];
}

// I-9b: batched n-gram ids — each of the nSeq rows is one token of an independent
// slot, hashed against that slot's own rolling 2-token context (_pleCtxSlot). The
// per-slot [ctx0, ctx1, token] history + last-position extraction matches the
// single-session decode (T=1) result exactly, so conc=1 == single-session.
void Qwen4ExpBackend::computeNgramIdsBatched(
        std::size_t nSeq, std::vector<std::int64_t>& outIds) const {
    const int          nh  = _pleNgramHeads;
    const std::int32_t eos = static_cast<std::int32_t>(_pleEos);
    outIds.resize(nSeq * static_cast<std::size_t>(nh));
    for (std::size_t s = 0; s < nSeq; ++s) {
        const std::vector<std::int32_t> hist = {
            _pleCtxSlot[s][0], _pleCtxSlot[s][1], _pleTokens[s]};
        std::vector<std::int32_t> shiftedLast(static_cast<std::size_t>(_pleNgramSize));
        for (int sh = 0; sh < _pleNgramSize; ++sh) {
            const auto sv = shiftRightIgnoreEos(hist, sh, eos);
            shiftedLast[static_cast<std::size_t>(sh)] = sv.back();
        }
        for (int ngram = 2; ngram <= _pleNgramSize; ++ngram) {
            const int start = (ngram - 2) * _pleHeadsPerNgram;
            std::uint64_t mixed = static_cast<std::uint64_t>(shiftedLast[0])
                                * static_cast<std::uint64_t>(_pleMult[0]);
            for (int pos = 1; pos < ngram; ++pos)
                mixed ^= static_cast<std::uint64_t>(
                             shiftedLast[static_cast<std::size_t>(pos)])
                       * static_cast<std::uint64_t>(
                             _pleMult[static_cast<std::size_t>(pos)]);
            const std::int64_t m = static_cast<std::int64_t>(mixed);
            for (int j = 0; j < _pleHeadsPerNgram; ++j) {
                const std::int64_t hv =
                    _pleHeadVocab[static_cast<std::size_t>(start + j)];
                const std::int64_t r = ((m % hv) + hv) % hv;
                outIds[s * static_cast<std::size_t>(nh)
                       + static_cast<std::size_t>(start + j)]
                    = r + _pleHeadOff[static_cast<std::size_t>(start + j)];
            }
        }
    }
}

void Qwen4ExpBackend::growPleScratch(std::size_t T) {
    if (T <= _pleCapT) return;
    const std::size_t d = _config.embeddingLength;
    const std::size_t hcd = _config.hcCount * d;
    const std::size_t embDim = static_cast<std::size_t>(_pleEmbedDim);
    _pleEmb     = _ops.allocate(T * embDim * sizeof(float));
    _pleKey     = _ops.allocate(T * hcd * sizeof(float));
    _pleVal     = _ops.allocate(T * d * sizeof(float));
    _pleKeyN    = _ops.allocate(T * hcd * sizeof(float));
    _pleQryN    = _ops.allocate(T * hcd * sizeof(float));
    _pleGated   = _ops.allocate(T * hcd * sizeof(float));
    _pleGvn     = _ops.allocate(T * hcd * sizeof(float));
    _pleConvOut = _ops.allocate(T * hcd * sizeof(float));
    _pleEmbHost.assign(T * embDim, 0.0F);
    _pleIdsHost.assign(T * static_cast<std::size_t>(_pleNgramHeads), 0);
    _pleCapT = T;
}

// PLE forward: host n-gram gather -> device projections + gate + dilated conv,
// added into the hyper-connection stream state _hcStreams. (5.27 I-4.)
void Qwen4ExpBackend::pleForward(std::size_t T, BlockBuffers& s) {
    growPleScratch(T);
    const std::size_t d = _config.embeddingLength;
    const std::size_t hc = _config.hcCount;
    const std::size_t hcd = hc * d;
    const std::size_t embDim = static_cast<std::size_t>(_pleEmbedDim);
    const float eps = _config.rmsNormEps;
    const std::size_t K = _config.pleConvKernelSize;
    const std::size_t dilation = static_cast<std::size_t>(_pleNgramSize);
    const std::size_t stateLen = (K - 1) * dilation;

    // 1. host: n-gram ids -> gather+dequant [T, embDim] F32 -> device.
    if (_pleBatchedNSeq > 0) {
        computeNgramIdsBatched(T, _pleIdsHost);   // T == nSeq (one row per slot)
    } else {
        computeNgramIds(T, _pleIdsHost);
    }
    _pleTable.gatherDequantF32(_pleIdsHost, _pleEmbHost.data());
    _ops.uploadHostBytes(_pleEmb.get(), _pleEmbHost.data(), T * embDim * sizeof(float));

    float* const emb     = _pleEmb.as<float>();
    float* const scratch = s.matmulScratch.as<float>();
    float* const streams = _hcStreams.as<float>();

    const auto& kW  = requireBlockT(_weights, _pleGgufLayer, "ple.key_proj.weight");
    const auto& vW  = requireBlockT(_weights, _pleGgufLayer, "ple.value_proj.weight");
    const auto& nkW = requireBlockT(_weights, _pleGgufLayer, "ple.norm_key.weight");
    const auto& nqW = requireBlockT(_weights, _pleGgufLayer, "ple.norm_query.weight");
    const auto& ncW = requireBlockT(_weights, _pleGgufLayer, "ple.norm_conv.weight");
    const auto& cW  = requireBlockT(_weights, _pleGgufLayer, "ple.conv1d.weight");

    // 2. key = norm_key(key_proj(emb)); value = value_proj(emb);
    //    query = norm_query(streams).
    _gmm.matmulAsync(kW.type, kW.usmPtr, hcd, embDim, emb, T, _pleKey.as<float>(), scratch);
    _ops.hcGroupedRmsNormAsync(_pleKey.as<float>(),
                               static_cast<const float*>(nkW.usmPtr),
                               _pleKeyN.as<float>(), T, hc, d, eps);
    _gmm.matmulAsync(vW.type, vW.usmPtr, d, embDim, emb, T, _pleVal.as<float>(), scratch);
    _ops.hcGroupedRmsNormAsync(streams, static_cast<const float*>(nqW.usmPtr),
                               _pleQryN.as<float>(), T, hc, d, eps);

    // 3. signed-sqrt gate -> gated value; norm_conv; dilated conv+silu.
    _ops.pleGateAsync(_pleKeyN.as<float>(), _pleQryN.as<float>(), _pleVal.as<float>(),
                      _pleGated.as<float>(), T, hc, d);
    _ops.hcGroupedRmsNormAsync(_pleGated.as<float>(),
                               static_cast<const float*>(ncW.usmPtr),
                               _pleGvn.as<float>(), T, hc, d, eps);
    // 5.27.11.2: the dilated (dilation=ngramSize) causal conv1d convolves along
    // the T dimension. In SINGLE-SESSION T is one sequence's tokens (correct
    // temporal conv). In BATCHED DECODE the T=nSeq rows are INDEPENDENT slots
    // (one token each), so a temporal conv would reach across slots — with
    // dilation=3 a row at position p pulls the slot at p-3, mixing slots at
    // nSeq>=4 (localized via per-row diag: rows identical through blk0, diverge
    // at the PLE conv). Run the conv PER ROW as an independent T=1 sequence
    // (zero state), matching the single-session decode step. Prefill still goes
    // through the single-session runBlock path (T=prompt), unaffected.
    if (_pleBatchedNSeq > 0) {
        for (std::size_t r = 0; r < _pleBatchedNSeq; ++r) {
            _ops.pleConvSiluAsync(_pleGvn.as<float>() + r * hcd, /*state=*/nullptr,
                                  static_cast<const float*>(cW.usmPtr),
                                  _pleConvOut.as<float>() + r * hcd,
                                  /*T=*/1, hcd, K, dilation, stateLen);
        }
    } else {
        _ops.pleConvSiluAsync(_pleGvn.as<float>(), /*state=*/nullptr,
                              static_cast<const float*>(cW.usmPtr),
                              _pleConvOut.as<float>(), T, hcd, K, dilation, stateLen);
    }

    dumpNorm(_ops, _gmm, "ple emb", emb, T * embDim);
    dumpNorm(_ops, _gmm, "ple gated", _pleGated.as<float>(), T * hcd);
    dumpNorm(_ops, _gmm, "ple conv", _pleConvOut.as<float>(), T * hcd);

    // 4. streams += gated + conv.
    _ops.addResidualAsync(streams, _pleGated.as<float>(), T * hcd);
    _ops.addResidualAsync(streams, _pleConvOut.as<float>(), T * hcd);
    dumpNorm(_ops, _gmm, "ple streams-after", streams, T * hcd, /*nRows=*/T);

    // 5. roll the n-gram context forward (per-slot in batched mode).
    if (_pleBatchedNSeq > 0) {
        for (std::size_t s = 0; s < _pleBatchedNSeq && s < _pleCtxSlot.size(); ++s) {
            _pleCtxSlot[s][0] = _pleCtxSlot[s][1];
            _pleCtxSlot[s][1] = _pleTokens[s];
        }
    } else if (T >= 2) {
        _pleCtx[0] = _pleTokens[T - 2];
        _pleCtx[1] = _pleTokens[T - 1];
    } else if (T == 1) {
        _pleCtx[0] = _pleCtx[1];
        _pleCtx[1] = _pleTokens[0];
    }
    // I-9b: single-slot prefill — persist the rolled context back to the slot so
    // the next chunk of the same request continues its n-gram history.
    if (_pleBatchedNSeq == 0 && _pleSlotStore >= 0 &&
        static_cast<std::size_t>(_pleSlotStore) < _pleCtxSlot.size()) {
        _pleCtxSlot[static_cast<std::size_t>(_pleSlotStore)] = _pleCtx;
    }
}

void Qwen4ExpBackend::collapseHyperStreams(std::size_t T, BlockBuffers& s,
                                           float* out) {
    growHcScratch(T);
    // The mixer's hc_norm arrives via the model.norm remap as gguf output_norm.
    const auto& mixerNorm = requireTopT(_weights, "output_norm.weight");
    hcGatedResidual(kMixerBlock, T,
                    static_cast<const float*>(mixerNorm.usmPtr),
                    "hyper_connection_mixer.", /*combine=*/false, s, out);
    dumpNorm(_ops, _gmm, "mixer out (pre-lmhead)", out, T * _config.embeddingLength);
}

} // namespace mimirmind::runtime::arch
