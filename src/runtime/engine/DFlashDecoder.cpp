// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/engine/DFlashDecoder.hpp"

#include "runtime/InferenceEngine.hpp"
#include "runtime/KvCache.hpp"
#include "runtime/SsmState.hpp"
#include "runtime/arch/Qwen35MoeBackend.hpp"
#include "runtime/dflash/DFlashDraftRunner.hpp"
#include "compute/ComputeMatmul.hpp"
#include "compute/Embedding.hpp"
#include "core/gguf/GgufReader.hpp"
#include "core/log/Log.hpp"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>

namespace mimirmind::runtime::engine {

namespace {

[[nodiscard]] std::int32_t argmaxHost(const std::vector<float>& row) {
    std::size_t best = 0;
    float       bv   = row.empty() ? 0.0F : row[0];
    for (std::size_t v = 1; v < row.size(); ++v) {
        if (row[v] > bv) { bv = row[v]; best = v; }
    }
    return static_cast<std::int32_t>(best);
}

/// Read `mask_token_id` from the drafter's config.json (fallback 248077 — the
/// Qwen3.6 DFlash checkpoint value — if the file is absent / unparseable).
[[nodiscard]] std::int32_t readMaskToken(std::string_view dir) {
    const std::int32_t kFallback = 248077;
    std::string path{dir};
    if (!path.empty() && path.back() != '/') path.push_back('/');
    path += "config.json";
    try {
        std::ifstream in(path);
        if (!in) return kFallback;
        nlohmann::json j;
        in >> j;
        if (j.contains("mask_token_id") && j["mask_token_id"].is_number_integer()) {
            return j["mask_token_id"].get<std::int32_t>();
        }
    } catch (const std::exception&) {
        // fall through to the checkpoint default
    }
    return kFallback;
}

} // namespace

DFlashDecoder::DFlashDecoder(InferenceEngine& engine) : _e{engine} {}
DFlashDecoder::~DFlashDecoder() = default;

void DFlashDecoder::ensureLoaded(std::string_view drafterDir) {
    if (_loaded) {
        return;
    }
    _model.load(drafterDir, *_e._ops);

    // Borrow the target embed (noise embedding) + lm_head (draft logits).
    const auto* tokEmb = _e._weights->find("token_embd.weight");
    if (tokEmb == nullptr) {
        tokEmb = _e._weights->find("tok_embeddings.weight");
    }
    const auto* lmHead = _e._weights->find("output.weight");
    if (lmHead == nullptr) {
        lmHead = tokEmb;
    }
    if (tokEmb == nullptr || lmHead == nullptr) {
        throw std::runtime_error(
            "DFlashDecoder: target token_embd / output weight missing");
    }
    _model.borrowTarget(tokEmb, lmHead);

    _runner = std::make_unique<dflash::DFlashDraftRunner>(_model, *_e._ops, *_e._gmm);

    const auto& c = _model.config();
    _d    = c.hidden;
    _taps = c.taps;
    if (_taps != 8) {
        throw std::runtime_error("DFlashDecoder: expected 8 taps, got "
                                 + std::to_string(_taps));
    }
    _vocabLm  = lmHead->dimensions.size() >= 2 ? lmHead->dimensions[1]
                                               : _e._tokenizer.vocabSize();
    _vocabEmb = tokEmb->dimensions.size() >= 2 ? tokEmb->dimensions[1]
                                               : _vocabLm;
    _maskTok  = readMaskToken(drafterDir);

    const std::size_t maxPos = _e._maxContextTokens;
    _tapSink.clear();
    _tapPtr.clear();
    _tapSink.reserve(_taps);
    _tapPtr.reserve(_taps);
    for (std::size_t k = 0; k < _taps; ++k) {
        _tapSink.push_back(_e._ops->allocate(maxPos * _d * sizeof(float)));
        _tapPtr.push_back(_tapSink.back().as<float>());
    }
    _ctxHidden = _e._ops->allocate(maxPos * _taps * _d * sizeof(float));
    _noise     = _e._ops->allocate(kMaxBlock * _d * sizeof(float));
    _draftOut  = _e._ops->allocate(kMaxBlock * _d * sizeof(float));
    _draftLogits    = _e._ops->allocate((kMaxBlock - 1) * _vocabLm * sizeof(float));
    _draftArgmaxDev = _e._ops->allocate((kMaxBlock - 1) * sizeof(std::int32_t));
    _draftArgmaxHost.assign(kMaxBlock - 1, 0);

    // GDN ReplaySSM fold (MIMIRMIND_DFLASH_FOLD): allocate per-recurrent-block
    // capture rings and arm the backend so a partial accept can fold the
    // accepted prefix instead of re-forwarding the trunk.
    _foldMode = std::getenv("MIMIRMIND_DFLASH_FOLD") != nullptr;
    if (_foldMode) {
        auto* qb = dynamic_cast<arch::Qwen35MoeBackend*>(_e._backend.get());
        if (qb == nullptr) {
            _foldMode = false;
        } else {
            _hV      = qb->gdnVHeads();
            _sState  = qb->gdnStateSize();
            _convDim = qb->gdnConvDim();
            const std::size_t dConv = qb->gdnConvKernel();
            const std::size_t stateRows = dConv > 0 ? dConv - 1 : 0;
            _convStateElems = stateRows * _convDim;
            _recurBlocks.clear();
            for (std::size_t b = 0; b < qb->layerCount(); ++b) {
                if (qb->isRecurrent(b)) _recurBlocks.push_back(b);
            }
            const std::size_t nR = _recurBlocks.size();
            _capK.clear(); _capV.clear(); _capG.clear(); _capB.clear(); _capConv.clear();
            for (std::size_t r = 0; r < nR; ++r) {
                _capK.push_back(_e._ops->allocate(kMaxBlock * _hV * _sState * sizeof(float)));
                _capV.push_back(_e._ops->allocate(kMaxBlock * _hV * _sState * sizeof(float)));
                _capG.push_back(_e._ops->allocate(kMaxBlock * _hV * sizeof(float)));
                _capB.push_back(_e._ops->allocate(kMaxBlock * _hV * sizeof(float)));
                _capConv.push_back(
                    _e._ops->allocate((stateRows + kMaxBlock) * _convDim * sizeof(float)));
            }
            MM_LOG_INFO("dflash",
                        "DFlash ReplaySSM fold ENABLED — {} recurrent blocks, "
                        "hV={} S={} convDim={} dConv={}",
                        nR, _hV, _sState, _convDim, dConv);
        }
    }

    MM_LOG_INFO("dflash",
                "DFlashDecoder ready — drafter={} layers={} hidden={} taps={} "
                "vocab={} mask={}",
                std::string{drafterDir}, c.numLayers, _d, _taps, _vocabLm,
                _maskTok);
    _loaded = true;
}

void DFlashDecoder::feedContext(std::size_t p0, std::size_t p1) {
    const std::size_t rowC = _taps * _d;
    float* const      ctx  = _ctxHidden.as<float>();
    for (std::size_t pos = p0; pos < p1; ++pos) {
        for (std::size_t k = 0; k < _taps; ++k) {
            _e._ops->appendMemoryCopy(ctx + pos * rowC + k * _d,
                                      _tapPtr[k] + pos * _d,
                                      _d * sizeof(float));
        }
    }
}

std::vector<std::int32_t>
DFlashDecoder::generate(std::span<const std::int32_t> promptIds,
                        std::size_t maxNew, std::size_t draftN,
                        std::int32_t eosId, std::string_view drafterDir,
                        std::size_t* draftedOut, std::size_t* acceptedOut) {
    auto* qb = dynamic_cast<arch::Qwen35MoeBackend*>(_e._backend.get());
    if (qb == nullptr) {
        throw std::runtime_error("generateDflash: requires CUDA qwen35moe target");
    }
    if (promptIds.empty() || maxNew == 0) {
        return {};
    }
    if (draftN == 0) draftN = 1;
    if (draftN > kMaxBlock - 1) draftN = kMaxBlock - 1;

    ensureLoaded(drafterDir);

    const auto* lmHead = _model.lmHead();
    const auto* tokEmb = _model.embedTokens();

    // ---- Prefill: trunk forward with the hidden tap live ------------------
    _e.resetCache();
    qb->configureHiddenTap(
        std::span<const std::size_t>{kTapLayers, _taps},
        std::span<float* const>{_tapPtr.data(), _tapPtr.size()});

    // Arm the GDN ReplaySSM capture for this decode (disarmed at the end so a
    // plain generate() on the same backend stays prod-inert).
    if (_foldMode && !_recurBlocks.empty()) {
        std::vector<float*> kS, vS, gS, bS, cS;
        const std::size_t nR = _capK.size();
        kS.reserve(nR); vS.reserve(nR); gS.reserve(nR); bS.reserve(nR); cS.reserve(nR);
        for (std::size_t r = 0; r < nR; ++r) {
            kS.push_back(_capK[r].as<float>());
            vS.push_back(_capV[r].as<float>());
            gS.push_back(_capG[r].as<float>());
            bS.push_back(_capB[r].as<float>());
            cS.push_back(_capConv[r].as<float>());
        }
        qb->configureGdnCapture(_recurBlocks, kS, vS, gS, bS, cS, kMaxBlock);
    }

    std::vector<std::int32_t> prompt(promptIds.begin(), promptIds.end());
    auto pf = _e.forwardVerify(prompt);          // tap sinks fill rows [0, P)
    const std::size_t P = prompt.size();
    _e.commitVerified(prompt);
    _e._ops->flush();
    feedContext(0, P);
    std::size_t ctxLen = P;

    std::int32_t token0 = argmaxHost(pf.back());

    std::vector<std::int32_t> out;
    std::size_t drafted = 0, accepted = 0;
    bool stop = false;
    auto emit = [&](std::int32_t t) -> bool {
        out.push_back(t);
        if (eosId >= 0 && t == eosId) { stop = true; return false; }
        return out.size() < maxNew;
    };

    float* const logitsSc = _e._logitsScH.as<float>();

    // Conditioning diagnostic (MIMIRMIND_DFLASH_DEBUG): dump per-tap RMS, fused
    // context RMS, and drafts-vs-true-greedy for the first rounds so we can see
    // whether the tap hidden is sane and how far the draft is from the target.
    const bool  dbg      = std::getenv("MIMIRMIND_DFLASH_DEBUG") != nullptr;
    std::size_t roundIdx = 0;

    // Per-phase timing breakdown (MIMIRMIND_DFLASH_TIMING) to localise the
    // per-round cost: draft-forward vs verify-forward vs the D2H/commit rest.
    const bool tmg = std::getenv("MIMIRMIND_DFLASH_TIMING") != nullptr;
    double tDraft = 0, tReadout = 0, tVerify = 0, tRest = 0;
    std::size_t nTmg = 0;
    auto nowMs = [] {
        return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
    };
    auto rmsOf = [&](const float* dev, std::size_t n) -> double {
        std::vector<float> h(n);
        _e._ops->flush();
        _e._ops->readbackToHost(h.data(), dev, n * sizeof(float));
        double s = 0.0;
        for (float v : h) s += static_cast<double>(v) * v;
        return std::sqrt(s / static_cast<double>(n > 0 ? n : 1));
    };

    while (out.size() < maxNew && !stop) {
        const std::size_t K = std::min(draftN, maxNew - out.size());

        // ---- Draft the whole K-token block in ONE forward ----------------
        // block = [token0, mask x K]; embed via borrowed target embed_tokens.
        std::vector<std::int32_t> block;
        block.reserve(K + 1);
        block.push_back(token0);
        for (std::size_t i = 0; i < K; ++i) block.push_back(_maskTok);

        const double _t0 = tmg ? nowMs() : 0.0;
        compute::embeddingLookup(tokEmb->type, tokEmb->usmPtr, _d, _vocabEmb,
                                 std::span<const std::int32_t>{block},
                                 _noise.as<float>());

        _runner->draftForward(_noise.as<float>(), _ctxHidden.as<float>(),
                              K + 1, ctxLen, _draftOut.as<float>());
        const double _t1 = tmg ? nowMs() : 0.0;

        // Readout: borrowed target lm_head on draft positions 1..K. The runner
        // already applied the drafter's final norm, so NO target output_norm.
        // Batched draft readout (Hebel 1): one lm_head matmul over draft
        // positions 1..K -> [K, vocab], on-device per-row argmax, then read back
        // only K token ids — replaces K synchronous full-vocab D2H readbacks.
        std::vector<std::int32_t> drafts;
        drafts.reserve(K);
        float* const draftOut = _draftOut.as<float>();
        _e._gmm->matmulAsync(lmHead->type, lmHead->usmPtr, _vocabLm, _d,
                             draftOut + _d, K, _draftLogits.as<float>(), logitsSc);
        _e._ops->argmaxRowsAsync(_draftLogits.as<float>(),
                                 _draftArgmaxDev.as<std::int32_t>(), K, _vocabLm);
        _e._ops->flush();
        _e._ops->readbackToHost(_draftArgmaxHost.data(), _draftArgmaxDev.get(),
                                K * sizeof(std::int32_t));
        for (std::size_t i = 0; i < K; ++i) {
            drafts.push_back(_draftArgmaxHost[i]);
            ++drafted;
        }
        const double _t2 = tmg ? nowMs() : 0.0;

        // ---- Verify: one trunk forward on [token0, drafts...] ------------
        std::vector<std::int32_t> vtoks;
        vtoks.reserve(K + 1);
        vtoks.push_back(token0);
        vtoks.insert(vtoks.end(), drafts.begin(), drafts.end());

        // Snapshot the trunk GatedDeltaNet recurrent + conv state (advances
        // monolithically over K+1; restored on partial accept). Mirror MTP.
        SsmState* const ssm = _e._ssmState.get();
        std::size_t stBytes = 0, cvBytes = 0;
        if (ssm != nullptr) {
            stBytes = ssm->blockCount() * ssm->stateLayerStride()     * sizeof(float);
            cvBytes = ssm->blockCount() * ssm->convStateLayerStride() * sizeof(float);
            if (_ssmBak.get() == nullptr) {
                _ssmBak  = _e._ops->allocate(stBytes);
                _convBak = _e._ops->allocate(cvBytes);
            }
            _e._ops->appendMemoryCopy(_ssmBak.get(),  ssm->statePtr(),     stBytes);
            _e._ops->appendMemoryCopy(_convBak.get(), ssm->convStatePtr(), cvBytes);
        }

        const std::size_t preKvLen = _e._kvCache->length();   // == ctxLen
        auto vl = _e.forwardVerify(vtoks);                    // tap rows [ctxLen, ctxLen+K]
        const double _t3 = tmg ? nowMs() : 0.0;

        // ---- Accept longest greedy prefix -------------------------------
        std::size_t a = 0;
        for (std::size_t i = 0; i < K; ++i) {
            if (argmaxHost(vl[i]) == drafts[i]) ++a; else break;
        }
        const std::int32_t corrected = argmaxHost(vl[a]);
        accepted += a;

        if (dbg && roundIdx < 3) {
            std::printf("[dflash-dbg] round=%zu ctxLen=%zu token0=%d K=%zu\n",
                        roundIdx, ctxLen, token0, K);
            for (std::size_t k = 0; k < _taps; ++k) {
                std::printf("  tap[%zu](L=%zu) rms=%.4f\n", k, kTapLayers[k],
                            rmsOf(_tapPtr[k], ctxLen * _d));
            }
            {
                compute::ComputeBuffer cp = _e._ops->allocate(ctxLen * _d * sizeof(float));
                _runner->materializeContext(_ctxHidden.as<float>(), ctxLen,
                                            cp.as<float>());
                std::printf("  ctxProj rms=%.4f (expect ~1 after hidden_norm)\n",
                            rmsOf(cp.as<float>(), ctxLen * _d));
            }
            std::printf("  drafts    =");
            for (std::int32_t d : drafts) std::printf(" %d", d);
            std::printf("\n  tgt-greedy=");
            for (std::size_t i = 0; i <= K; ++i)
                std::printf(" %d", argmaxHost(vl[i]));
            std::printf("\n  accepted=%zu\n", a);
            std::fflush(stdout);
        }
        ++roundIdx;

        std::vector<std::int32_t> committed(
            vtoks.begin(), vtoks.begin() + static_cast<std::ptrdiff_t>(a + 1));

        if (a == K) {
            _e.commitVerified(committed);            // tap rows already committed
        } else if (_foldMode && ssm != nullptr) {
            // ReplaySSM: restore the SSM recurrent state to the pre-window
            // checkpoint, fold ONLY the accepted prefix per recurrent block, and
            // slice the committed conv state from the captured conv input — no
            // trunk re-forward. The verify forward already wrote provisional KV +
            // tap rows for all K+1 positions; commitVerified(a+1) commits exactly
            // the accepted prefix (KV) and feedContext reads the accepted taps.
            _e._ops->appendMemoryCopy(ssm->statePtr(), _ssmBak.get(), stBytes);
            const std::size_t aLen = a + 1;
            for (std::size_t r = 0; r < _recurBlocks.size(); ++r) {
                const std::size_t b = _recurBlocks[r];
                float* const stB = ssm->statePtr() + b * ssm->stateLayerStride();
                _e._ops->gatedDeltaNetFoldAsync(
                    _capK[r].as<float>(), _capV[r].as<float>(),
                    _capG[r].as<float>(), _capB[r].as<float>(),
                    stB, aLen, _hV, _sState);
                float* const cvB =
                    ssm->convStatePtr() + b * ssm->convStateLayerStride();
                _e._ops->appendMemoryCopy(
                    cvB, _capConv[r].as<float>() + aLen * _convDim,
                    _convStateElems * sizeof(float));
            }
            _e.commitVerified(committed);
        } else {
            // Partial accept (re-forward path): restore SSM, undo provisional KV,
            // re-forward the accepted prefix so KV + SSM + tap sinks land on the
            // committed suffix.
            if (ssm != nullptr) {
                _e._ops->appendMemoryCopy(ssm->statePtr(),     _ssmBak.get(),  stBytes);
                _e._ops->appendMemoryCopy(ssm->convStatePtr(), _convBak.get(), cvBytes);
            }
            _e._kvCache->truncate(preKvLen);
            (void)_e.forwardVerify(committed);       // tap rows [ctxLen, ctxLen+a]
            _e.commitVerified(committed);
        }
        _e._ops->flush();
        feedContext(preKvLen, preKvLen + a + 1);
        ctxLen = preKvLen + a + 1;
        if (tmg) {
            tDraft   += _t1 - _t0;   // embed + 6-layer draft forward
            tReadout += _t2 - _t1;   // batched lm_head + argmax + K-int D2H
            tVerify  += _t3 - _t2;   // ssm snapshot + verify trunk forward (M=K+1)
            tRest    += nowMs() - _t3; // accept scan + partial re-forward + commit + feed
            ++nTmg;
        }

        // ---- Emit token0 + accepted drafts; corrected -> next token0 -----
        bool cont = emit(token0);
        for (std::size_t i = 0; i < a && cont; ++i) cont = emit(drafts[i]);
        token0 = corrected;
        if (!cont) break;
    }

    qb->clearHiddenTap();
    if (_foldMode) qb->clearGdnCapture();

    if (tmg && nTmg > 0) {
        const double tot = tDraft + tReadout + tVerify + tRest;
        std::printf("[dflash-timing] rounds=%zu total=%.1fms/round | "
                    "draft=%.1f (%.0f%%) readout=%.1f (%.0f%%) "
                    "verify=%.1f (%.0f%%) rest=%.1f (%.0f%%)\n",
                    nTmg, tot / nTmg,
                    tDraft / nTmg,   100.0 * tDraft   / tot,
                    tReadout / nTmg, 100.0 * tReadout / tot,
                    tVerify / nTmg,  100.0 * tVerify  / tot,
                    tRest / nTmg,    100.0 * tRest    / tot);
        std::fflush(stdout);
    }

    if (draftedOut)  *draftedOut  = drafted;
    if (acceptedOut) *acceptedOut = accepted;
    return out;
}

} // namespace mimirmind::runtime::engine
