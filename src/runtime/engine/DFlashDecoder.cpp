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
    _logits    = _e._ops->allocate(_vocabLm * sizeof(float));

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

    auto argmaxDev = [&](float* devLogits) -> std::int32_t {
        if (_e._logitsHostScratch.size() < _vocabLm) {
            _e._logitsHostScratch.resize(_vocabLm);
        }
        _e._ops->flush();
        _e._ops->readbackToHost(_e._logitsHostScratch.data(), devLogits,
                                _vocabLm * sizeof(float));
        std::size_t best = 0;
        float       bv   = _e._logitsHostScratch[0];
        for (std::size_t v = 1; v < _vocabLm; ++v) {
            if (_e._logitsHostScratch[v] > bv) { bv = _e._logitsHostScratch[v]; best = v; }
        }
        return static_cast<std::int32_t>(best);
    };

    // ---- Prefill: trunk forward with the hidden tap live ------------------
    _e.resetCache();
    qb->configureHiddenTap(
        std::span<const std::size_t>{kTapLayers, _taps},
        std::span<float* const>{_tapPtr.data(), _tapPtr.size()});

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

    while (out.size() < maxNew && !stop) {
        const std::size_t K = std::min(draftN, maxNew - out.size());

        // ---- Draft the whole K-token block in ONE forward ----------------
        // block = [token0, mask x K]; embed via borrowed target embed_tokens.
        std::vector<std::int32_t> block;
        block.reserve(K + 1);
        block.push_back(token0);
        for (std::size_t i = 0; i < K; ++i) block.push_back(_maskTok);

        compute::embeddingLookup(tokEmb->type, tokEmb->usmPtr, _d, _vocabEmb,
                                 std::span<const std::int32_t>{block},
                                 _noise.as<float>());

        _runner->draftForward(_noise.as<float>(), _ctxHidden.as<float>(),
                              K + 1, ctxLen, _draftOut.as<float>());

        // Readout: borrowed target lm_head on draft positions 1..K. The runner
        // already applied the drafter's final norm, so NO target output_norm.
        std::vector<std::int32_t> drafts;
        drafts.reserve(K);
        float* const draftOut = _draftOut.as<float>();
        for (std::size_t i = 1; i <= K; ++i) {
            _e._gmm->matmul(lmHead->type, lmHead->usmPtr, _vocabLm, _d,
                            draftOut + i * _d, 1, _logits.as<float>(), logitsSc);
            drafts.push_back(argmaxDev(_logits.as<float>()));
            ++drafted;
        }

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

        // ---- Accept longest greedy prefix -------------------------------
        std::size_t a = 0;
        for (std::size_t i = 0; i < K; ++i) {
            if (argmaxHost(vl[i]) == drafts[i]) ++a; else break;
        }
        const std::int32_t corrected = argmaxHost(vl[a]);
        accepted += a;

        std::vector<std::int32_t> committed(
            vtoks.begin(), vtoks.begin() + static_cast<std::ptrdiff_t>(a + 1));

        if (a == K) {
            _e.commitVerified(committed);            // tap rows already committed
        } else {
            // Partial accept: restore SSM, undo provisional KV, re-forward the
            // accepted prefix so KV + SSM + tap sinks land on the committed suffix.
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

        // ---- Emit token0 + accepted drafts; corrected -> next token0 -----
        bool cont = emit(token0);
        for (std::size_t i = 0; i < a && cont; ++i) cont = emit(drafts[i]);
        token0 = corrected;
        if (!cont) break;
    }

    qb->clearHiddenTap();

    if (draftedOut)  *draftedOut  = drafted;
    if (acceptedOut) *acceptedOut = accepted;
    return out;
}

} // namespace mimirmind::runtime::engine
