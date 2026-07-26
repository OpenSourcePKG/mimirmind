// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/engine/MtpDecoder.hpp"

#include "runtime/InferenceEngine.hpp"
#include "runtime/BlockBuffers.hpp"
#include "runtime/KvCache.hpp"
#include "runtime/SsmState.hpp"
#include "runtime/arch/Qwen35MoeBackend.hpp"

#include <algorithm>
#include <stdexcept>

namespace mimirmind::runtime::engine {

std::vector<std::int32_t>
MtpDecoder::generate(std::span<const std::int32_t> promptIds,
                     std::size_t maxNew, std::size_t mtpDepth,
                     std::int32_t eosId,
                     std::size_t* draftedOut, std::size_t* acceptedOut) {
    auto* qb = dynamic_cast<arch::Qwen35MoeBackend*>(_e._backend.get());
    if (qb == nullptr || !_e.mtpAvailable()) {
        throw std::runtime_error("generateMtp: requires CUDA qwen35moe + MTP head");
    }
    if (promptIds.empty() || maxNew == 0) {
        return {};
    }
    if (mtpDepth == 0) mtpDepth = 1;

    const std::size_t d      = _e._config.embeddingLength;
    const auto*       tokEmb = _e._weights->find("token_embd.weight");
    const auto*       lmHead = _e._weights->find("output.weight");
    if (lmHead == nullptr) lmHead = tokEmb;
    const std::size_t vocab  = (lmHead != nullptr && lmHead->dimensions.size() >= 2)
                                   ? lmHead->dimensions[1] : _e._tokenizer.vocabSize();

    // Clean trunk state, then size trunk scratch (forwardVerify's ensureCapacity
    // handles _kvCache / _blockBuffers alloc on the first call below).
    _e.resetCache();

    // Lazy MTP KV + per-step scratch (own 1-layer cache for blk.<blockCount>).
    if (_kv == nullptr) {
        _kv = std::make_unique<KvCache>(
            *_e._ops, /*nLayers=*/1, _e._maxContextTokens,
            _e._config.headCountKv, _e._config.headDim(), _e._kvDtype);
        _emb    = _e._ops->allocate(d * sizeof(float));
        _cat    = _e._ops->allocate(2 * d * sizeof(float));
        _eh     = _e._ops->allocate(d * sizeof(float));
        _hidden = _e._ops->allocate(d * sizeof(float));
        _logits = _e._ops->allocate(vocab * sizeof(float));
    }
    _kv->reset();

    float* const embS   = _emb.as<float>();
    float* const catS   = _cat.as<float>();
    float* const ehS    = _eh.as<float>();
    float* const hidS   = _hidden.as<float>();
    float* const mtpLog = _logits.as<float>();
    float* const logSc  = _e._logitsScH.as<float>();

    auto argmaxDev = [&](float* devLogits) -> std::int32_t {
        if (_e._logitsHostScratch.size() < vocab) _e._logitsHostScratch.resize(vocab);
        _e._ops->flush();
        _e._ops->readbackToHost(_e._logitsHostScratch.data(), devLogits,
                                vocab * sizeof(float));
        std::size_t best = 0; float bv = _e._logitsHostScratch[0];
        for (std::size_t v = 1; v < vocab; ++v)
            if (_e._logitsHostScratch[v] > bv) { bv = _e._logitsHostScratch[v]; best = v; }
        return static_cast<std::int32_t>(best);
    };
    auto argmaxHost = [&](const std::vector<float>& row) -> std::int32_t {
        std::size_t best = 0; float bv = row[0];
        for (std::size_t v = 1; v < row.size(); ++v)
            if (row[v] > bv) { bv = row[v]; best = v; }
        return static_cast<std::int32_t>(best);
    };

    // ---- Prefill: trunk forward (provisional KV) + seed the MTP KV -------
    std::vector<std::int32_t> prompt(promptIds.begin(), promptIds.end());
    auto pfLogits = _e.forwardVerify(prompt);  // _xBufH now holds the P trunk hiddens
    const std::size_t P    = prompt.size();
    BlockBuffers&     buf  = *_e._blockBuffers;
    float*            xBuf = _e._xBufH.as<float>();
    for (std::size_t p = 0; p + 1 < P; ++p) {
        qb->runMtpDraftStep(xBuf + p * d, prompt[p + 1], *_kv, buf,
                            embS, catS, ehS, mtpLog, logSc);
        _kv->commit(1);
    }
    _e.commitVerified(prompt);
    _e._ops->appendMemoryCopy(hidS, xBuf + (P - 1) * d, d * sizeof(float));
    _e._ops->flush();
    std::int32_t token0 = argmaxHost(pfLogits.back());

    std::vector<std::int32_t> out;
    std::size_t drafted = 0, accepted = 0;
    bool stop = false;
    auto emit = [&](std::int32_t t) -> bool {
        out.push_back(t);
        if (eosId >= 0 && t == eosId) { stop = true; return false; }
        return out.size() < maxNew;
    };

    while (out.size() < maxNew && !stop) {
        const std::size_t K = std::min(mtpDepth, maxNew - out.size());

        // ---- Draft K tokens with the nextn module -----------------------
        std::vector<std::int32_t> drafts;
        drafts.reserve(K);
        std::int32_t       prev = token0;
        const float*       hcur = hidS;
        const std::size_t  mtpPre = _kv->length();
        for (std::size_t k = 0; k < K; ++k) {
            qb->runMtpDraftStep(hcur, prev, *_kv, buf,
                                embS, catS, ehS, mtpLog, logSc);
            _kv->commit(1);
            const std::int32_t dk = argmaxDev(mtpLog);
            drafts.push_back(dk);
            hcur = ehS;   // block-<mtp> output = next-step hidden
            prev = dk;
            ++drafted;
        }

        // ---- Verify: one trunk forward on [token0, drafts...] -----------
        std::vector<std::int32_t> vtoks;
        vtoks.reserve(K + 1);
        vtoks.push_back(token0);
        vtoks.insert(vtoks.end(), drafts.begin(), drafts.end());

        // Snapshot the trunk GatedDeltaNet recurrent state + rolling conv tail
        // before verify — it advances monolithically over all K+1 tokens and
        // (unlike the KV cache) cannot be truncated. On a partial accept we
        // restore it and re-forward just the accepted prefix.
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

        const std::size_t preKvLen = _e._kvCache->length();
        auto vlogits = _e.forwardVerify(vtoks);   // advances trunk KV(prov) + SSM by K+1
        xBuf = _e._xBufH.as<float>();

        // ---- Accept the longest matching prefix (greedy) ----------------
        std::size_t a = 0;
        for (std::size_t i = 0; i < K; ++i) {
            if (argmaxHost(vlogits[i]) == drafts[i]) ++a; else break;
        }
        const std::int32_t corrected = argmaxHost(vlogits[a]);
        accepted += a;

        std::vector<std::int32_t> committed(vtoks.begin(),
                                            vtoks.begin() + static_cast<std::ptrdiff_t>(a + 1));

        if (a == K) {
            // All K drafts accepted (committed == all K+1 verify tokens): the
            // SSM state already reflects exactly the committed prefix.
            _e.commitVerified(committed);
            _e._ops->appendMemoryCopy(hidS, xBuf + a * d, d * sizeof(float));
        } else {
            // Partial accept: the SSM state over-advanced. Restore it, undo the
            // provisional KV, and re-forward the accepted prefix so both the
            // KV and the recurrent state land exactly on the committed suffix.
            if (ssm != nullptr) {
                _e._ops->appendMemoryCopy(ssm->statePtr(),     _ssmBak.get(),  stBytes);
                _e._ops->appendMemoryCopy(ssm->convStatePtr(), _convBak.get(), cvBytes);
            }
            _e._kvCache->truncate(preKvLen);
            auto vl2 = _e.forwardVerify(committed);   // KV(prov) + SSM by a+1
            (void)vl2;
            xBuf = _e._xBufH.as<float>();
            _e.commitVerified(committed);
            _e._ops->appendMemoryCopy(hidS, xBuf + a * d, d * sizeof(float));
        }
        _kv->truncate(mtpPre + a + 1);  // MTP-attn KV: token0 + accepted

        // ---- Emit token0 + accepted drafts (corrected -> next token0) ---
        bool cont = emit(token0);
        for (std::size_t i = 0; i < a && cont; ++i) cont = emit(drafts[i]);

        _e._ops->flush();
        token0 = corrected;
        if (!cont) break;
    }

    if (draftedOut)  *draftedOut  = drafted;
    if (acceptedOut) *acceptedOut = accepted;
    return out;
}

} // namespace mimirmind::runtime::engine
