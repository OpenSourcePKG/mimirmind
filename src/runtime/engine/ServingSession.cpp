// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/engine/ServingSession.hpp"

#include "compute/Embedding.hpp"
#include "core/log/Log.hpp"
#include "runtime/BlockBuffers.hpp"
#include "runtime/SsmState.hpp"
#include "runtime/arch/Qwen35MoeBackend.hpp"
#include "runtime/serving/PagedKvPool.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <optional>
#include <stdexcept>

namespace mimirmind::runtime::engine {

// =======================================================================
// Persistent per-slot substrate that lets an external event loop
// (ContinuousBatcher) admit/decode/complete requests asynchronously. Each
// of the `maxBatch` physical slots owns a contiguous run of paged-KV blocks
// and one SsmState sequence-slice, pinned for a request's lifetime.
// stepServing() runs one batched forward over a contiguous slot prefix,
// each slot at its OWN position.
// =======================================================================
struct ServingState {
    std::size_t maxBatch{0};
    std::size_t maxContext{0};
    std::size_t blockSize{16};
    std::size_t blocksPerSeq{0};
    std::size_t numBlocks{0};

    // Cached model dims + weights (valid while the model stays loaded).
    std::size_t d_model{0};
    std::size_t vocab_lm{0};
    std::size_t vocab_emb{0};
    std::size_t blockCount{0};
    const core::gguf::GgufTensor* tokEmb{nullptr};
    const core::gguf::GgufTensor* outNorm{nullptr};
    const core::gguf::GgufTensor* lmHead{nullptr};
    arch::Qwen35MoeBackend*       qb{nullptr};

    // Persistent device state.
    std::unique_ptr<serving::PagedKvPool> pool;
    std::unique_ptr<SsmState>             ssm;
    std::optional<BlockBuffers>           sb;

    // Scratch (device).
    compute::ComputeBuffer expIdxBuf, kwBuf;
    compute::ComputeBuffer blockTablesDev, seqLensDev, startPosDev;
    compute::ComputeBuffer xBufB, normB, logitsB, lmScr;

    // Per-iteration host staging.
    std::vector<std::int32_t>  blockTablesH;  // [maxBatch * blocksPerSeq] static
    std::vector<std::uint32_t> writeBlockId;  // [maxBatch]
    std::vector<std::int32_t>  writeSlot;      // [maxBatch]
    std::vector<std::int32_t>  seqLensH;       // [maxBatch]
    std::vector<std::int32_t>  startPosH;      // [maxBatch]
    std::vector<std::uint8_t>  isSeqStart;     // [maxBatch]
    std::vector<std::int32_t>  inputTok;       // [maxBatch]
    std::vector<float>         hostLogits;     // [maxBatch * vocab_lm]
};

ServingSession::ServingSession(InferenceEngine& engine) : _e{engine} {}
ServingSession::~ServingSession() = default;

std::vector<std::vector<std::int32_t>>
ServingSession::generateBatch(
        const std::vector<std::vector<std::int32_t>>& prompts,
        std::size_t maxNew, std::int32_t eosId) {
    namespace cmp = mimirmind::compute;

    if (_e._backend == nullptr) {
        throw std::runtime_error("generateBatch: no model loaded");
    }
    auto* qb = dynamic_cast<arch::Qwen35MoeBackend*>(_e._backend.get());
    if (qb == nullptr) {
        throw std::runtime_error(
            "generateBatch: batched serving only supports qwen35moe");
    }
    const std::size_t nSeq = prompts.size();
    if (nSeq == 0 || maxNew == 0) {
        throw std::runtime_error("generateBatch: empty request");
    }
    std::vector<std::size_t> Tps(nSeq);
    std::size_t maxTp = 0;
    for (std::size_t s = 0; s < nSeq; ++s) {
        if (prompts[s].empty()) {
            throw std::runtime_error("generateBatch: empty prompt in batch");
        }
        Tps[s] = prompts[s].size();
        maxTp  = std::max(maxTp, Tps[s]);
    }

    const auto* tokEmb  = _e._weights->find("token_embd.weight");
    const auto* outNorm = _e._weights->find("output_norm.weight");
    const auto* lmHead  = _e._weights->find("output.weight");
    if (lmHead == nullptr) {
        lmHead = tokEmb;
    }
    if (tokEmb == nullptr || outNorm == nullptr || lmHead == nullptr) {
        throw std::runtime_error("generateBatch: embed/norm/lm_head missing");
    }

    const std::size_t d_model   = _e._config.embeddingLength;
    const std::size_t vocab_lm  = lmHead->dimensions.size() >= 2
                                      ? lmHead->dimensions[1] : _e._tokenizer.vocabSize();
    const std::size_t vocab_emb = tokEmb->dimensions.size() >= 2
                                      ? tokEmb->dimensions[1] : _e._tokenizer.vocabSize();
    const std::size_t total     = maxTp + maxNew;

    const std::size_t nKvHeads   = _e._config.headCountKv;
    const std::size_t headDim    = _e._config.headDim();
    const std::size_t K          = _e._config.expertUsedCount;
    const std::size_t blockCount = _e._config.blockCount;
    std::size_t nFullAttn = 0;
    for (std::size_t b = 0; b < blockCount; ++b) {
        if (!_e._config.isRecurrentLayer(b)) ++nFullAttn;
    }

    // Paged pool with contiguous blocks per sequence (validation harness).
    const std::size_t blockSize    = 16;
    const std::size_t blocksPerSeq = (total + blockSize - 1) / blockSize;
    const std::size_t numBlocks    = nSeq * blocksPerSeq;
    serving::PagedKvPool pool(*_e._ops, nFullAttn, numBlocks, blockSize,
                              nKvHeads, headDim);

    SsmState ssm(*_e._ops, blockCount, _e._config.ssmStateElemsPerLayer(),
                 _e._config.ssmConvStateElemsPerLayer(), nSeq);

    const auto qkv = qb->maxQKVDims();
    BlockBuffers sb = allocBlockBuffers(*_e._ops, _e._config, /*maxT=*/nSeq,
                                        /*maxSeq=*/nSeq, qkv.first, qkv.second,
                                        /*withFusedQkv=*/false,
                                        /*withKvFp32Scratch=*/true,
                                        /*withQGate=*/true, /*withSsm=*/true,
                                        /*perSeqConvInput=*/true);
    sb.ssmStatePtr     = ssm.statePtr();
    sb.ssmConvStatePtr = ssm.convStatePtr();
    sb.ssmSlabNSeq     = ssm.nSeq();

    auto expIdxBuf = _e._ops->allocate(nSeq * K * sizeof(std::int32_t));
    auto kwBuf     = _e._ops->allocate(nSeq * K * sizeof(float));

    std::vector<std::int32_t> blockTablesH(nSeq * blocksPerSeq);
    for (std::size_t s = 0; s < nSeq; ++s) {
        for (std::size_t i = 0; i < blocksPerSeq; ++i) {
            blockTablesH[s * blocksPerSeq + i] =
                static_cast<std::int32_t>(s * blocksPerSeq + i);
        }
    }
    auto blockTablesDev = _e._ops->allocate(blockTablesH.size() * sizeof(std::int32_t));
    _e._ops->uploadHostBytes(blockTablesDev.get(), blockTablesH.data(),
                             blockTablesH.size() * sizeof(std::int32_t));
    auto seqLensDev  = _e._ops->allocate(nSeq * sizeof(std::int32_t));
    auto startPosDev = _e._ops->allocate(nSeq * sizeof(std::int32_t));

    auto xBufB   = _e._ops->allocate(nSeq * d_model  * sizeof(float));
    auto normB   = _e._ops->allocate(nSeq * d_model  * sizeof(float));
    auto logitsB = _e._ops->allocate(nSeq * vocab_lm * sizeof(float));
    auto lmScr   = _e._ops->allocate(std::max(d_model, vocab_lm) * sizeof(float));
    float* const xBuf   = xBufB.as<float>();
    float* const normBuf= normB.as<float>();
    float* const logits = logitsB.as<float>();

    std::vector<std::uint32_t> writeBlockId(nSeq);
    std::vector<std::int32_t>  writeSlot(nSeq);
    std::vector<std::int32_t>  seqLensH(nSeq);
    std::vector<std::int32_t>  startPosH(nSeq);
    std::vector<std::uint8_t>  isSeqStart(nSeq);
    std::vector<std::int32_t>  inputTok(nSeq);
    std::vector<std::vector<std::int32_t>> out(nSeq);

    const bool timing = std::getenv("MIMIRMIND_BATCH_TIMING") != nullptr;
    double tFull = 0.0, tLin = 0.0, tLm = 0.0, tPre = 0.0;
    using clk = std::chrono::steady_clock;

    auto stepForward = [&](std::size_t p) {
        const auto tp0 = clk::now();
        cmp::embeddingLookup(tokEmb->type, tokEmb->usmPtr, d_model, vocab_emb,
                             std::span<const std::int32_t>{inputTok.data(), nSeq},
                             xBuf);
        for (std::size_t s = 0; s < nSeq; ++s) {
            writeBlockId[s] = static_cast<std::uint32_t>(s * blocksPerSeq + p / blockSize);
            writeSlot[s]    = static_cast<std::int32_t>(p % blockSize);
            seqLensH[s]     = static_cast<std::int32_t>(p + 1);
            startPosH[s]    = static_cast<std::int32_t>(p);
            isSeqStart[s]   = (p == 0) ? 1U : 0U;
        }
        _e._ops->uploadHostBytes(seqLensDev.get(),  seqLensH.data(),  nSeq * sizeof(std::int32_t));
        _e._ops->uploadHostBytes(startPosDev.get(), startPosH.data(), nSeq * sizeof(std::int32_t));

        arch::BatchedDecodeCtx ctx{};
        ctx.nSeq            = nSeq;
        ctx.pool            = &pool;
        ctx.writeBlockId    = writeBlockId.data();
        ctx.writeSlot       = writeSlot.data();
        ctx.blockTablesDev  = static_cast<const std::int32_t*>(blockTablesDev.get());
        ctx.seqLensDev      = static_cast<const std::int32_t*>(seqLensDev.get());
        ctx.maxBlocksPerSeq = blocksPerSeq;
        ctx.startPosDev     = static_cast<const std::int32_t*>(startPosDev.get());
        ctx.expIdxSlot      = expIdxBuf.as<std::int32_t>();
        ctx.kwSlot          = kwBuf.as<float>();
        ctx.isSeqStart      = isSeqStart.data();

        if (timing) {
            _e._ops->flush();
            tPre += std::chrono::duration<double, std::milli>(clk::now() - tp0).count();
        }
        for (std::size_t b = 0; b < blockCount; ++b) {
            const auto tb0 = timing ? clk::now() : clk::time_point{};
            qb->runBlockBatched(b, xBuf, ctx, sb);
            if (timing) {
                _e._ops->flush();
                const double dt =
                    std::chrono::duration<double, std::milli>(clk::now() - tb0).count();
                if (_e._config.isRecurrentLayer(b)) tLin += dt; else tFull += dt;
            }
        }
    };

    auto lmHeadSample = [&]() {
        _e._ops->rmsNormAsync(xBuf, nSeq, d_model,
                              static_cast<const float*>(outNorm->usmPtr),
                              _e._config.rmsNormEps, normBuf);
        _e._gmm->matmul(lmHead->type, lmHead->usmPtr, vocab_lm, d_model,
                        normBuf, nSeq, logits, lmScr.as<float>());
        _e._ops->flush();
        std::vector<float> host(nSeq * vocab_lm);
        _e._ops->readbackToHost(host.data(), logits, nSeq * vocab_lm * sizeof(float));
        std::vector<std::int32_t> toks(nSeq);
        for (std::size_t s = 0; s < nSeq; ++s) {
            const float* row = host.data() + s * vocab_lm;
            std::size_t best = 0;
            float bv = row[0];
            for (std::size_t v = 1; v < vocab_lm; ++v) {
                if (row[v] > bv) { bv = row[v]; best = v; }
            }
            toks[s] = static_cast<std::int32_t>(best);
        }
        return toks;
    };

    // Lockstep decode: at global step g every sequence is at position g,
    // feeding its prompt token while g < its prompt length, then its own
    // last sampled token.
    std::vector<std::int32_t> lastTok(nSeq, 0);
    std::vector<char>         finished(nSeq, 0);
    const std::size_t totalSteps = maxTp + maxNew;
    for (std::size_t g = 0; g < totalSteps; ++g) {
        for (std::size_t s = 0; s < nSeq; ++s) {
            inputTok[s] = (g < Tps[s]) ? prompts[s][g] : lastTok[s];
        }
        stepForward(g);
        const auto tl0 = timing ? clk::now() : clk::time_point{};
        const std::vector<std::int32_t> toks = lmHeadSample();
        if (timing) {
            tLm += std::chrono::duration<double, std::milli>(clk::now() - tl0).count();
        }
        bool allDone = true;
        for (std::size_t s = 0; s < nSeq; ++s) {
            if (finished[s] != 0) {
                continue;
            }
            if (g + 1 >= Tps[s]) {              // toks[s] is a generated token
                out[s].push_back(toks[s]);
                lastTok[s] = toks[s];
                if (toks[s] == eosId || out[s].size() >= maxNew) {
                    finished[s] = 1;
                }
            }
            if (finished[s] == 0) {
                allDone = false;
            }
        }
        if (allDone) {
            break;
        }
    }
    if (timing) {
        std::fprintf(stderr,
            "[batch-timing] nSeq=%zu full-attn=%.1f ms  linear=%.1f ms  "
            "lm-head=%.1f ms  prep=%.1f ms  (total blocks-only=%.1f ms)\n",
            nSeq, tFull, tLin, tLm, tPre, tFull + tLin);
        std::fflush(stderr);
    }
    return out;
}

std::vector<std::vector<std::int32_t>>
ServingSession::generateServingParity(std::span<const std::int32_t> promptIds,
                                      std::size_t nSeq, std::size_t maxNew) {
    if (promptIds.empty()) {
        throw std::runtime_error("generateServingParity: empty prompt");
    }
    // nSeq copies of the SAME prompt, no EOS stop (eosId=-1 never matches a
    // valid token id) => exactly maxNew tokens per sequence, so the streams
    // are directly comparable to single-session greedy generate().
    const std::vector<std::int32_t>        p(promptIds.begin(), promptIds.end());
    std::vector<std::vector<std::int32_t>> prompts(nSeq, p);
    return generateBatch(prompts, maxNew, /*eosId=*/-1);
}

void ServingSession::ensureServingState(std::size_t maxBatch,
                                        std::size_t maxContext) {
    if (_e._backend == nullptr) {
        throw std::runtime_error("ensureServingState: no model loaded");
    }
    auto* qb = dynamic_cast<arch::Qwen35MoeBackend*>(_e._backend.get());
    if (qb == nullptr) {
        throw std::runtime_error(
            "ensureServingState: continuous batching only supports qwen35moe");
    }
    if (maxBatch == 0 || maxContext == 0) {
        throw std::runtime_error("ensureServingState: maxBatch/maxContext must be > 0");
    }
    // Idempotent unless capacity must grow.
    if (_state != nullptr && _state->maxBatch >= maxBatch &&
        _state->maxContext >= maxContext) {
        return;
    }

    const auto* tokEmb  = _e._weights->find("token_embd.weight");
    const auto* outNorm = _e._weights->find("output_norm.weight");
    const auto* lmHead  = _e._weights->find("output.weight");
    if (lmHead == nullptr) {
        lmHead = tokEmb;
    }
    if (tokEmb == nullptr || outNorm == nullptr || lmHead == nullptr) {
        throw std::runtime_error("ensureServingState: embed/norm/lm_head missing");
    }

    auto st = std::make_unique<ServingState>();
    st->maxBatch   = maxBatch;
    st->maxContext = maxContext;
    st->blockSize  = 16;
    st->blocksPerSeq =
        (maxContext + st->blockSize - 1) / st->blockSize;
    st->numBlocks  = maxBatch * st->blocksPerSeq;

    st->d_model   = _e._config.embeddingLength;
    st->vocab_lm  = lmHead->dimensions.size() >= 2 ? lmHead->dimensions[1]
                                                   : _e._tokenizer.vocabSize();
    st->vocab_emb = tokEmb->dimensions.size() >= 2 ? tokEmb->dimensions[1]
                                                   : _e._tokenizer.vocabSize();
    st->blockCount = _e._config.blockCount;
    st->tokEmb  = tokEmb;
    st->outNorm = outNorm;
    st->lmHead  = lmHead;
    st->qb      = qb;

    const std::size_t nKvHeads = _e._config.headCountKv;
    const std::size_t headDim  = _e._config.headDim();
    const std::size_t K        = _e._config.expertUsedCount;
    std::size_t nFullAttn = 0;
    for (std::size_t b = 0; b < st->blockCount; ++b) {
        if (!_e._config.isRecurrentLayer(b)) ++nFullAttn;
    }

    st->pool = std::make_unique<serving::PagedKvPool>(
        *_e._ops, nFullAttn, st->numBlocks, st->blockSize, nKvHeads, headDim);
    st->ssm = std::make_unique<SsmState>(
        *_e._ops, st->blockCount, _e._config.ssmStateElemsPerLayer(),
        _e._config.ssmConvStateElemsPerLayer(), maxBatch);

    const auto qkv = qb->maxQKVDims();
    st->sb = allocBlockBuffers(*_e._ops, _e._config, /*maxT=*/maxBatch,
                               /*maxSeq=*/maxBatch, qkv.first, qkv.second,
                               /*withFusedQkv=*/false, /*withKvFp32Scratch=*/true,
                               /*withQGate=*/true, /*withSsm=*/true,
                               /*perSeqConvInput=*/true);
    st->sb->ssmStatePtr     = st->ssm->statePtr();
    st->sb->ssmConvStatePtr = st->ssm->convStatePtr();
    st->sb->ssmSlabNSeq     = st->ssm->nSeq();

    st->expIdxBuf = _e._ops->allocate(maxBatch * K * sizeof(std::int32_t));
    st->kwBuf     = _e._ops->allocate(maxBatch * K * sizeof(float));

    // Static per-slot block table: slot s owns physical blocks
    // [s*blocksPerSeq, (s+1)*blocksPerSeq). Uploaded once.
    st->blockTablesH.resize(maxBatch * st->blocksPerSeq);
    for (std::size_t s = 0; s < maxBatch; ++s) {
        for (std::size_t i = 0; i < st->blocksPerSeq; ++i) {
            st->blockTablesH[s * st->blocksPerSeq + i] =
                static_cast<std::int32_t>(s * st->blocksPerSeq + i);
        }
    }
    st->blockTablesDev =
        _e._ops->allocate(st->blockTablesH.size() * sizeof(std::int32_t));
    _e._ops->uploadHostBytes(st->blockTablesDev.get(), st->blockTablesH.data(),
                             st->blockTablesH.size() * sizeof(std::int32_t));
    st->seqLensDev  = _e._ops->allocate(maxBatch * sizeof(std::int32_t));
    st->startPosDev = _e._ops->allocate(maxBatch * sizeof(std::int32_t));

    st->xBufB   = _e._ops->allocate(maxBatch * st->d_model  * sizeof(float));
    st->normB   = _e._ops->allocate(maxBatch * st->d_model  * sizeof(float));
    st->logitsB = _e._ops->allocate(maxBatch * st->vocab_lm * sizeof(float));
    st->lmScr   = _e._ops->allocate(std::max(st->d_model, st->vocab_lm) * sizeof(float));

    st->writeBlockId.resize(maxBatch);
    st->writeSlot.resize(maxBatch);
    st->seqLensH.resize(maxBatch);
    st->startPosH.resize(maxBatch);
    st->isSeqStart.resize(maxBatch);
    st->inputTok.resize(maxBatch);
    st->hostLogits.resize(maxBatch * st->vocab_lm);

    _state = std::move(st);
    MM_LOG_INFO("serving",
                "ensureServingState: maxBatch={} maxContext={} blocksPerSeq={} "
                "numBlocks={} vocab_lm={}",
                maxBatch, maxContext, _state->blocksPerSeq, _state->numBlocks,
                _state->vocab_lm);
}

void ServingSession::stepServing(
        std::span<const InferenceEngine::ServingSlotStep> steps,
        std::span<std::int32_t>                           outTokens) {
    namespace cmp = mimirmind::compute;
    if (_state == nullptr) {
        throw std::runtime_error("stepServing: ensureServingState not called");
    }
    auto& st = *_state;
    const std::size_t nSeq = steps.size();
    if (nSeq == 0) {
        return;
    }
    if (nSeq > st.maxBatch) {
        throw std::runtime_error("stepServing: nSeq exceeds serving maxBatch");
    }
    if (outTokens.size() != nSeq) {
        throw std::runtime_error("stepServing: outTokens size != steps size");
    }

    for (std::size_t i = 0; i < nSeq; ++i) {
        const InferenceEngine::ServingSlotStep& s = steps[i];
        if (s.slot != i) {
            throw std::runtime_error(
                "stepServing: steps must be ordered by slot over a contiguous "
                "prefix (steps[i].slot == i)");
        }
        const std::size_t pos = static_cast<std::size_t>(s.pos);
        if (pos >= st.maxContext) {
            throw std::runtime_error("stepServing: position exceeds maxContext");
        }
        st.inputTok[i]     = s.token;
        st.writeBlockId[i] = static_cast<std::uint32_t>(
            i * st.blocksPerSeq + pos / st.blockSize);
        st.writeSlot[i]    = static_cast<std::int32_t>(pos % st.blockSize);
        st.seqLensH[i]     = static_cast<std::int32_t>(pos + 1);
        st.startPosH[i]    = static_cast<std::int32_t>(pos);
        st.isSeqStart[i]   = s.seqStart ? 1U : 0U;
    }
    _e._ops->uploadHostBytes(st.seqLensDev.get(),  st.seqLensH.data(),
                             nSeq * sizeof(std::int32_t));
    _e._ops->uploadHostBytes(st.startPosDev.get(), st.startPosH.data(),
                             nSeq * sizeof(std::int32_t));

    float* const xBuf   = st.xBufB.as<float>();
    float* const normBuf= st.normB.as<float>();
    float* const logits = st.logitsB.as<float>();

    cmp::embeddingLookup(st.tokEmb->type, st.tokEmb->usmPtr, st.d_model,
                         st.vocab_emb,
                         std::span<const std::int32_t>{st.inputTok.data(), nSeq},
                         xBuf);

    arch::BatchedDecodeCtx ctx{};
    ctx.nSeq            = nSeq;
    ctx.pool            = st.pool.get();
    ctx.writeBlockId    = st.writeBlockId.data();
    ctx.writeSlot       = st.writeSlot.data();
    ctx.blockTablesDev  = static_cast<const std::int32_t*>(st.blockTablesDev.get());
    ctx.seqLensDev      = static_cast<const std::int32_t*>(st.seqLensDev.get());
    ctx.maxBlocksPerSeq = st.blocksPerSeq;
    ctx.startPosDev     = static_cast<const std::int32_t*>(st.startPosDev.get());
    ctx.expIdxSlot      = st.expIdxBuf.as<std::int32_t>();
    ctx.kwSlot          = st.kwBuf.as<float>();
    ctx.isSeqStart      = st.isSeqStart.data();

    for (std::size_t b = 0; b < st.blockCount; ++b) {
        st.qb->runBlockBatched(b, xBuf, ctx, *st.sb);
    }

    _e._ops->rmsNormAsync(xBuf, nSeq, st.d_model,
                          static_cast<const float*>(st.outNorm->usmPtr),
                          _e._config.rmsNormEps, normBuf);
    _e._gmm->matmul(st.lmHead->type, st.lmHead->usmPtr, st.vocab_lm, st.d_model,
                    normBuf, nSeq, logits, st.lmScr.as<float>());
    _e._ops->flush();
    _e._ops->readbackToHost(st.hostLogits.data(), logits,
                            nSeq * st.vocab_lm * sizeof(float));
    for (std::size_t i = 0; i < nSeq; ++i) {
        const float* row = st.hostLogits.data() + i * st.vocab_lm;
        std::size_t best = 0;
        float bv = row[0];
        for (std::size_t v = 1; v < st.vocab_lm; ++v) {
            if (row[v] > bv) { bv = row[v]; best = v; }
        }
        outTokens[i] = static_cast<std::int32_t>(best);
    }
}

std::size_t ServingSession::maxBatch() const noexcept {
    return _state != nullptr ? _state->maxBatch : 0;
}

std::size_t ServingSession::maxContext() const noexcept {
    return _state != nullptr ? _state->maxContext : 0;
}

} // namespace mimirmind::runtime::engine
