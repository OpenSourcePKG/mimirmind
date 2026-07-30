// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/engine/ServingSession.hpp"

#include "compute/Embedding.hpp"
#include "core/log/Log.hpp"
#include "runtime/BlockBuffers.hpp"
#include "runtime/KvCache.hpp"
#include "runtime/SsmState.hpp"
#include "runtime/arch/Qwen35MoeBackend.hpp"
#include "runtime/serving/PagedKvPool.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
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

    // ---- Increment E1: MTP batched-verify scratch (lazily sized) --------
    // Sized for up to `maxBatch` slots × (verifyDepth + 1) verify tokens =
    // Mcap virtual rows. Full-attention runs over all Mcap rows at once;
    // the GatedDeltaNet layers run (verifyDepth + 1) sequential steps of
    // nSeq = maxBatch, snapshotting the recurrent state after each step.
    std::size_t verifyDepth{0};   // K (0 => verify scratch not yet built)
    std::size_t verifyMcap{0};    // maxBatch * (verifyDepth + 1)
    std::optional<BlockBuffers> vsb;                 // maxT = verifyMcap
    compute::ComputeBuffer vExpIdx, vKw;             // [Mcap * expertUsedCount]
    compute::ComputeBuffer vBlockTablesDev;          // [Mcap * blocksPerSeq]
    compute::ComputeBuffer vSeqLensDev, vStartPosDev;// [Mcap]
    compute::ComputeBuffer vXBuf, vNormB, vLogitsB, vLmScr;
    std::vector<compute::ComputeBuffer> ssmSnap;     // [K+1] full SsmState state images
    std::vector<compute::ComputeBuffer> convSnap;    // [K+1] full SsmState conv images
    std::vector<std::int32_t>  vBlockTablesH;        // [Mcap * blocksPerSeq]
    std::vector<std::int32_t>  vSeqLensH, vStartPosH;
    std::vector<std::uint32_t> vWriteBlockId;
    std::vector<std::int32_t>  vWriteSlot;
    std::vector<std::uint8_t>  vIsSeqStart;          // full-attn virt-slot start flags
    std::vector<std::uint8_t>  vGdnSeqStart;         // [(K+1) * maxBatch] per-step GDN start flags
    std::vector<std::int32_t>  vInputTok;
    std::vector<float>         vHostLogits;          // [Mcap * vocab_lm]

    // ---- Increment E2: per-slot nextn (MTP) KV + draft scratch ----------
    // One 1-layer nextn self-attention KV cache per physical slot (the MTP
    // draft chain is independent per slot). Draft scratch is shared — the
    // draft loop runs one slot at a time (blk.<blockCount> is cheap).
    bool mtpReady{false};
    std::vector<std::unique_ptr<KvCache>> mtpKv;     // [maxBatch] 1-layer nextn cache
    compute::ComputeBuffer mtpEmb;                   // [d_model]   embed(prevTok)
    compute::ComputeBuffer mtpCat;                   // [2*d_model] concat(norms)
    compute::ComputeBuffer mtpEh;                    // [d_model]   eh_proj / next hidden
    compute::ComputeBuffer mtpDraftLogits;           // [vocab_lm]  draft logits
    compute::ComputeBuffer mtpLmScr;                 // [max(d,vocab)] lm-head scratch
    compute::ComputeBuffer mtpHid;                   // [maxBatch, d] per-slot round hidden
    std::vector<float>     mtpHostLogits;            // [vocab_lm]  argmax readback

    // ---- Increment E5b: BATCHED nextn draft (perf) ----------------------
    // The nextn KV lives in `pool` at a dedicated extra layer (mtpPoolLayer =
    // nFullAttn), reusing the per-slot block tables; per-slot nextn length is
    // tracked in nextnLen. One batched runMtpDraftStepBatched over N slots
    // replaces the N sequential runMtpDraftStep + per-step flush.
    std::size_t mtpPoolLayer{std::numeric_limits<std::size_t>::max()};
    std::vector<std::size_t>   nextnLen;             // [maxBatch] per-slot nextn KV length
    compute::ComputeBuffer mtpEmbB, mtpCatB, mtpEhB; // [maxBatch,d]/[maxBatch,2d]/[maxBatch,d]
    compute::ComputeBuffer mtpTmpE, mtpTmpH;         // [maxBatch,d] rmsnorm temps
    compute::ComputeBuffer mtpDraftLogitsB, mtpLmScrB;
    compute::ComputeBuffer mtpSeqLensDev, mtpStartPosDev; // [maxBatch] nextn positions
    compute::ComputeBuffer mtpSeedHid;               // [maxBatch,d] broadcast seed hidden
    std::vector<std::int32_t>  mtpSeqLensH, mtpStartPosH;
    std::vector<std::uint32_t> mtpWriteBlockId;
    std::vector<std::int32_t>  mtpWriteSlot;
    std::vector<std::int32_t>  mtpPrevTok;           // [maxBatch] host prevTok
    std::vector<float>         mtpHostLogitsB;       // [maxBatch*vocab_lm] batched readback
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
    // Overhead probe: per decode step, measure host-enqueue (dispatch, no
    // flush) vs GPU-drain (a single flush after). dispatch/(dispatch+drain)
    // ~= the launch/host-sync fraction a CUDA graph would remove. Distinct
    // from `timing` (which flushes per block and thus serialises).
    const bool ohMode = std::getenv("MIMIRMIND_BATCH_OVERHEAD") != nullptr;
    double tFull = 0.0, tLin = 0.0, tLm = 0.0, tPre = 0.0;
    double tDisp = 0.0, tDrain = 0.0;
    std::size_t ohSteps = 0;
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
        if (ohMode) {
            // Pure host enqueue of all blocks (no per-block flush), then one
            // drain. Skip the first steps (first-touch PTX JIT / warm-up).
            const auto td0 = clk::now();
            for (std::size_t b = 0; b < blockCount; ++b) {
                qb->runBlockBatched(b, xBuf, ctx, sb);
            }
            const double tdisp =
                std::chrono::duration<double, std::milli>(clk::now() - td0).count();
            const auto tf0 = clk::now();
            _e._ops->flush();
            const double tdrain =
                std::chrono::duration<double, std::milli>(clk::now() - tf0).count();
            if (p >= 4) {
                tDisp  += tdisp;
                tDrain += tdrain;
                ++ohSteps;
            }
        } else {
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
    if (ohMode && ohSteps > 0) {
        const double disp  = tDisp / static_cast<double>(ohSteps);
        const double drain = tDrain / static_cast<double>(ohSteps);
        const double frac  = 100.0 * tDisp / (tDisp + tDrain);
        std::fprintf(stderr,
            "[batch-overhead] nSeq=%zu steps=%zu blocks=%zu  "
            "dispatch(host-enqueue)=%.3f ms/step  drain(gpu)=%.3f ms/step  "
            "=> dispatch-bound=%.1f%% (CUDA-graph-removable)\n",
            nSeq, ohSteps, blockCount, disp, drain, frac);
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

    // Increment E5b: one extra paged layer holds the nextn (MTP) KV so the
    // batched draft can attend it through the same per-slot block tables.
    const bool        hasMtp      = _e.mtpAvailable();
    const std::size_t nPoolLayers = nFullAttn + (hasMtp ? 1 : 0);
    st->mtpPoolLayer = hasMtp ? nFullAttn
                              : std::numeric_limits<std::size_t>::max();
    st->pool = std::make_unique<serving::PagedKvPool>(
        *_e._ops, nPoolLayers, st->numBlocks, st->blockSize, nKvHeads, headDim);
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

void ServingSession::ensureVerifyCapacity(std::size_t depth) {
    if (_state == nullptr) {
        throw std::runtime_error("stepServingVerify: ensureServingState not called");
    }
    auto& st = *_state;
    if (depth == 0) {
        depth = 1;   // K >= 1 (token0 + at least one draft)
    }
    if (st.verifyDepth >= depth && st.vsb.has_value()) {
        return;      // already sized for at least this depth
    }

    const std::size_t K       = std::max(st.verifyDepth, depth);
    const std::size_t Mcap    = st.maxBatch * (K + 1);
    const std::size_t d_model = st.d_model;
    const std::size_t vocab   = st.vocab_lm;
    const std::size_t nExp    = _e._config.expertUsedCount;

    st.verifyDepth = K;
    st.verifyMcap  = Mcap;

    const auto qkv = st.qb->maxQKVDims();
    st.vsb = allocBlockBuffers(*_e._ops, _e._config, /*maxT=*/Mcap,
                               /*maxSeq=*/Mcap, qkv.first, qkv.second,
                               /*withFusedQkv=*/false, /*withKvFp32Scratch=*/true,
                               /*withQGate=*/true, /*withSsm=*/true,
                               /*perSeqConvInput=*/true);
    // Bind the recurrent slab to the PERSISTENT per-slot SsmState (slab
    // width = maxBatch). Verify advances it in place; the snapshots let a
    // partial accept (Increment E3) restore it without a re-forward.
    st.vsb->ssmStatePtr     = st.ssm->statePtr();
    st.vsb->ssmConvStatePtr = st.ssm->convStatePtr();
    st.vsb->ssmSlabNSeq     = st.ssm->nSeq();

    st.vExpIdx = _e._ops->allocate(Mcap * nExp * sizeof(std::int32_t));
    st.vKw     = _e._ops->allocate(Mcap * nExp * sizeof(float));
    st.vBlockTablesDev =
        _e._ops->allocate(Mcap * st.blocksPerSeq * sizeof(std::int32_t));
    st.vSeqLensDev  = _e._ops->allocate(Mcap * sizeof(std::int32_t));
    st.vStartPosDev = _e._ops->allocate(Mcap * sizeof(std::int32_t));
    st.vXBuf    = _e._ops->allocate(Mcap * d_model * sizeof(float));
    st.vNormB   = _e._ops->allocate(Mcap * d_model * sizeof(float));
    st.vLogitsB = _e._ops->allocate(Mcap * vocab   * sizeof(float));
    st.vLmScr   = _e._ops->allocate(std::max(d_model, vocab) * sizeof(float));

    // Full recurrent-state images (one per verify step): [blockCount, slab,
    // stateElems] / conv. On a partial accept, slot s's slice is restored
    // from ssmSnap[a_s]; E1 only produces them (the logits are the gate).
    const std::size_t stImgElems = st.ssm->blockCount() * st.ssm->stateLayerStride();
    const std::size_t cvImgElems = st.ssm->blockCount() * st.ssm->convStateLayerStride();
    st.ssmSnap.clear();
    st.convSnap.clear();
    st.ssmSnap.reserve(K + 1);
    st.convSnap.reserve(K + 1);
    for (std::size_t j = 0; j <= K; ++j) {
        st.ssmSnap.push_back(_e._ops->allocate(stImgElems * sizeof(float)));
        st.convSnap.push_back(_e._ops->allocate(cvImgElems * sizeof(float)));
    }

    st.vBlockTablesH.assign(Mcap * st.blocksPerSeq, 0);
    st.vSeqLensH.assign(Mcap, 0);
    st.vStartPosH.assign(Mcap, 0);
    st.vWriteBlockId.assign(Mcap, 0);
    st.vWriteSlot.assign(Mcap, 0);
    st.vIsSeqStart.assign(Mcap, 0);
    st.vGdnSeqStart.assign((K + 1) * st.maxBatch, 0);
    st.vInputTok.assign(Mcap, 0);
    st.vHostLogits.assign(Mcap * vocab, 0.0F);

    MM_LOG_INFO("serving",
                "ensureVerifyCapacity: depth={} Mcap={} snapStateBytes={} x{}",
                K, Mcap, stImgElems * sizeof(float), K + 1);
}

std::vector<std::vector<float>>
ServingSession::stepServingVerify(
        std::span<const InferenceEngine::VerifySlot> slots,
        std::span<const std::int32_t>                tokensTimeMajor,
        std::size_t                                  depth) {
    namespace cmp = mimirmind::compute;
    if (_state == nullptr) {
        throw std::runtime_error("stepServingVerify: ensureServingState not called");
    }
    auto& st = *_state;
    const std::size_t N = slots.size();
    if (N == 0) {
        return {};
    }
    if (depth == 0) {
        depth = 1;
    }
    const std::size_t K   = depth;
    const std::size_t Kp1 = K + 1;
    const std::size_t M   = N * Kp1;
    if (N > st.maxBatch) {
        throw std::runtime_error("stepServingVerify: N exceeds serving maxBatch");
    }
    if (tokensTimeMajor.size() != M) {
        throw std::runtime_error(
            "stepServingVerify: tokensTimeMajor size != N*(depth+1)");
    }
    for (std::size_t s = 0; s < N; ++s) {
        if (slots[s].slot != s) {
            throw std::runtime_error(
                "stepServingVerify: slots must be a contiguous prefix "
                "(slots[s].slot == s)");
        }
        const std::size_t endPos =
            static_cast<std::size_t>(slots[s].basePos) + K;
        if (endPos >= st.maxContext) {
            throw std::runtime_error(
                "stepServingVerify: basePos+depth exceeds maxContext");
        }
    }
    ensureVerifyCapacity(K);

    const std::size_t d_model = st.d_model;
    const std::size_t vocab   = st.vocab_lm;
    const std::size_t bps     = st.blocksPerSeq;
    const std::size_t bsz     = st.blockSize;

    // --- host metadata for the N*(K+1) full-attention virtual slots ------
    // Row r = j*N + s is slot s's j-th verify token, at absolute position
    // basePos_s + j. It writes its K/V into slot s's paged blocks and, via
    // seqLens = pos+1, attends causally over [0, pos] — i.e. slot s's
    // committed prefix plus this round's earlier drafts, never the later ones.
    for (std::size_t j = 0; j < Kp1; ++j) {
        for (std::size_t s = 0; s < N; ++s) {
            const std::size_t r   = j * N + s;
            const std::size_t pos =
                static_cast<std::size_t>(slots[s].basePos) + j;
            st.vInputTok[r]     = tokensTimeMajor[r];
            st.vWriteBlockId[r] = static_cast<std::uint32_t>(s * bps + pos / bsz);
            st.vWriteSlot[r]    = static_cast<std::int32_t>(pos % bsz);
            st.vSeqLensH[r]     = static_cast<std::int32_t>(pos + 1);
            st.vStartPosH[r]    = static_cast<std::int32_t>(pos);
            st.vIsSeqStart[r]   = (pos == 0) ? 1U : 0U;
            for (std::size_t i = 0; i < bps; ++i) {
                st.vBlockTablesH[r * bps + i] =
                    static_cast<std::int32_t>(s * bps + i);
            }
        }
    }
    // per-step GatedDeltaNet sequence-start flags (nSeq = N per step): a
    // slot only starts fresh if its verify token j sits at absolute pos 0.
    for (std::size_t j = 0; j < Kp1; ++j) {
        for (std::size_t s = 0; s < N; ++s) {
            st.vGdnSeqStart[j * st.maxBatch + s] =
                (static_cast<std::size_t>(slots[s].basePos) + j == 0) ? 1U : 0U;
        }
    }

    _e._ops->uploadHostBytes(st.vBlockTablesDev.get(), st.vBlockTablesH.data(),
                             M * bps * sizeof(std::int32_t));
    _e._ops->uploadHostBytes(st.vSeqLensDev.get(),  st.vSeqLensH.data(),
                             M * sizeof(std::int32_t));
    _e._ops->uploadHostBytes(st.vStartPosDev.get(), st.vStartPosH.data(),
                             M * sizeof(std::int32_t));

    float* const xBuf = st.vXBuf.as<float>();
    cmp::embeddingLookup(st.tokEmb->type, st.tokEmb->usmPtr, d_model,
                         st.vocab_emb,
                         std::span<const std::int32_t>{st.vInputTok.data(), M},
                         xBuf);

    // --- full-attention context (all M virtual slots at once) ------------
    arch::BatchedDecodeCtx ctxFull{};
    ctxFull.nSeq            = M;
    ctxFull.pool            = st.pool.get();
    ctxFull.writeBlockId    = st.vWriteBlockId.data();
    ctxFull.writeSlot       = st.vWriteSlot.data();
    ctxFull.blockTablesDev  = static_cast<const std::int32_t*>(st.vBlockTablesDev.get());
    ctxFull.seqLensDev      = static_cast<const std::int32_t*>(st.vSeqLensDev.get());
    ctxFull.maxBlocksPerSeq = bps;
    ctxFull.startPosDev     = static_cast<const std::int32_t*>(st.vStartPosDev.get());
    ctxFull.expIdxSlot      = st.vExpIdx.as<std::int32_t>();
    ctxFull.kwSlot          = st.vKw.as<float>();
    ctxFull.isSeqStart      = st.vIsSeqStart.data();

    const std::size_t stStride = st.ssm->stateLayerStride();
    const std::size_t cvStride = st.ssm->convStateLayerStride();

    for (std::size_t b = 0; b < st.blockCount; ++b) {
        if (_e._config.isRecurrentLayer(b)) {
            // GatedDeltaNet: K+1 sequential batched steps (nSeq = N). Step j
            // advances the per-slot recurrent state by verify token j; a
            // full state image is snapshotted after each step so a partial
            // accept can restore it (Increment E3) without a re-forward.
            for (std::size_t j = 0; j < Kp1; ++j) {
                arch::BatchedDecodeCtx ctxGdn{};
                ctxGdn.nSeq       = N;
                ctxGdn.expIdxSlot = st.vExpIdx.as<std::int32_t>();
                ctxGdn.kwSlot     = st.vKw.as<float>();
                ctxGdn.isSeqStart = st.vGdnSeqStart.data() + j * st.maxBatch;
                st.qb->runBlockBatched(b, xBuf + j * N * d_model, ctxGdn, *st.vsb);
                _e._ops->appendMemoryCopy(
                    st.ssmSnap[j].as<float>() + b * stStride,
                    st.ssm->statePtr()        + b * stStride,
                    stStride * sizeof(float));
                _e._ops->appendMemoryCopy(
                    st.convSnap[j].as<float>() + b * cvStride,
                    st.ssm->convStatePtr()     + b * cvStride,
                    cvStride * sizeof(float));
            }
        } else {
            st.qb->runBlockBatched(b, xBuf, ctxFull, *st.vsb);
        }
    }

    // --- per-position logits over all M rows -----------------------------
    float* const normBuf = st.vNormB.as<float>();
    float* const logits  = st.vLogitsB.as<float>();
    _e._ops->rmsNormAsync(xBuf, M, d_model,
                          static_cast<const float*>(st.outNorm->usmPtr),
                          _e._config.rmsNormEps, normBuf);
    _e._gmm->matmul(st.lmHead->type, st.lmHead->usmPtr, vocab, d_model,
                    normBuf, M, logits, st.vLmScr.as<float>());
    _e._ops->flush();
    _e._ops->readbackToHost(st.vHostLogits.data(), logits,
                            M * vocab * sizeof(float));

    std::vector<std::vector<float>> out;
    out.reserve(M);
    for (std::size_t r = 0; r < M; ++r) {
        const float* row = st.vHostLogits.data() + r * vocab;
        out.emplace_back(row, row + vocab);
    }
    return out;
}

void ServingSession::ensureMtpServingState() {
    if (_state == nullptr) {
        throw std::runtime_error("mtp serving: ensureServingState not called");
    }
    auto& st = *_state;
    if (st.mtpReady) {
        return;
    }
    if (!_e.mtpAvailable()) {
        throw std::runtime_error("mtp serving: model has no nextn (MTP) head");
    }
    const std::size_t d        = st.d_model;
    const std::size_t vocab    = st.vocab_lm;
    const std::size_t nKvHeads = _e._config.headCountKv;
    const std::size_t headDim  = _e._config.headDim();

    st.mtpKv.clear();
    st.mtpKv.reserve(st.maxBatch);
    for (std::size_t s = 0; s < st.maxBatch; ++s) {
        st.mtpKv.push_back(std::make_unique<KvCache>(
            *_e._ops, /*nLayers=*/1, st.maxContext, nKvHeads, headDim, _e._kvDtype));
    }
    st.mtpEmb         = _e._ops->allocate(d * sizeof(float));
    st.mtpCat         = _e._ops->allocate(2 * d * sizeof(float));
    st.mtpEh          = _e._ops->allocate(d * sizeof(float));
    st.mtpDraftLogits = _e._ops->allocate(vocab * sizeof(float));
    st.mtpLmScr       = _e._ops->allocate(std::max(d, vocab) * sizeof(float));
    st.mtpHid         = _e._ops->allocate(st.maxBatch * d * sizeof(float));
    st.mtpHostLogits.assign(vocab, 0.0F);

    // Increment E5b — batched nextn draft scratch (nSeq rows at a time).
    const std::size_t B = st.maxBatch;
    st.nextnLen.assign(B, 0);
    st.mtpEmbB         = _e._ops->allocate(B * d * sizeof(float));
    st.mtpCatB         = _e._ops->allocate(B * 2 * d * sizeof(float));
    st.mtpEhB          = _e._ops->allocate(B * d * sizeof(float));
    st.mtpTmpE         = _e._ops->allocate(B * d * sizeof(float));
    st.mtpTmpH         = _e._ops->allocate(B * d * sizeof(float));
    st.mtpSeedHid      = _e._ops->allocate(B * d * sizeof(float));
    st.mtpDraftLogitsB = _e._ops->allocate(B * vocab * sizeof(float));
    st.mtpLmScrB       = _e._ops->allocate(std::max(d, vocab) * sizeof(float));
    st.mtpSeqLensDev   = _e._ops->allocate(B * sizeof(std::int32_t));
    st.mtpStartPosDev  = _e._ops->allocate(B * sizeof(std::int32_t));
    st.mtpSeqLensH.assign(B, 0);
    st.mtpStartPosH.assign(B, 0);
    st.mtpWriteBlockId.assign(B, 0);
    st.mtpWriteSlot.assign(B, 0);
    st.mtpPrevTok.assign(B, 0);
    st.mtpHostLogitsB.assign(B * vocab, 0.0F);

    st.mtpReady = true;

    MM_LOG_INFO("serving",
                "ensureMtpServingState: {} per-slot nextn KV caches (maxContext={})",
                st.maxBatch, st.maxContext);
}

void ServingSession::draftKInto(KvCache& kv, const float* hidden0,
                                std::int32_t prevTok, std::size_t K,
                                std::vector<std::int32_t>& out) {
    auto& st = *_state;
    float* const emb  = st.mtpEmb.as<float>();
    float* const cat  = st.mtpCat.as<float>();
    float* const eh   = st.mtpEh.as<float>();
    float* const dlog = st.mtpDraftLogits.as<float>();
    float* const lmSc = st.mtpLmScr.as<float>();
    const std::size_t vocab = st.vocab_lm;

    const float*  hcur = hidden0;
    std::int32_t  prev = prevTok;
    for (std::size_t k = 0; k < K; ++k) {
        // one nextn draft step: writes logits into dlog, next hidden into eh.
        st.qb->runMtpDraftStep(hcur, prev, kv, *st.sb, emb, cat, eh, dlog, lmSc);
        kv.commit(1);
        _e._ops->flush();
        _e._ops->readbackToHost(st.mtpHostLogits.data(), dlog, vocab * sizeof(float));
        std::size_t best = 0;
        float       bv   = st.mtpHostLogits[0];
        for (std::size_t v = 1; v < vocab; ++v) {
            if (st.mtpHostLogits[v] > bv) { bv = st.mtpHostLogits[v]; best = v; }
        }
        out.push_back(static_cast<std::int32_t>(best));
        hcur = eh;                                   // block-<mtp> out = next hidden
        prev = static_cast<std::int32_t>(best);
    }
}

void ServingSession::mtpSeedBatched(std::size_t nSeq, const float* promptHiddens,
                                    std::span<const std::int32_t> prompt) {
    auto& st = *_state;
    const std::size_t d   = st.d_model;
    const std::size_t bps = st.blocksPerSeq;
    const std::size_t bsz = st.blockSize;
    const std::size_t P   = prompt.size();
    float* const seedHid = st.mtpSeedHid.as<float>();
    for (std::size_t s = 0; s < nSeq; ++s) st.nextnLen[s] = 0;
    for (std::size_t g = 0; g + 1 < P; ++g) {
        for (std::size_t s = 0; s < nSeq; ++s) {
            _e._ops->appendMemoryCopy(seedHid + s * d, promptHiddens + g * d,
                                      d * sizeof(float));
            st.mtpPrevTok[s]      = prompt[g + 1];
            st.mtpWriteBlockId[s] = static_cast<std::uint32_t>(s * bps + g / bsz);
            st.mtpWriteSlot[s]    = static_cast<std::int32_t>(g % bsz);
            st.mtpSeqLensH[s]     = static_cast<std::int32_t>(g + 1);
            st.mtpStartPosH[s]    = static_cast<std::int32_t>(g);
        }
        _e._ops->uploadHostBytes(st.mtpSeqLensDev.get(),  st.mtpSeqLensH.data(),
                                 nSeq * sizeof(std::int32_t));
        _e._ops->uploadHostBytes(st.mtpStartPosDev.get(), st.mtpStartPosH.data(),
                                 nSeq * sizeof(std::int32_t));
        arch::BatchedDecodeCtx ctx{};
        ctx.nSeq            = nSeq;
        ctx.pool            = st.pool.get();
        ctx.writeBlockId    = st.mtpWriteBlockId.data();
        ctx.writeSlot       = st.mtpWriteSlot.data();
        ctx.blockTablesDev  = static_cast<const std::int32_t*>(st.blockTablesDev.get());
        ctx.seqLensDev      = static_cast<const std::int32_t*>(st.mtpSeqLensDev.get());
        ctx.maxBlocksPerSeq = bps;
        ctx.startPosDev     = static_cast<const std::int32_t*>(st.mtpStartPosDev.get());
        ctx.expIdxSlot      = st.expIdxBuf.as<std::int32_t>();
        ctx.kwSlot          = st.kwBuf.as<float>();
        st.qb->runMtpDraftStepBatched(
            seedHid, st.mtpPrevTok.data(), nSeq, ctx, st.mtpPoolLayer, *st.sb,
            st.mtpEmbB.as<float>(), st.mtpCatB.as<float>(), st.mtpEhB.as<float>(),
            st.mtpTmpE.as<float>(), st.mtpTmpH.as<float>(),
            st.mtpDraftLogitsB.as<float>(), st.mtpLmScrB.as<float>(),
            /*skipHead=*/true);
        for (std::size_t s = 0; s < nSeq; ++s) ++st.nextnLen[s];
    }
}

void ServingSession::draftBatchRound(
        std::size_t nSeq, std::size_t K,
        const std::vector<std::int32_t>&        token0,
        std::vector<std::vector<std::int32_t>>& drafts) {
    auto& st = *_state;
    const std::size_t d     = st.d_model;
    const std::size_t vocab = st.vocab_lm;
    const std::size_t bps   = st.blocksPerSeq;
    const std::size_t bsz   = st.blockSize;
    float* const hidB = st.mtpHid.as<float>();
    float* const ehB  = st.mtpEhB.as<float>();
    for (std::size_t s = 0; s < nSeq; ++s) {
        drafts[s].clear();
        st.mtpPrevTok[s] = token0[s];
    }
    const float* hcur = hidB;
    for (std::size_t k = 0; k < K; ++k) {
        for (std::size_t s = 0; s < nSeq; ++s) {
            const std::size_t pos = st.nextnLen[s];
            st.mtpWriteBlockId[s] = static_cast<std::uint32_t>(s * bps + pos / bsz);
            st.mtpWriteSlot[s]    = static_cast<std::int32_t>(pos % bsz);
            st.mtpSeqLensH[s]     = static_cast<std::int32_t>(pos + 1);
            st.mtpStartPosH[s]    = static_cast<std::int32_t>(pos);
        }
        _e._ops->uploadHostBytes(st.mtpSeqLensDev.get(),  st.mtpSeqLensH.data(),
                                 nSeq * sizeof(std::int32_t));
        _e._ops->uploadHostBytes(st.mtpStartPosDev.get(), st.mtpStartPosH.data(),
                                 nSeq * sizeof(std::int32_t));
        arch::BatchedDecodeCtx ctx{};
        ctx.nSeq            = nSeq;
        ctx.pool            = st.pool.get();
        ctx.writeBlockId    = st.mtpWriteBlockId.data();
        ctx.writeSlot       = st.mtpWriteSlot.data();
        ctx.blockTablesDev  = static_cast<const std::int32_t*>(st.blockTablesDev.get());
        ctx.seqLensDev      = static_cast<const std::int32_t*>(st.mtpSeqLensDev.get());
        ctx.maxBlocksPerSeq = bps;
        ctx.startPosDev     = static_cast<const std::int32_t*>(st.mtpStartPosDev.get());
        ctx.expIdxSlot      = st.expIdxBuf.as<std::int32_t>();
        ctx.kwSlot          = st.kwBuf.as<float>();
        st.qb->runMtpDraftStepBatched(
            hcur, st.mtpPrevTok.data(), nSeq, ctx, st.mtpPoolLayer, *st.sb,
            st.mtpEmbB.as<float>(), st.mtpCatB.as<float>(), ehB,
            st.mtpTmpE.as<float>(), st.mtpTmpH.as<float>(),
            st.mtpDraftLogitsB.as<float>(), st.mtpLmScrB.as<float>(),
            /*skipHead=*/false);
        _e._ops->flush();
        _e._ops->readbackToHost(st.mtpHostLogitsB.data(), st.mtpDraftLogitsB.get(),
                                nSeq * vocab * sizeof(float));
        for (std::size_t s = 0; s < nSeq; ++s) {
            const float* row = st.mtpHostLogitsB.data() + s * vocab;
            std::size_t best = 0;
            float       bv   = row[0];
            for (std::size_t v = 1; v < vocab; ++v) {
                if (row[v] > bv) { bv = row[v]; best = v; }
            }
            drafts[s].push_back(static_cast<std::int32_t>(best));
            st.mtpPrevTok[s] = static_cast<std::int32_t>(best);
            ++st.nextnLen[s];
        }
        hcur = ehB;
    }
}

InferenceEngine::MtpDraftParityResult
ServingSession::mtpDraftParity(std::span<const std::int32_t> prompt,
                               std::size_t nSeq, std::size_t depth) {
    InferenceEngine::MtpDraftParityResult res{};
    if (_e._backend == nullptr) {
        throw std::runtime_error("mtpDraftParity: no model loaded");
    }
    auto* qb = dynamic_cast<arch::Qwen35MoeBackend*>(_e._backend.get());
    if (qb == nullptr || !_e.mtpAvailable()) {
        throw std::runtime_error(
            "mtpDraftParity: requires CUDA qwen35moe with a nextn head");
    }
    if (prompt.empty()) {
        throw std::runtime_error("mtpDraftParity: empty prompt");
    }
    if (nSeq == 0)  nSeq = 1;
    if (depth == 0) depth = 1;
    const std::size_t P = prompt.size();
    const std::size_t d = _e._config.embeddingLength;

    ensureServingState(nSeq, P + depth + 8);
    ensureMtpServingState();
    auto& st = *_state;

    // --- trunk prefill: single-session forwardVerify(prompt) -------------
    // Leaves the P trunk hiddens in _xBufH (untouched by draft steps, which
    // write only into MTP scratch); last row = the draft seed hidden.
    _e.resetCache();
    const std::vector<std::int32_t> pvec(prompt.begin(), prompt.end());
    const auto pf = _e.forwardVerify(pvec);
    float* const xBuf = _e._xBufH.as<float>();
    const float* const hidS = xBuf + (P - 1) * d;
    std::size_t t0 = 0;
    {
        const auto& last = pf.back();
        float bv = last[0];
        for (std::size_t v = 1; v < last.size(); ++v) {
            if (last[v] > bv) { bv = last[v]; t0 = v; }
        }
    }
    const std::int32_t token0 = static_cast<std::int32_t>(t0);

    // Seed a nextn KV cache by replaying the prompt (P-1 steps), exactly
    // like MtpDecoder's prefill (runMtpDraftStep over prompt[1..P-1]).
    auto seed = [&](KvCache& kv) {
        kv.reset();
        float* const emb  = st.mtpEmb.as<float>();
        float* const cat  = st.mtpCat.as<float>();
        float* const eh   = st.mtpEh.as<float>();
        float* const dlog = st.mtpDraftLogits.as<float>();
        float* const lmSc = st.mtpLmScr.as<float>();
        for (std::size_t p = 0; p + 1 < P; ++p) {
            st.qb->runMtpDraftStep(xBuf + p * d, prompt[p + 1], kv, *st.sb,
                                   emb, cat, eh, dlog, lmSc);
            kv.commit(1);
        }
    };

    // --- reference: independent single-sequence nextn KV cache -----------
    KvCache refKv(*_e._ops, /*nLayers=*/1, st.maxContext,
                  _e._config.headCountKv, _e._config.headDim(), _e._kvDtype);
    seed(refKv);
    draftKInto(refKv, hidS, token0, depth, res.refDrafts);

    // --- serving path: one per-slot MTP-KV per sequence ------------------
    res.slotDrafts.assign(nSeq, {});
    res.allSlotsAgree    = true;
    res.matchesReference = true;
    for (std::size_t s = 0; s < nSeq; ++s) {
        seed(*st.mtpKv[s]);
        draftKInto(*st.mtpKv[s], hidS, token0, depth, res.slotDrafts[s]);
        if (res.slotDrafts[s] != res.slotDrafts[0]) res.allSlotsAgree = false;
        if (res.slotDrafts[s] != res.refDrafts)     res.matchesReference = false;
    }
    return res;
}

void ServingSession::restoreSlotSsm(std::size_t slot, std::size_t snapIdx) {
    auto& st = *_state;
    float* const       stDst = st.ssm->statePtr();
    float* const       cvDst = st.ssm->convStatePtr();
    const float* const stSrc = st.ssmSnap[snapIdx].as<float>();
    const float* const cvSrc = st.convSnap[snapIdx].as<float>();
    const std::size_t stStride = st.ssm->stateLayerStride();
    const std::size_t cvStride = st.ssm->convStateLayerStride();
    const std::size_t stElems  = st.ssm->stateElemsPerLayer();
    const std::size_t cvElems  = st.ssm->convStateElemsPerLayer();
    for (std::size_t L = 0; L < st.blockCount; ++L) {
        if (!_e._config.isRecurrentLayer(L)) {
            continue;   // full-attention layers keep no recurrent state
        }
        _e._ops->appendMemoryCopy(stDst + L * stStride + slot * stElems,
                                  stSrc + L * stStride + slot * stElems,
                                  stElems * sizeof(float));
        _e._ops->appendMemoryCopy(cvDst + L * cvStride + slot * cvElems,
                                  cvSrc + L * cvStride + slot * cvElems,
                                  cvElems * sizeof(float));
    }
}

std::vector<std::vector<std::int32_t>>
ServingSession::generateBatchMtp(std::span<const std::int32_t> prompt,
                                 std::size_t nSeq, std::size_t maxNew,
                                 std::size_t depth, std::int32_t eosId) {
    if (_e._backend == nullptr) {
        throw std::runtime_error("generateBatchMtp: no model loaded");
    }
    auto* qb = dynamic_cast<arch::Qwen35MoeBackend*>(_e._backend.get());
    if (qb == nullptr || !_e.mtpAvailable()) {
        throw std::runtime_error(
            "generateBatchMtp: requires CUDA qwen35moe with a nextn head");
    }
    if (nSeq == 0)  nSeq = 1;
    if (depth == 0) depth = 1;
    if (prompt.empty() || maxNew == 0) {
        return std::vector<std::vector<std::int32_t>>(nSeq);
    }
    const std::size_t P = prompt.size();
    const std::size_t d = _e._config.embeddingLength;

    ensureServingState(nSeq, P + maxNew + 8);
    ensureMtpServingState();
    ensureVerifyCapacity(depth);
    auto& st = *_state;

    auto argmax = [](const std::vector<float>& row) -> std::int32_t {
        std::size_t best = 0;
        float       bv   = row[0];
        for (std::size_t v = 1; v < row.size(); ++v) {
            if (row[v] > bv) { bv = row[v]; best = v; }
        }
        return static_cast<std::int32_t>(best);
    };

    // --- trunk prefill: single-session forwardVerify(prompt) -------------
    // Provides token0 + the trunk hiddens used to seed the nextn KV and the
    // first-round draft hidden (matches single-session generateMtp exactly).
    _e.resetCache();
    const std::vector<std::int32_t> pvec(prompt.begin(), prompt.end());
    const auto pf = _e.forwardVerify(pvec);
    float* const xBufH = _e._xBufH.as<float>();
    const std::int32_t token0v = argmax(pf.back());

    // Increment E5b: seed the BATCHED nextn KV (paged pool layer, all slots
    // at once) by replaying the identical prompt, and broadcast the last
    // trunk hidden into each slot's round hidden.
    float* const mtpHid = st.mtpHid.as<float>();
    mtpSeedBatched(nSeq, xBufH, prompt);
    for (std::size_t s = 0; s < nSeq; ++s) {
        _e._ops->appendMemoryCopy(mtpHid + s * d, xBufH + (P - 1) * d,
                                  d * sizeof(float));
    }

    // --- seed the paged trunk KV + SSM via a lockstep stepServing prefill.
    for (std::size_t g = 0; g < P; ++g) {
        std::vector<InferenceEngine::ServingSlotStep> steps(nSeq);
        for (std::size_t s = 0; s < nSeq; ++s) {
            steps[s].slot     = static_cast<std::uint32_t>(s);
            steps[s].token    = prompt[g];
            steps[s].pos      = static_cast<std::int32_t>(g);
            steps[s].seqStart = (g == 0);
        }
        std::vector<std::int32_t> toks(nSeq, 0);
        stepServing(steps, toks);
    }

    // --- per-slot decode state -------------------------------------------
    std::vector<std::size_t>               basePos(nSeq, P);
    std::vector<std::int32_t>              token0(nSeq, token0v);
    std::vector<std::vector<std::int32_t>> out(nSeq);
    std::vector<char>                      finished(nSeq, 0);

    while (true) {
        std::size_t nDone = 0;
        for (std::size_t s = 0; s < nSeq; ++s) nDone += (finished[s] != 0);
        if (nDone == nSeq) break;
        const std::size_t produced = out[0].size();
        if (produced >= maxNew) break;
        const std::size_t K = std::min(depth, maxNew - produced);

        // --- draft K tokens for ALL slots in one batched nextn pass (E5b) -
        std::vector<std::size_t>               nextnPre(nSeq);
        std::vector<std::vector<std::int32_t>> drafts(nSeq);
        for (std::size_t s = 0; s < nSeq; ++s) nextnPre[s] = st.nextnLen[s];
        draftBatchRound(nSeq, K, token0, drafts);

        // --- one batched verify over [token0, drafts...] per slot --------
        std::vector<InferenceEngine::VerifySlot> slots(nSeq);
        std::vector<std::int32_t> vtokTM((K + 1) * nSeq);
        for (std::size_t s = 0; s < nSeq; ++s) {
            slots[s].slot    = static_cast<std::uint32_t>(s);
            slots[s].basePos = static_cast<std::int32_t>(basePos[s]);
            vtokTM[0 * nSeq + s] = token0[s];
            for (std::size_t j = 1; j <= K; ++j) {
                vtokTM[j * nSeq + s] = drafts[s][j - 1];
            }
        }
        const auto vlog = stepServingVerify(slots, vtokTM, K);
        float* const vX = st.vXBuf.as<float>();   // M trunk hiddens (time-major)

        // --- per-slot accept-longest-prefix + snapshot restore -----------
        for (std::size_t s = 0; s < nSeq; ++s) {
            if (finished[s] != 0) continue;
            std::size_t a = 0;
            for (std::size_t i = 0; i < K; ++i) {
                if (argmax(vlog[i * nSeq + s]) == drafts[s][i]) ++a; else break;
            }
            const std::int32_t corrected = argmax(vlog[a * nSeq + s]);

            auto emit = [&](std::int32_t t) -> bool {
                out[s].push_back(t);
                if (eosId >= 0 && t == eosId) { finished[s] = 1; return false; }
                if (out[s].size() >= maxNew)  { finished[s] = 1; return false; }
                return true;
            };
            bool cont = emit(token0[s]);
            for (std::size_t i = 0; i < a && cont; ++i) cont = emit(drafts[s][i]);

            // Commit: KV for the accepted a+1 tokens is already correct (the
            // verify wrote it), so committing is just advancing the slot's
            // length. Restore the GatedDeltaNet state to the accepted step
            // (no re-forward), truncate the nextn KV, and carry the trunk
            // hidden at the accepted position into the next round's draft.
            basePos[s] += a + 1;
            if (a < K) restoreSlotSsm(s, a);
            st.nextnLen[s] = nextnPre[s] + a + 1;
            _e._ops->appendMemoryCopy(mtpHid + s * d, vX + (a * nSeq + s) * d,
                                      d * sizeof(float));
            token0[s] = corrected;
        }
        _e._ops->flush();
    }
    return out;
}

std::vector<std::vector<std::int32_t>>
ServingSession::generateBatchMtpMulti(
        const std::vector<std::vector<std::int32_t>>& prompts,
        std::size_t maxNew, std::size_t depth, std::int32_t eosId) {
    if (_e._backend == nullptr) {
        throw std::runtime_error("generateBatchMtpMulti: no model loaded");
    }
    auto* qb = dynamic_cast<arch::Qwen35MoeBackend*>(_e._backend.get());
    if (qb == nullptr || !_e.mtpAvailable()) {
        throw std::runtime_error(
            "generateBatchMtpMulti: requires CUDA qwen35moe with a nextn head");
    }
    const std::size_t N = prompts.size();
    if (N == 0 || maxNew == 0) {
        return std::vector<std::vector<std::int32_t>>(N);
    }
    if (depth == 0) depth = 1;
    for (const auto& p : prompts) {
        if (p.empty()) {
            throw std::runtime_error("generateBatchMtpMulti: empty prompt");
        }
    }
    const std::size_t d = _e._config.embeddingLength;

    // Physical slot p handles prompt order[p], sorted by DESCENDING length so
    // the token-by-token prefill always steps a contiguous slot prefix
    // (slots drop out of the prefill prefix once their prompt is exhausted).
    std::vector<std::size_t> order(N);
    for (std::size_t i = 0; i < N; ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t a, std::size_t b) {
                         return prompts[a].size() > prompts[b].size();
                     });
    const std::size_t maxLen = prompts[order[0]].size();

    // maxContext must hold the longest prompt plus the worst-case commit
    // growth: a slot rides along (padded) until the slowest slot finishes —
    // at most maxNew rounds, each committing at most depth+1 tokens.
    ensureServingState(N, maxLen + maxNew * (depth + 1) + 8);
    ensureMtpServingState();
    ensureVerifyCapacity(depth);
    auto& st = *_state;

    auto argmax = [](const std::vector<float>& row) -> std::int32_t {
        std::size_t best = 0;
        float       bv   = row[0];
        for (std::size_t v = 1; v < row.size(); ++v) {
            if (row[v] > bv) { bv = row[v]; best = v; }
        }
        return static_cast<std::int32_t>(best);
    };

    float* const mtpHid = st.mtpHid.as<float>();

    // --- per-slot prefill: single-session forwardVerify(prompt) for the
    //     first token + trunk hidden + nextn-KV seed. (The MTP head only
    //     proposes drafts, so seed/hidden fidelity affects accept-rate, not
    //     the greedy output — the trunk verify guarantees correctness.) -----
    std::vector<std::int32_t> token0(N);
    std::vector<std::size_t>  basePos(N);
    for (std::size_t p = 0; p < N; ++p) {
        const auto& pr = prompts[order[p]];
        const std::size_t P = pr.size();
        basePos[p] = P;
        _e.resetCache();
        const auto pf = _e.forwardVerify(pr);
        float* const xBufH = _e._xBufH.as<float>();
        token0[p] = argmax(pf.back());
        st.mtpKv[p]->reset();
        {
            float* const emb  = st.mtpEmb.as<float>();
            float* const cat  = st.mtpCat.as<float>();
            float* const eh   = st.mtpEh.as<float>();
            float* const dlog = st.mtpDraftLogits.as<float>();
            float* const lmSc = st.mtpLmScr.as<float>();
            for (std::size_t q = 0; q + 1 < P; ++q) {
                st.qb->runMtpDraftStep(xBufH + q * d, pr[q + 1], *st.mtpKv[p],
                                       *st.sb, emb, cat, eh, dlog, lmSc);
                st.mtpKv[p]->commit(1);
            }
        }
        _e._ops->appendMemoryCopy(mtpHid + p * d, xBufH + (P - 1) * d,
                                  d * sizeof(float));
        _e._ops->flush();   // land the hidden copy before the next slot's forward
    }

    // --- seed the paged trunk KV + SSM via a sorted-descending lockstep
    //     stepServing prefill (each step advances a contiguous prefix). -----
    for (std::size_t g = 0; g < maxLen; ++g) {
        std::size_t k = 0;
        for (std::size_t p = 0; p < N; ++p) {
            if (prompts[order[p]].size() > g) ++k; else break;
        }
        std::vector<InferenceEngine::ServingSlotStep> steps(k);
        for (std::size_t p = 0; p < k; ++p) {
            steps[p].slot     = static_cast<std::uint32_t>(p);
            steps[p].token    = prompts[order[p]][g];
            steps[p].pos      = static_cast<std::int32_t>(g);
            steps[p].seqStart = (g == 0);
        }
        std::vector<std::int32_t> toks(k, 0);
        stepServing(steps, toks);
    }

    // --- per-slot divergent decode: all slots draft+verify+accept each
    //     round at their OWN positions; a finished slot stops emitting but
    //     keeps riding along (padding) so the batch stays a contiguous
    //     prefix. Every active slot emits >= 1 token/round, so <= maxNew
    //     rounds suffice for the slowest slot. --------------------------------
    std::vector<std::vector<std::int32_t>> out(N);
    std::vector<char>                      finished(N, 0);
    for (std::size_t round = 0; round < maxNew; ++round) {
        std::size_t nDone = 0;
        for (const char f : finished) nDone += (f != 0);
        if (nDone == N) break;
        const std::size_t K = depth;

        std::vector<std::size_t>               mtpPre(N);
        std::vector<std::vector<std::int32_t>> drafts(N);
        for (std::size_t p = 0; p < N; ++p) {
            mtpPre[p] = st.mtpKv[p]->length();
            draftKInto(*st.mtpKv[p], mtpHid + p * d, token0[p], K, drafts[p]);
        }

        std::vector<InferenceEngine::VerifySlot> slots(N);
        std::vector<std::int32_t> vtokTM((K + 1) * N);
        for (std::size_t p = 0; p < N; ++p) {
            slots[p].slot    = static_cast<std::uint32_t>(p);
            slots[p].basePos = static_cast<std::int32_t>(basePos[p]);
            vtokTM[0 * N + p] = token0[p];
            for (std::size_t j = 1; j <= K; ++j) {
                vtokTM[j * N + p] = drafts[p][j - 1];
            }
        }
        const auto vlog = stepServingVerify(slots, vtokTM, K);
        float* const vX = st.vXBuf.as<float>();

        for (std::size_t p = 0; p < N; ++p) {
            std::size_t a = 0;
            for (std::size_t i = 0; i < K; ++i) {
                if (argmax(vlog[i * N + p]) == drafts[p][i]) ++a; else break;
            }
            const std::int32_t corrected = argmax(vlog[a * N + p]);
            if (finished[p] == 0) {
                auto emit = [&](std::int32_t t) -> bool {
                    out[p].push_back(t);
                    if (eosId >= 0 && t == eosId) { finished[p] = 1; return false; }
                    if (out[p].size() >= maxNew)  { finished[p] = 1; return false; }
                    return true;
                };
                bool cont = emit(token0[p]);
                for (std::size_t i = 0; i < a && cont; ++i) cont = emit(drafts[p][i]);
            }
            basePos[p] += a + 1;
            if (a < K) restoreSlotSsm(p, a);
            st.mtpKv[p]->truncate(mtpPre[p] + a + 1);
            _e._ops->appendMemoryCopy(mtpHid + p * d, vX + (a * N + p) * d,
                                      d * sizeof(float));
            token0[p] = corrected;
        }
        _e._ops->flush();
    }

    // --- restore input order ---------------------------------------------
    std::vector<std::vector<std::int32_t>> result(N);
    for (std::size_t p = 0; p < N; ++p) result[order[p]] = std::move(out[p]);
    return result;
}

std::size_t ServingSession::maxBatch() const noexcept {
    return _state != nullptr ? _state->maxBatch : 0;
}

std::size_t ServingSession::maxContext() const noexcept {
    return _state != nullptr ? _state->maxContext : 0;
}

} // namespace mimirmind::runtime::engine
