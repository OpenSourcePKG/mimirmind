// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/engine/ServingSession.hpp"

#include "compute/Embedding.hpp"
#include "core/log/Log.hpp"
#include "runtime/BlockBuffers.hpp"
#include "runtime/KvCache.hpp"
#include "runtime/SsmState.hpp"
#include "runtime/arch/Qwen3_5MoeBackend.hpp"
#include "runtime/engine/DFlashDecoder.hpp"
#include "runtime/serving/KvCacheSlabPool.hpp"
#include "runtime/serving/PagedKvPool.hpp"
#include "runtime/serving/SlabDecodeStepper.hpp"
#ifdef MIMIRMIND_HAVE_CUDA
#include "core/gpu/cuda/CudaGraph.hpp"
#include "compute/cuda/GpuOps.hpp"
#endif

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

// Serving KV cache element dtype for the CUDA paged pool (5.14 I1). FP16
// halves the KV read bandwidth of growing-context attention + halves KV
// memory. Dev-gated for A/B + parity; F32 stays the default so prod is
// untouched until the win is measured. The server (not a user toggle) will
// own the final default once validated.
namespace {
[[nodiscard]] KvDtype servingKvDtype() noexcept {
    const char* e = std::getenv("MIMIRMIND_SERVING_KV_FP16");
    const bool fp16 = (e != nullptr && e[0] != '\0' && !(e[0] == '0' && e[1] == '\0'));
    return fp16 ? KvDtype::FP16 : KvDtype::F32;
}
} // namespace

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
    arch::Qwen3_5MoeBackend*       qb{nullptr};

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

    // ---- Increment A: chunked multi-token prefill scratch ---------------
    // A newly-admitted request's prompt is prefilled as a T>1 forward per
    // physical slot (reusing the single-session runBlock path over the slot's
    // contiguous paged-KV region + its SsmState slice) instead of token-by-
    // token. sbPrefill is sized for maxT = prefillChunk rows; xBufP holds the
    // chunk's hidden states. Built only when chunked prefill is enabled
    // (MIMIRMIND_CHUNKED_PREFILL != 0; chunk size MIMIRMIND_PREFILL_CHUNK).
    bool        chunkedPrefill{false};
    std::size_t prefillChunk{0};               // C (tokens per prefill forward)
    std::optional<BlockBuffers> sbPrefill;     // maxT = prefillChunk
    compute::ComputeBuffer      xBufP;         // [prefillChunk, d_model]
    std::vector<std::int32_t>   prefillTokH;   // [prefillChunk] host token staging
    // Device MoE routing scratch for the amortised batched-fused-K MoE the
    // prefill forwards use (runMoeFfnBatched via setPrefillMoeScratch).
    compute::ComputeBuffer      prefillExpIdx; // [prefillChunk * expertUsedCount]
    compute::ComputeBuffer      prefillKw;     // [prefillChunk * expertUsedCount]
    // 5.21-III Teil 4 — varlen prefill driver (prefillSlotVarlen). Per-token ctx
    // scratch: rope positions + KV write targets [prefillChunk]; per-slot (nSeq=1)
    // seqT/seqOff/convInOff/startPos [1]. Host staging + device copies.
    compute::ComputeBuffer      vlRopePosDev, vlWriteBlockIdDev, vlWriteSlotDev;
    compute::ComputeBuffer      vlSeqTDev, vlSeqOffDev, vlConvInOffDev, vlStartPosDev;
    std::vector<std::int32_t>   vlRopePosH, vlWriteSlotH, vlStartPosH;
    std::vector<std::uint32_t>  vlWriteBlockIdH;

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
    compute::ComputeBuffer vArgmaxDev;               // [Mcap] device per-row argmax ids
    std::vector<std::int32_t>  vArgmaxHost;          // [Mcap] argmax readback (draft+verify)
    // MV-d: compact per-position recurrent-state export written by the fused
    // verify kernel — [blockCount, Kp1, maxBatch, stateElems], time-major on
    // the verify position, packed with the runtime slot stride N. Replaces the
    // K+1 full-slab recurrent snapshots. conv state still snapshotted per
    // position (cheap) into convSnap.
    compute::ComputeBuffer ssmExport;
    std::vector<compute::ComputeBuffer> convSnap;    // [K+1] full SsmState conv images
    std::vector<std::int32_t>  vBlockTablesH;        // [Mcap * blocksPerSeq]
    std::vector<std::int32_t>  vSeqLensH, vStartPosH;
    std::vector<std::uint32_t> vWriteBlockId;
    std::vector<std::int32_t>  vWriteSlot;
    std::vector<std::uint8_t>  vIsSeqStart;          // full-attn virt-slot start flags
    std::vector<std::uint8_t>  vGdnSeqStart;         // [(K+1) * maxBatch] per-step GDN start flags
    std::vector<std::int32_t>  vInputTok;
    std::vector<float>         vHostLogits;          // [Mcap * vocab_lm]

    // ---- DFlash serving tap capture (5.9.1) -----------------------------
    // When armed by generateBatchDflash, verifyForward copies xBuf (the
    // [M=N*(K+1), d] time-major residual stream, row j*N+s = slot s / verify
    // position j) after each tapped block {1,6,11,16,22,27,32,37} into per-tap
    // sinks [verifyMcap, d], so each slot's 8-tap context is built from its
    // accepted verify positions (no separate tap forward). Inert when unarmed.
    bool                                dfTapActive{false};
    std::vector<int>                    dfTapSlot;   // [blockCount] tap-slot or -1
    std::vector<compute::ComputeBuffer> dfTapSink;   // [taps] each [verifyMcap, d]

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

// =======================================================================
// Non-paged L0 / Xe-LPG serving substrate (M9.1 / ADR 2026-08-18). One
// contiguous KvCache slab per slot (KvCacheSlabPool) driven by the
// backend-neutral ArchBackend::runBlockBatched via SlabDecodeStepper. No
// paged pool, no SsmState, no MoE routing scratch — Gemma 4 has no GDN and
// all layers hold KV, so the slab pool IS the per-sequence KvCache contract
// runBlockBatched already writes into. Selected when the loaded backend is
// not qwen35moe but implements supportsBatchedDecode() (Gemma 4 MoE).
// =======================================================================
struct L0ServingState {
    std::size_t maxBatch{0};
    std::size_t maxContext{0};
    std::size_t prefillChunk{0};

    std::unique_ptr<serving::KvCacheSlabPool>   slab;
    std::unique_ptr<serving::SlabDecodeStepper> stepper;
    std::optional<BlockBuffers>                 decodeSb;   // maxT = maxBatch
    std::optional<BlockBuffers>                 prefillSb;  // maxT = prefillChunk

    std::vector<std::int32_t> stepTokens;   // [maxBatch] decode input staging
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
    auto* qb = dynamic_cast<arch::Qwen3_5MoeBackend*>(_e._backend.get());
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
                              nKvHeads, headDim, servingKvDtype());

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
    // flush) vs GPU-drain (a single flush after). NOTE: host-enqueue time is
    // NOT purely CUDA-graph-removable launch overhead — once the GPU is
    // saturated the async launches block on queue back-pressure, so dispatch
    // then tracks GPU compute, not removable overhead. Use the nSeq=1 dispatch
    // (~no back-pressure) as the genuine launch/host-overhead baseline; growth
    // beyond that with nSeq is dominated by GPU back-pressure (GPU-bound).
    // Distinct from `timing` (which flushes per block and thus serialises).
    const bool ohMode = std::getenv("MIMIRMIND_BATCH_OVERHEAD") != nullptr;
    double tFull = 0.0, tLin = 0.0, tLm = 0.0, tPre = 0.0;
    double tDisp = 0.0, tDrain = 0.0;
    double tDispFull = 0.0, tDispLin = 0.0;
    std::size_t ohSteps = 0;
    using clk = std::chrono::steady_clock;

    // Serving-decode CUDA-graph probe (Milestone 0): capture the device-driven
    // block loop once (after a warmup step so lazy decode-shape work is done) and
    // replay it. NOTE: correctness needs the per-step ctx values
    // (writeBlockId/writeSlot/maxSeqLen) to become device slots updated OUTSIDE
    // the graph; until then replay reuses the capture step's KV write location,
    // so this probe measures capturability + replay speed, not correct tokens.
    // Env-gated, CUDA-only.
#ifdef MIMIRMIND_HAVE_CUDA
    // Piecewise serving-decode CUDA graph (vLLM-style): capture the RECURRENT
    // (GDN) blocks — fixed-shape at T=1, no paged attention, no paged-KV write —
    // one graph per block, and run the full-attention blocks EAGER (their varlen
    // paged attention reads the true seqLensDev, so no context-length is baked
    // into a graph). The GDN blocks are ~9.4ms of the ~12.9ms host launch
    // overhead, so this captures the bulk while sidestepping varlen entirely.
    const bool servingGraphOn = std::getenv("MIMIRMIND_SERVING_GRAPH") != nullptr;
    std::vector<core::cuda::CudaGraph> blockGraphs;
    bool        servingGraphFailed = false;
    std::size_t decStep            = 0;
    if (servingGraphOn) {
        blockGraphs.resize(blockCount);
    }
#endif

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
        ctx.maxSeqLen       = static_cast<std::int32_t>(p + 1); // uniform bench pos
        ctx.startPosDev     = static_cast<const std::int32_t*>(startPosDev.get());
        ctx.expIdxSlot      = expIdxBuf.as<std::int32_t>();
        ctx.kwSlot          = kwBuf.as<float>();
        ctx.isSeqStart      = isSeqStart.data();

        if (timing) {
            _e._ops->flush();
            tPre += std::chrono::duration<double, std::milli>(clk::now() - tp0).count();
        }
        if (ohMode) {
            // Pure host enqueue per block (no per-block flush → non-serialising,
            // GPU work overlaps), bucketed by block class, then one drain.
            // Skip the first steps (first-touch PTX JIT / warm-up).
            double dFull = 0.0, dLin = 0.0;
            for (std::size_t b = 0; b < blockCount; ++b) {
                const auto tb0 = clk::now();
                qb->runBlockBatched(b, xBuf, ctx, sb);
                const double dt =
                    std::chrono::duration<double, std::milli>(clk::now() - tb0).count();
                if (_e._config.isRecurrentLayer(b)) dLin += dt; else dFull += dt;
            }
            const double tdisp = dFull + dLin;
            const auto tf0 = clk::now();
            _e._ops->flush();
            const double tdrain =
                std::chrono::duration<double, std::milli>(clk::now() - tf0).count();
            if (p >= 4) {
                tDisp     += tdisp;
                tDrain    += tdrain;
                tDispFull += dFull;
                tDispLin  += dLin;
                ++ohSteps;
            }
        } else {
#ifdef MIMIRMIND_HAVE_CUDA
            if (servingGraphOn && !servingGraphFailed && !timing) {
                auto& gstream = _e.cudaOps().stream();
                // Piecewise by SEGMENT: capture each contiguous run of recurrent
                // (GDN) blocks as ONE graph (collapsing its many kernel launches
                // into a single replay — this is where the launch-overhead win
                // is), and run the full-attention blocks eager (varlen). The
                // segment's graph is stored at its start-block index.
                for (std::size_t b = 0; b < blockCount; ) {
                    if (!_e._config.isRecurrentLayer(b)) {
                        qb->runBlockBatched(b, xBuf, ctx, sb);      // full-attn eager
                        ++b;
                        continue;
                    }
                    std::size_t segEnd = b;
                    while (segEnd < blockCount &&
                           _e._config.isRecurrentLayer(segEnd)) {
                        ++segEnd;
                    }
                    auto runSeg = [&] {
                        for (std::size_t bb = b; bb < segEnd; ++bb) {
                            qb->runBlockBatched(bb, xBuf, ctx, sb);
                        }
                    };
                    core::cuda::CudaGraph& g = blockGraphs[b];
                    if (g.valid()) {
                        g.launch(gstream);
                    } else if (decStep >= 1) {                      // warmup at 0
                        try {
                            g.capture(gstream, runSeg);
                            g.launch(gstream);
                        } catch (const std::exception& ex) {
                            MM_LOG_WARN("serving-graph",
                                "segment [{}, {}) capture FAILED: {} — eager",
                                b, segEnd, ex.what());
                            servingGraphFailed = true;
                            runSeg();
                        }
                    } else {
                        runSeg();                                   // warmup
                    }
                    b = segEnd;
                }
                if (decStep == 1 && !servingGraphFailed) {
                    MM_LOG_INFO("serving-graph",
                        "piecewise capture done (recurrent blocks graphed, "
                        "full-attn eager)");
                }
                ++decStep;
            } else
#endif
            {
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
        }
    };

    // Greedy sampling on-device: compute the per-row argmax with a device kernel
    // and read back only nSeq token ids (4 B each), instead of the full logits
    // (nSeq * vocab_lm floats = ~600 KB/seq) plus a host argmax over the whole
    // vocabulary. That host round-trip is a big per-step cost that CUDA graphs
    // do not remove (it is not launch overhead) and that vLLM avoids by sampling
    // on the GPU.
    auto argmaxDev = _e._ops->allocate(nSeq * sizeof(std::int32_t));
    std::vector<std::int32_t> toksHost(nSeq);
    auto lmHeadSample = [&]() -> std::vector<std::int32_t> {
        _e._ops->rmsNormAsync(xBuf, nSeq, d_model,
                              static_cast<const float*>(outNorm->usmPtr),
                              _e._config.rmsNormEps, normBuf);
        _e._gmm->matmul(lmHead->type, lmHead->usmPtr, vocab_lm, d_model,
                        normBuf, nSeq, logits, lmScr.as<float>());
        _e._ops->argmaxRowsAsync(logits, argmaxDev.as<std::int32_t>(),
                                 nSeq, vocab_lm);
        _e._ops->flush();
        _e._ops->readbackToHost(toksHost.data(), argmaxDev.get(),
                                nSeq * sizeof(std::int32_t));
        return toksHost;
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
            "=> host-enqueue=%.1f%% (launch overhead + GPU back-pressure; "
            "compare vs nSeq=1 for the removable share)\n"
            "               dispatch split: full-attn-blocks=%.3f ms/step  "
            "linear/GDN-blocks=%.3f ms/step\n",
            nSeq, ohSteps, blockCount, disp, drain, frac,
            tDispFull / static_cast<double>(ohSteps),
            tDispLin / static_cast<double>(ohSteps));
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
    auto* qb = dynamic_cast<arch::Qwen3_5MoeBackend*>(_e._backend.get());
    if (qb == nullptr) {
        // L0 / Xe-LPG path: a backend implementing the neutral synchronized
        // batched decode (Gemma 4 MoE) serves through the non-paged slab
        // substrate instead of the qwen35moe paged pool.
        if (!_e._backend->supportsBatchedDecode()) {
            throw std::runtime_error(
                "ensureServingState: continuous batching requires qwen35moe or "
                "a backend with supportsBatchedDecode()");
        }
        if (maxBatch == 0 || maxContext == 0) {
            throw std::runtime_error(
                "ensureServingState: maxBatch/maxContext must be > 0");
        }
        if (_l0 != nullptr && _l0->maxBatch >= maxBatch &&
            _l0->maxContext >= maxContext) {
            return;   // idempotent unless capacity must grow
        }

        const auto* tokEmb  = _e._weights->find("token_embd.weight");
        const auto* outNorm = _e._weights->find("output_norm.weight");
        const auto* lmHead  = _e._weights->find("output.weight");
        if (lmHead == nullptr) {
            lmHead = tokEmb;
        }
        if (tokEmb == nullptr || outNorm == nullptr || lmHead == nullptr) {
            throw std::runtime_error(
                "ensureServingState: embed/norm/lm_head missing");
        }

        auto l0 = std::make_unique<L0ServingState>();
        l0->maxBatch   = maxBatch;
        l0->maxContext = maxContext;
        // One prefill forward handles up to prefillChunk prompt tokens; the
        // batcher splits longer prompts into successive appends.
        l0->prefillChunk = std::min<std::size_t>(maxContext, 512);

        // Optional blocked-NVFP4 lm_head sibling (loader step 5f-lmhead); the
        // stepper dispatches it only at low batch (MIMIRMIND_LMHEAD_NVFP4).
        const auto* lmHeadNv = _e._weights->find("output.weight.nv");
        serving::SlabDecodeStepper::Weights w{tokEmb, outNorm, lmHead, lmHeadNv};
        serving::SlabDecodeStepper::Dims dims{};
        dims.dModel   = _e._config.embeddingLength;
        dims.vocabLm  = lmHead->dimensions.size() >= 2 ? lmHead->dimensions[1]
                                                       : _e._tokenizer.vocabSize();
        dims.vocabEmb = tokEmb->dimensions.size() >= 2 ? tokEmb->dimensions[1]
                                                       : _e._tokenizer.vocabSize();
        dims.blockCount     = _e._config.blockCount;
        dims.rmsNormEps     = _e._config.rmsNormEps;
        dims.scaleEmbedding = _e._backend->scalesEmbedding();

        // F32 KV only. The Gemma 4 batched-decode attention (M-L0.Batch
        // Phase 1, runAttentionSectionBatched) supports F32 KV exclusively —
        // it throws on Q8_0/FP16 — so the serving slab substrate is always
        // F32, independent of the engine's single-session `_kvDtype` (which
        // may be Q8_0 on the NUC). This mirrors PagedKvPool's F32 baseline;
        // the two KV substrates and the single-session cache are separate.
        const KvDtype kvDtype = KvDtype::F32;
        l0->slab = serving::KvCacheSlabPool::forBackend(
            *_e._ops, *_e._backend, maxBatch, maxContext, kvDtype);
        l0->stepper = std::make_unique<serving::SlabDecodeStepper>(
            *_e._ops, *_e._gmm, *_e._backend, *l0->slab, w, dims,
            l0->prefillChunk);

        const auto [qDimMax, kvDimMax] = _e._backend->maxQKVDims();
        const bool withFusedQkv =
            _e._fusedQkv != nullptr && _e._fusedQkv->anyFused();
        const bool withKvFp32Scratch =
            (kvDtype == KvDtype::Q8_0 || kvDtype == KvDtype::FP16);
        const bool withQGate = _e._backend->needsQGateScratch();
        const bool withSsm   = _e._backend->needsSsmScratch();
        l0->decodeSb = allocBlockBuffers(
            *_e._ops, _e._config, /*maxT=*/maxBatch, /*maxSeq=*/maxContext,
            qDimMax, kvDimMax, withFusedQkv, withKvFp32Scratch,
            withQGate, withSsm);
        l0->prefillSb = allocBlockBuffers(
            *_e._ops, _e._config, /*maxT=*/l0->prefillChunk, /*maxSeq=*/maxContext,
            qDimMax, kvDimMax, withFusedQkv, withKvFp32Scratch,
            withQGate, withSsm);
        l0->stepTokens.resize(maxBatch);

        MM_LOG_INFO("serving",
                    "L0 slab serving state: maxBatch={} maxContext={} "
                    "prefillChunk={} (backend '{}')",
                    maxBatch, maxContext, l0->prefillChunk, _e._backend->name());
        _l0 = std::move(l0);
        return;
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
        *_e._ops, nPoolLayers, st->numBlocks, st->blockSize, nKvHeads, headDim,
        servingKvDtype());
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

    // ---- Increment A: chunked multi-token prefill scratch ---------------
    // Default ON with an env rollback (MIMIRMIND_CHUNKED_PREFILL=0 -> the
    // token-by-token prefill path in ContinuousBatcher stays). The prefill
    // forward reuses the single-session runBlock(T>1) path, so its scratch is
    // sized exactly like the single-session BlockBuffers (perSeqConvInput
    // false, no Q8_0 kv scratch), maxT = prefillChunk.
    {
        const char* cpEnv = std::getenv("MIMIRMIND_CHUNKED_PREFILL");
        st->chunkedPrefill = (cpEnv == nullptr) || !(cpEnv[0] == '0' && cpEnv[1] == '\0');
        // Tokens per prefill forward. A prompt longer than this ingests as
        // several chunked T>1 forwards, each carrying the KV + recurrent state
        // forward (the between-chunk flush in prefillSlot commits each chunk
        // before the next reads its state). Bounds the prefill scratch
        // (BlockBuffers + xBufP are sized for maxT = chunk) and the per-chunk
        // step latency. Overridable via MIMIRMIND_PREFILL_CHUNK.
        std::size_t chunk = 512;
        if (const char* c = std::getenv("MIMIRMIND_PREFILL_CHUNK")) {
            const long v = std::strtol(c, nullptr, 10);
            if (v >= 1) chunk = static_cast<std::size_t>(v);
        }
        // Never exceed the slot capacity; a chunk spanning more than the
        // context would overrun the slot's block run.
        chunk = std::min(chunk, maxContext);
        st->prefillChunk = chunk;
        if (st->chunkedPrefill) {
            // FP16 KV needs the fp32 staging scratch in the PREFILL path too
            // (the fp16Path projects K/V into kvKFp32Scratch before the
            // kv_commit_fp16 cast); F32 prefill writes the cache in place and
            // does not (5.14 I1).
            const bool prefillKvFp32Scratch =
                (servingKvDtype() == KvDtype::FP16);
            st->sbPrefill = allocBlockBuffers(
                *_e._ops, _e._config, /*maxT=*/chunk, /*maxSeq=*/maxContext,
                qkv.first, qkv.second, /*withFusedQkv=*/false,
                // 5.21-III: the varlen prefill driver uses runFullAttentionBlock-
                // Batched, whose kProj/vProj scratch is kvKFp32Scratch — always
                // allocate it (not just for FP16 KV) so the ragged path has it.
                /*withKvFp32Scratch=*/true, /*withQGate=*/true,
                /*withSsm=*/true, /*perSeqConvInput=*/false);
            // SSM state pointers are (re)bound per prefillSlot call to the
            // target slot's slab slice; ssmSlabNSeq is the full slab width so
            // the per-layer stride matches stepServing's slab layout.
            st->sbPrefill->ssmSlabNSeq = st->ssm->nSeq();
            st->xBufP = _e._ops->allocate(chunk * st->d_model * sizeof(float));
            st->prefillExpIdx = _e._ops->allocate(chunk * K * sizeof(std::int32_t));
            st->prefillKw     = _e._ops->allocate(chunk * K * sizeof(float));
            st->prefillTokH.resize(chunk);
            // 5.21-III varlen prefill driver scratch.
            st->vlRopePosDev      = _e._ops->allocate(chunk * sizeof(std::int32_t));
            st->vlWriteBlockIdDev = _e._ops->allocate(chunk * sizeof(std::uint32_t));
            st->vlWriteSlotDev    = _e._ops->allocate(chunk * sizeof(std::int32_t));
            st->vlSeqTDev      = _e._ops->allocate(sizeof(std::int32_t));
            st->vlSeqOffDev    = _e._ops->allocate(sizeof(std::int32_t));
            st->vlConvInOffDev = _e._ops->allocate(sizeof(std::int32_t));
            st->vlStartPosDev  = _e._ops->allocate(sizeof(std::int32_t));
            st->vlRopePosH.resize(chunk);
            st->vlWriteBlockIdH.resize(chunk);
            st->vlWriteSlotH.resize(chunk);
            st->vlStartPosH.resize(1);
            MM_LOG_INFO("serving",
                        "chunked prefill ENABLED (chunk={}) — prompts prefill "
                        "as T>1 forwards per slot", chunk);
        } else {
            MM_LOG_INFO("serving", "chunked prefill DISABLED "
                        "(MIMIRMIND_CHUNKED_PREFILL=0) — token-by-token prefill");
        }
    }

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
    if (_l0 != nullptr) {
        // L0 slab path: each slot's slab sits at its own length (its decode
        // position), so the neutral batched decode needs only the per-slot
        // input tokens over the contiguous active prefix.
        const std::size_t nSeq = steps.size();
        if (nSeq == 0) {
            return;
        }
        if (nSeq > _l0->maxBatch) {
            throw std::runtime_error("stepServing: nSeq exceeds serving maxBatch");
        }
        if (outTokens.size() != nSeq) {
            throw std::runtime_error("stepServing: outTokens size != steps size");
        }
        for (std::size_t i = 0; i < nSeq; ++i) {
            if (steps[i].slot != i) {
                throw std::runtime_error(
                    "stepServing: steps must be ordered by slot over a "
                    "contiguous prefix (steps[i].slot == i)");
            }
            _l0->stepTokens[i] = steps[i].token;
        }
        _l0->stepper->step(
            std::span<const std::int32_t>{_l0->stepTokens.data(), nSeq},
            outTokens, *_l0->decodeSb);
        return;
    }
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
    {
        std::int32_t mx = 0;
        for (std::size_t i = 0; i < nSeq; ++i) {
            mx = std::max(mx, st.seqLensH[i]);
        }
        ctx.maxSeqLen = mx;
    }
    ctx.startPosDev     = static_cast<const std::int32_t*>(st.startPosDev.get());
    ctx.expIdxSlot      = st.expIdxBuf.as<std::int32_t>();
    ctx.kwSlot          = st.kwBuf.as<float>();
    ctx.isSeqStart      = st.isSeqStart.data();

    for (std::size_t b = 0; b < st.blockCount; ++b) {
        st.qb->runBlockBatched(b, xBuf, ctx, *st.sb);
    }

    _e._ops->profileSection("lmhead");
    _e._ops->rmsNormAsync(xBuf, nSeq, st.d_model,
                          static_cast<const float*>(st.outNorm->usmPtr),
                          _e._config.rmsNormEps, normBuf);
    _e._gmm->matmul(st.lmHead->type, st.lmHead->usmPtr, st.vocab_lm, st.d_model,
                    normBuf, nSeq, logits, st.lmScr.as<float>());
    _e._ops->profileStepEnd();
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

std::int32_t ServingSession::prefillSlot(
        std::size_t                   slot,
        std::span<const std::int32_t> tokens,
        std::size_t                   startPos,
        bool                          produceToken) {
    namespace cmp = mimirmind::compute;
    if (_l0 != nullptr) {
        if (slot >= _l0->maxBatch) {
            throw std::runtime_error("prefillSlot: slot out of range");
        }
        // A new request (startPos 0) resets its slab; later chunks append at
        // the slab's current length (== startPos for sequential prefill).
        if (startPos == 0) {
            _l0->slab->resetSlot(slot);
        }
        return _l0->stepper->prefillSlot(slot, tokens, *_l0->prefillSb,
                                         produceToken);
    }
    if (_state == nullptr) {
        throw std::runtime_error("prefillSlot: ensureServingState not called");
    }
    auto& st = *_state;
    if (!st.chunkedPrefill || !st.sbPrefill.has_value()) {
        throw std::runtime_error("prefillSlot: chunked prefill not enabled");
    }
    const std::size_t T = tokens.size();
    if (T == 0) {
        return -1;
    }
    if (T > st.prefillChunk) {
        throw std::runtime_error("prefillSlot: chunk larger than prefillChunk");
    }
    if (slot >= st.maxBatch) {
        throw std::runtime_error("prefillSlot: slot out of range");
    }
    if (startPos + T > st.maxContext) {
        throw std::runtime_error("prefillSlot: startPos+T exceeds maxContext");
    }

    // 5.21-III Teil 4 — varlen prefill driver: prefill this chunk through the
    // RAGGED batched forward (runBlockBatched: paged KV + prefill-causal attention
    // + varlen GDN/conv) instead of the single-session runBlock. nSeq=1, seqT=[T].
    // Reuses sbPrefill (sized prefillChunk). Env-gated MIMIRMIND_PREFILL_VARLEN=1
    // for the batched-prefill-vs-single-session A/B (greedy token-match gate).
    static const bool varlenPrefill = [] {
        const char* e = std::getenv("MIMIRMIND_PREFILL_VARLEN");
        return e != nullptr && e[0] == '1' && e[1] == '\0';
    }();
    if (varlenPrefill) {
        const std::size_t d_model      = st.d_model;
        const std::size_t blocksPerSeq = st.blocksPerSeq;
        const std::size_t blockSize    = st.blockSize;
        for (std::size_t i = 0; i < T; ++i) {
            const std::size_t pos = startPos + i;
            st.vlRopePosH[i]      = static_cast<std::int32_t>(pos);
            st.vlWriteBlockIdH[i] = static_cast<std::uint32_t>(
                slot * blocksPerSeq + pos / blockSize);
            st.vlWriteSlotH[i]    = static_cast<std::int32_t>(pos % blockSize);
        }
        const std::int32_t seqTv = static_cast<std::int32_t>(T);
        const std::int32_t zero  = 0;
        st.vlStartPosH[0] = static_cast<std::int32_t>(startPos);
        _e._ops->uploadHostBytes(st.vlRopePosDev.get(),      st.vlRopePosH.data(),      T * sizeof(std::int32_t));
        _e._ops->uploadHostBytes(st.vlWriteBlockIdDev.get(), st.vlWriteBlockIdH.data(), T * sizeof(std::uint32_t));
        _e._ops->uploadHostBytes(st.vlWriteSlotDev.get(),    st.vlWriteSlotH.data(),    T * sizeof(std::int32_t));
        _e._ops->uploadHostBytes(st.vlSeqTDev.get(),      &seqTv, sizeof(std::int32_t));
        _e._ops->uploadHostBytes(st.vlSeqOffDev.get(),    &zero,  sizeof(std::int32_t));
        _e._ops->uploadHostBytes(st.vlConvInOffDev.get(), &zero,  sizeof(std::int32_t));
        _e._ops->uploadHostBytes(st.vlStartPosDev.get(),  st.vlStartPosH.data(), sizeof(std::int32_t));

        const std::size_t stateElems     = _e._config.ssmStateElemsPerLayer();
        const std::size_t convStateElems = _e._config.ssmConvStateElemsPerLayer();
        BlockBuffers& sb = *st.sbPrefill;
        sb.ssmStatePtr     = st.ssm->statePtr()     + slot * stateElems;
        sb.ssmConvStatePtr = st.ssm->convStatePtr() + slot * convStateElems;

        for (std::size_t i = 0; i < T; ++i) st.prefillTokH[i] = tokens[i];
        float* const xBuf = st.xBufP.as<float>();
        cmp::embeddingLookup(st.tokEmb->type, st.tokEmb->usmPtr, d_model,
                             st.vocab_emb,
                             std::span<const std::int32_t>{st.prefillTokH.data(), T}, xBuf);

        const std::uint8_t seqStartH = (startPos == 0) ? 1u : 0u;
        arch::BatchedDecodeCtx ctx{};
        ctx.nSeq            = 1;
        ctx.pool            = st.pool.get();
        ctx.writeBlockIdDev = static_cast<const std::uint32_t*>(st.vlWriteBlockIdDev.get());
        ctx.writeSlotDev    = static_cast<const std::int32_t*>(st.vlWriteSlotDev.get());
        ctx.blockTablesDev  = static_cast<const std::int32_t*>(st.blockTablesDev.get())
                              + slot * blocksPerSeq;
        ctx.maxBlocksPerSeq = blocksPerSeq;
        ctx.startPosDev     = static_cast<const std::int32_t*>(st.vlStartPosDev.get());
        ctx.expIdxSlot      = st.prefillExpIdx.as<std::int32_t>();
        ctx.kwSlot          = st.prefillKw.as<float>();
        ctx.isSeqStart      = &seqStartH;
        ctx.nRow            = T;
        ctx.seqTDev         = static_cast<const std::int32_t*>(st.vlSeqTDev.get());
        ctx.seqTHost        = &seqTv;
        ctx.seqOffDev       = static_cast<const std::int32_t*>(st.vlSeqOffDev.get());
        ctx.convInOffDev    = static_cast<const std::int32_t*>(st.vlConvInOffDev.get());
        ctx.ropePosDev      = static_cast<const std::int32_t*>(st.vlRopePosDev.get());
        ctx.maxSeqT         = T;

        st.qb->setPrefillMoeScratch(st.prefillExpIdx.as<std::int32_t>(),
                                    st.prefillKw.as<float>());
        for (std::size_t b = 0; b < st.blockCount; ++b) {
            st.qb->runBlockBatched(b, xBuf, ctx, sb);
        }
        st.qb->setPrefillMoeScratch(nullptr, nullptr);
        _e._ops->profileStepEnd();

        std::int32_t firstTok = -1;
        if (produceToken) {
            float* const normBuf = st.normB.as<float>();
            float* const logits  = st.logitsB.as<float>();
            _e._ops->rmsNormAsync(xBuf + (T - 1) * d_model, 1, d_model,
                                  static_cast<const float*>(st.outNorm->usmPtr),
                                  _e._config.rmsNormEps, normBuf);
            _e._gmm->matmul(st.lmHead->type, st.lmHead->usmPtr, st.vocab_lm, d_model,
                            normBuf, 1, logits, st.lmScr.as<float>());
            _e._ops->flush();
            _e._ops->readbackToHost(st.hostLogits.data(), logits,
                                    st.vocab_lm * sizeof(float));
            std::size_t best = 0; float bv = st.hostLogits[0];
            for (std::size_t v = 1; v < st.vocab_lm; ++v) {
                if (st.hostLogits[v] > bv) { bv = st.hostLogits[v]; best = v; }
            }
            firstTok = static_cast<std::int32_t>(best);
        } else {
            _e._ops->flush();
        }
        sb.ssmStatePtr     = st.ssm->statePtr();
        sb.ssmConvStatePtr = st.ssm->convStatePtr();
        return firstTok;
    }

    const std::size_t d_model  = st.d_model;
    const std::size_t nKvHeads = _e._config.headCountKv;
    const std::size_t headDim  = _e._config.headDim();
    const std::size_t kvDim    = nKvHeads * headDim;
    const std::size_t slotCap  = st.blocksPerSeq * st.blockSize;   // tokens
    const std::size_t slotElemBase = slot * slotCap * kvDim;       // elements

    // Non-owning per-block KvCache view over this slot's contiguous paged-KV
    // region. runBlock indexes the cache by blockIdx, so the view carries
    // blockCount layers: a full-attention block points at its dense pool
    // layer's slot region; a recurrent (GatedDeltaNet) block gets a valid
    // placeholder pointer (runLinearBlock never reads/writes cache K/V).
    std::vector<std::size_t> kvDimPerLayer(st.blockCount, kvDim);
    std::vector<void*>       kBases(st.blockCount, nullptr);
    std::vector<void*>       vBases(st.blockCount, nullptr);
    for (std::size_t b = 0; b < st.blockCount; ++b) {
        const std::size_t dense = st.qb->fullAttnDenseLayer(b);
        const std::size_t layer =
            (dense == std::numeric_limits<std::size_t>::max()) ? 0 : dense;
        // Pool base is void*; the per-slot offset is in ELEMENTS, so stride by
        // the pool's element width (fp32=4, fp16=2) to land on this slot's
        // region regardless of dtype (5.14 I1).
        const std::size_t byteOff = slotElemBase * st.pool->elemBytes();
        kBases[b] = static_cast<char*>(st.pool->keyPool(layer))   + byteOff;
        vBases[b] = static_cast<char*>(st.pool->valuePool(layer)) + byteOff;
    }
    KvCache view(KvCache::ExternalView{}, /*maxSeq=*/slotCap,
                 std::move(kvDimPerLayer), std::move(kBases), std::move(vBases),
                 /*initialLength=*/startPos, st.pool->dtype());

    // Bind the prefill BlockBuffers' SSM state base to this slot's slab slice.
    // runLinearBlock indexes `ssmStatePtr + blockIdx*(ssmSlabNSeq*elems)`, so a
    // base pre-offset by slot*elems (with ssmSlabNSeq == full slab width)
    // lands on exactly this slot's per-layer state — the same memory
    // stepServing evolves during decode. Kernels bake the pointer at enqueue,
    // so restoring the base afterwards is safe without a sync.
    const std::size_t stateElems     = _e._config.ssmStateElemsPerLayer();
    const std::size_t convStateElems = _e._config.ssmConvStateElemsPerLayer();
    BlockBuffers& sb = *st.sbPrefill;
    sb.ssmStatePtr     = st.ssm->statePtr()     + slot * stateElems;
    sb.ssmConvStatePtr = st.ssm->convStatePtr() + slot * convStateElems;

    // Embed the chunk tokens into the prefill hidden buffer.
    for (std::size_t i = 0; i < T; ++i) {
        st.prefillTokH[i] = tokens[i];
    }
    float* const xBuf = st.xBufP.as<float>();
    cmp::embeddingLookup(st.tokEmb->type, st.tokEmb->usmPtr, d_model,
                         st.vocab_emb,
                         std::span<const std::int32_t>{st.prefillTokH.data(), T},
                         xBuf);

    // One T>1 forward reusing the single-session block path. runBlock reads
    // cache.length() == startPos for the causal boundary / KV write offset and
    // targets this slot's SSM slab slice. runLinearBlock zeroes the recurrent
    // + conv state only when cache.length() == 0 (startPos == 0 == the
    // request's first chunk), so multi-chunk prefill carries state forward.
    // Route this prefill forward's routed-MoE through the amortised batched
    // fused-K path (one pass over the T tokens) instead of the per-token
    // runMoeFfn — the prefill TTFT lever. Cleared after the loop so
    // single-session paths keep the per-token MoE.
    st.qb->setPrefillMoeScratch(st.prefillExpIdx.as<std::int32_t>(),
                                st.prefillKw.as<float>());
    for (std::size_t b = 0; b < st.blockCount; ++b) {
        st.qb->runBlock(b, xBuf, T, view, sb, /*traceBlock0=*/false);
    }
    st.qb->setPrefillMoeScratch(nullptr, nullptr);
    // FP4-TC-peak bench: one profiler "step" per prefill chunk (DECODE_PROFILE
    // only). moe.tc + attn + ... sections print (see MIMIRMIND_PROFILE_EVERY) so
    // the wide-M FP4-TC MoE GEMM ms is visible for the TFLOP/s / %-peak calc.
    _e._ops->profileStepEnd();

    std::int32_t firstTok = -1;
    if (produceToken) {
        // lm-head over the final prompt row -> the first generated token,
        // identical to the token-by-token step that samples at pos ==
        // promptLen-1 (isGen). Intermediate chunks skip this.
        float* const normBuf = st.normB.as<float>();
        float* const logits  = st.logitsB.as<float>();
        _e._ops->rmsNormAsync(xBuf + (T - 1) * d_model, 1, d_model,
                              static_cast<const float*>(st.outNorm->usmPtr),
                              _e._config.rmsNormEps, normBuf);
        _e._gmm->matmul(st.lmHead->type, st.lmHead->usmPtr, st.vocab_lm, d_model,
                        normBuf, 1, logits, st.lmScr.as<float>());
        _e._ops->flush();
        _e._ops->readbackToHost(st.hostLogits.data(), logits,
                                st.vocab_lm * sizeof(float));
        std::size_t best = 0;
        float bv = st.hostLogits[0];
        for (std::size_t v = 1; v < st.vocab_lm; ++v) {
            if (st.hostLogits[v] > bv) { bv = st.hostLogits[v]; best = v; }
        }
        firstTok = static_cast<std::int32_t>(best);
    } else {
        // Intermediate chunk: no lm-head. Flush so this chunk's KV/state
        // writes are fully committed before the next (carry) chunk reads
        // them — matches the single-chunk path's implicit flush.
        _e._ops->flush();
    }

    // Restore the prefill SSM base to the slab origin so the next prefillSlot
    // (a different slot) re-offsets from a clean base.
    sb.ssmStatePtr     = st.ssm->statePtr();
    sb.ssmConvStatePtr = st.ssm->convStatePtr();
    return firstTok;
}

std::size_t ServingSession::prefillChunkSize() const noexcept {
    if (_l0 != nullptr) {
        return _l0->prefillChunk;
    }
    if (_state == nullptr || !_state->chunkedPrefill) {
        return 0;
    }
    return _state->prefillChunk;
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
    st.vArgmaxDev = _e._ops->allocate(Mcap * sizeof(std::int32_t));
    st.vLmScr   = _e._ops->allocate(std::max(d_model, vocab) * sizeof(float));

    // MV-d: one compact recurrent-state export [blockCount, Kp1, maxBatch,
    // stateElems] (the fused verify kernel writes each layer's slab, packed
    // with slot stride N ≤ maxBatch). Replaces the K+1 full-slab recurrent
    // snapshots. conv state still snapshotted per position into convSnap.
    const std::size_t stateElems = st.ssm->stateElemsPerLayer();
    const std::size_t exportElems =
        st.ssm->blockCount() * (K + 1) * st.maxBatch * stateElems;
    st.ssmExport = _e._ops->allocate(exportElems * sizeof(float));

    const std::size_t cvImgElems = st.ssm->blockCount() * st.ssm->convStateLayerStride();
    st.convSnap.clear();
    st.convSnap.reserve(K + 1);
    for (std::size_t j = 0; j <= K; ++j) {
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
    st.vArgmaxHost.assign(Mcap, 0);

    MM_LOG_INFO("serving",
                "ensureVerifyCapacity: depth={} Mcap={} exportStateBytes={} convSnaps={}",
                K, Mcap, exportElems * sizeof(float), K + 1);
}

std::size_t
ServingSession::verifyForward(
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
        return 0;
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

    // MV-d: per-position convSnap slab-image bases (Kp1); the recurrent state
    // is exported by the fused verify kernel into the single st.ssmExport.
    std::vector<float*> convSnapBases(Kp1);
    for (std::size_t j = 0; j < Kp1; ++j) {
        convSnapBases[j] = st.convSnap[j].as<float>();
    }

    for (std::size_t b = 0; b < st.blockCount; ++b) {
        if (_e._config.isRecurrentLayer(b)) {
            // MV-c/d: GatedDeltaNet verify — ONE batched layer over M=N*(K+1)
            // rows (proj/out-proj/MoE read each weight once vs K+1x), conv per
            // position, and ONE fused verify kernel for the recurrence that
            // exports every position's state into st.ssmExport.
            st.qb->runLinearBlockVerify(
                b, xBuf, N, Kp1, st.vExpIdx.as<std::int32_t>(),
                st.vKw.as<float>(), st.vGdnSeqStart.data(), st.maxBatch,
                st.ssmExport.as<float>(), convSnapBases.data(), *st.vsb);
        } else {
            st.qb->runBlockBatched(b, xBuf, ctxFull, *st.vsb);
        }
        // DFlash serving tap: stash the residual after each tapped block into
        // this tap's sink (all M time-major rows at once). CLR-safe device copy.
        if (st.dfTapActive) {
            const int k = st.dfTapSlot[b];
            if (k >= 0) {
                _e._ops->appendMemoryCopy(
                    st.dfTapSink[static_cast<std::size_t>(k)].get(), xBuf,
                    M * d_model * sizeof(float));
            }
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
    // MTP-verify breakdown: one profiler "step" per verify round (DECODE_PROFILE
    // only). Accumulated verify.proj/conv/gdn/tail + moe.gemm/attn/lmhead sections
    // print every 32 rounds -> shows whether the batched-verify cost is the GDN
    // recurrence (verify.gdn) or the MoE expert-union (moe.gemm at M=N*(K+1)).
    _e._ops->profileStepEnd();
    return M;
}

std::vector<std::vector<float>>
ServingSession::stepServingVerify(
        std::span<const InferenceEngine::VerifySlot> slots,
        std::span<const std::int32_t>                tokensTimeMajor,
        std::size_t                                  depth) {
    const std::size_t M = verifyForward(slots, tokensTimeMajor, depth);
    if (M == 0) {
        return {};
    }
    auto& st = *_state;
    const std::size_t vocab  = st.vocab_lm;
    float* const      logits = st.vLogitsB.as<float>();
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

std::vector<std::int32_t>
ServingSession::stepServingVerifyIds(
        std::span<const InferenceEngine::VerifySlot> slots,
        std::span<const std::int32_t>                tokensTimeMajor,
        std::size_t                                  depth) {
    const std::size_t M = verifyForward(slots, tokensTimeMajor, depth);
    if (M == 0) {
        return {};
    }
    auto& st = *_state;
    const std::size_t vocab  = st.vocab_lm;
    float* const      logits = st.vLogitsB.as<float>();
    // Device argmax over each of the M rows, then read back only the M ids
    // instead of the full M*vocab logits (tens of MB → a few bytes).
    _e._ops->argmaxRowsAsync(logits, st.vArgmaxDev.as<std::int32_t>(),
                             static_cast<int>(M), static_cast<int>(vocab));
    _e._ops->flush();
    _e._ops->readbackToHost(st.vArgmaxHost.data(), st.vArgmaxDev.get(),
                            M * sizeof(std::int32_t));
    return {st.vArgmaxHost.begin(),
            st.vArgmaxHost.begin() + static_cast<std::ptrdiff_t>(M)};
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
        // Device argmax over vocab -> read back only nSeq token ids (was the
        // full nSeq*vocab logits, ~19 MB/step). Lowest-index tie-break matches
        // the previous host scan, so the drafted ids are identical.
        _e._ops->argmaxRowsAsync(st.mtpDraftLogitsB.as<float>(),
                                 st.vArgmaxDev.as<std::int32_t>(), nSeq, vocab);
        _e._ops->flush();
        _e._ops->readbackToHost(st.vArgmaxHost.data(), st.vArgmaxDev.get(),
                                nSeq * sizeof(std::int32_t));
        for (std::size_t s = 0; s < nSeq; ++s) {
            const std::int32_t best = st.vArgmaxHost[s];
            drafts[s].push_back(best);
            st.mtpPrevTok[s] = best;
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
    auto* qb = dynamic_cast<arch::Qwen3_5MoeBackend*>(_e._backend.get());
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

void ServingSession::restoreSlotSsm(std::size_t slot, std::size_t a,
                                    std::size_t N) {
    auto& st = *_state;
    float* const       stDst = st.ssm->statePtr();
    float* const       cvDst = st.ssm->convStatePtr();
    const float* const exp   = st.ssmExport.as<float>();
    const float* const cvSrc = st.convSnap[a].as<float>();
    const std::size_t stStride = st.ssm->stateLayerStride();       // live slab
    const std::size_t cvStride = st.ssm->convStateLayerStride();
    const std::size_t stElems  = st.ssm->stateElemsPerLayer();
    const std::size_t cvElems  = st.ssm->convStateElemsPerLayer();
    const std::size_t Kp1        = st.verifyDepth + 1;
    const std::size_t expLStride = Kp1 * st.maxBatch * stElems;    // export slab
    for (std::size_t L = 0; L < st.blockCount; ++L) {
        if (!_e._config.isRecurrentLayer(L)) {
            continue;   // full-attention layers keep no recurrent state
        }
        _e._ops->appendMemoryCopy(stDst + L * stStride + slot * stElems,
                                  exp + L * expLStride + (a * N + slot) * stElems,
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
    auto* qb = dynamic_cast<arch::Qwen3_5MoeBackend*>(_e._backend.get());
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
        const auto vids = stepServingVerifyIds(slots, vtokTM, K);
        float* const vX = st.vXBuf.as<float>();   // M trunk hiddens (time-major)

        // --- per-slot accept-longest-prefix + snapshot restore -----------
        for (std::size_t s = 0; s < nSeq; ++s) {
            if (finished[s] != 0) continue;
            std::size_t a = 0;
            for (std::size_t i = 0; i < K; ++i) {
                if (vids[i * nSeq + s] == drafts[s][i]) ++a; else break;
            }
            const std::int32_t corrected = vids[a * nSeq + s];

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
            // The fused verify kernel does not advance the live recurrent
            // state, so commit from the export for EVERY accept a (0..K).
            restoreSlotSsm(s, a, nSeq);
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
ServingSession::generateBatchDflash(std::span<const std::int32_t> prompt,
                                    std::size_t nSeq, std::size_t maxNew,
                                    std::size_t depth, std::int32_t eosId,
                                    std::string_view drafterDir,
                                    std::size_t* draftedOut,
                                    std::size_t* acceptedOut) {
    if (_e._backend == nullptr) {
        throw std::runtime_error("generateBatchDflash: no model loaded");
    }
    auto* qb = dynamic_cast<arch::Qwen3_5MoeBackend*>(_e._backend.get());
    if (qb == nullptr) {
        throw std::runtime_error("generateBatchDflash: requires CUDA qwen35moe");
    }
    if (nSeq == 0)  nSeq = 1;
    if (depth == 0) depth = 1;
    if (prompt.empty() || maxNew == 0) {
        return std::vector<std::vector<std::int32_t>>(nSeq);
    }
    const std::size_t P = prompt.size();
    const std::size_t d = _e._config.embeddingLength;
    const std::size_t K = depth;

    // --- drafter (borrowed via the engine's lazy DFlashDecoder) ----------
    if (_e._dflashDecoder == nullptr) {
        _e._dflashDecoder = std::make_unique<DFlashDecoder>(_e);
    }
    DFlashDecoder& dfd = *_e._dflashDecoder;
    dfd.ensureDflashLoaded(drafterDir);
    const std::size_t taps = dfd.tapCount();
    const std::size_t rowC = dfd.ctxRowStride();      // taps * d

    ensureServingState(nSeq, P + maxNew + 8);
    ensureVerifyCapacity(K);
    auto& st = *_state;

    // --- arm the batched DFlash tap capture in verifyForward -------------
    st.dfTapSlot.assign(st.blockCount, -1);
    {
        const auto tl = dfd.tapLayers();
        for (std::size_t k = 0; k < tl.size(); ++k) {
            if (tl[k] < st.blockCount) st.dfTapSlot[tl[k]] = static_cast<int>(k);
        }
    }
    st.dfTapSink.clear();
    for (std::size_t k = 0; k < taps; ++k) {
        st.dfTapSink.push_back(
            _e._ops->allocate(st.verifyMcap * d * sizeof(float)));
    }
    st.dfTapActive = true;

    // --- shared prompt context + anchor token0 (single-session tap prefill)
    std::size_t Pc = 0;
    const std::int32_t token0v = dfd.buildPromptContext(prompt, Pc);
    const float* const promptCtx = dfd.ctxHidden();   // rows [0, P), stride rowC

    // Per-slot context accumulators, seeded with the (identical) prompt ctx.
    std::vector<compute::ComputeBuffer> ctxSlot;
    std::vector<float*>                 ctxPtr;
    ctxSlot.reserve(nSeq);
    ctxPtr.reserve(nSeq);
    for (std::size_t s = 0; s < nSeq; ++s) {
        ctxSlot.push_back(
            _e._ops->allocate(st.maxContext * rowC * sizeof(float)));
        ctxPtr.push_back(ctxSlot.back().as<float>());
        _e._ops->appendMemoryCopy(ctxPtr[s], promptCtx, P * rowC * sizeof(float));
    }
    _e._ops->flush();

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
    std::vector<std::size_t>               basePos(nSeq, P), ctxLen(nSeq, P);
    std::vector<std::int32_t>              token0(nSeq, token0v);
    std::vector<std::vector<std::int32_t>> out(nSeq);
    std::vector<char>                      finished(nSeq, 0);
    std::vector<std::vector<std::int32_t>> drafts(nSeq);
    std::size_t drafted = 0, accepted = 0;

    while (true) {
        std::size_t nDone = 0;
        for (std::size_t s = 0; s < nSeq; ++s) nDone += (finished[s] != 0);
        if (nDone == nSeq) break;
        if (out[0].size() >= maxNew) break;
        const std::size_t Kr = std::min(K, maxNew - out[0].size());

        // --- block-draft Kr tokens per slot (drafter is cheap; sequential) -
        for (std::size_t s = 0; s < nSeq; ++s) {
            if (finished[s] != 0) continue;
            dfd.draftOneBlock(ctxPtr[s], ctxLen[s], token0[s], Kr, drafts[s]);
            drafted += Kr;
        }

        // --- ONE batched verify over [token0, drafts...] per slot --------
        std::vector<InferenceEngine::VerifySlot> slots(nSeq);
        std::vector<std::int32_t> vtokTM((Kr + 1) * nSeq);
        for (std::size_t s = 0; s < nSeq; ++s) {
            slots[s].slot    = static_cast<std::uint32_t>(s);
            slots[s].basePos = static_cast<std::int32_t>(basePos[s]);
            vtokTM[0 * nSeq + s] = token0[s];
            for (std::size_t j = 1; j <= Kr; ++j) {
                vtokTM[j * nSeq + s] = drafts[s][j - 1];
            }
        }
        const auto vids = stepServingVerifyIds(slots, vtokTM, Kr);

        // --- per-slot accept-longest-prefix + commit (no re-forward) ------
        for (std::size_t s = 0; s < nSeq; ++s) {
            if (finished[s] != 0) continue;
            std::size_t a = 0;
            for (std::size_t i = 0; i < Kr; ++i) {
                if (vids[i * nSeq + s] == drafts[s][i]) ++a; else break;
            }
            const std::int32_t corrected = vids[a * nSeq + s];
            accepted += a;

            auto emit = [&](std::int32_t t) -> bool {
                out[s].push_back(t);
                if (eosId >= 0 && t == eosId) { finished[s] = 1; return false; }
                if (out[s].size() >= maxNew)  { finished[s] = 1; return false; }
                return true;
            };
            bool cont = emit(token0[s]);
            for (std::size_t i = 0; i < a && cont; ++i) cont = emit(drafts[s][i]);

            // Commit: KV for the accepted a+1 is already correct; restore the
            // accepted-prefix SSM from the per-timestep export (no re-forward).
            restoreSlotSsm(s, a, nSeq);
            // feedContext: append the accepted a+1 positions' 8 taps (captured
            // in dfTapSink, time-major row j*nSeq+s) into this slot's context.
            for (std::size_t j = 0; j <= a; ++j) {
                for (std::size_t k = 0; k < taps; ++k) {
                    _e._ops->appendMemoryCopy(
                        ctxPtr[s] + (ctxLen[s] + j) * rowC + k * d,
                        st.dfTapSink[k].as<float>() + (j * nSeq + s) * d,
                        d * sizeof(float));
                }
            }
            basePos[s] += a + 1;
            ctxLen[s]  += a + 1;
            token0[s]   = corrected;
        }
        _e._ops->flush();
    }

    st.dfTapActive = false;
    if (draftedOut)  *draftedOut  = drafted;
    if (acceptedOut) *acceptedOut = accepted;
    return out;
}

std::vector<std::vector<std::int32_t>>
ServingSession::generateBatchMtpMulti(
        const std::vector<std::vector<std::int32_t>>& prompts,
        std::size_t maxNew, std::size_t depth, std::int32_t eosId) {
    if (_e._backend == nullptr) {
        throw std::runtime_error("generateBatchMtpMulti: no model loaded");
    }
    auto* qb = dynamic_cast<arch::Qwen3_5MoeBackend*>(_e._backend.get());
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
        const auto vids = stepServingVerifyIds(slots, vtokTM, K);
        float* const vX = st.vXBuf.as<float>();

        for (std::size_t p = 0; p < N; ++p) {
            std::size_t a = 0;
            for (std::size_t i = 0; i < K; ++i) {
                if (vids[i * N + p] == drafts[p][i]) ++a; else break;
            }
            const std::int32_t corrected = vids[a * N + p];
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
            // Fused verify kernel leaves live state at S_0 → commit every a.
            restoreSlotSsm(p, a, N);
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
