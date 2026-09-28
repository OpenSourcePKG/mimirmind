// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "mimirmind/ServingBenchModes.hpp"

#include "core/config/Config.hpp"
#include "core/log/Log.hpp"
#include "model/Tokenizer.hpp"
#include "runtime/InferenceEngine.hpp"
#include "runtime/serving/ContinuousBatcher.hpp"
#ifdef MIMIRMIND_HAVE_L0
#include "compute/l0/GpuOps.hpp"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace mimirmind::cli {

std::optional<int> ServingBenchModes::maybeRun(
    runtime::InferenceEngine& engine,
    const std::string& arch,
    const core::config::Config& cfg) {
    if (arch == "qwen35moe" &&
        std::getenv("MIMIRMIND_BATCH_BENCH") != nullptr) {
        return runBatchBench(engine, arch, cfg);
    }
    if ((arch == "qwen35moe" || arch == "qwen4_exp") &&
        std::getenv("MIMIRMIND_SERVING_PARITY") != nullptr) {
        return runServingParity(engine, arch, cfg);
    }
    if (std::getenv("MIMIRMIND_L0_BATCH") != nullptr) {
        return runL0Batch(engine, arch, cfg);
    }
    if (arch == "qwen35moe" &&
        std::getenv("MIMIRMIND_SERVING_LOOP") != nullptr) {
        return runServingLoop(engine, arch, cfg);
    }
    if (arch == "qwen35moe" &&
        std::getenv("MIMIRMIND_BATCHER_TEST") != nullptr) {
        return runBatcherTest(engine, arch, cfg);
    }
    if (arch == "qwen35moe" &&
        std::getenv("MIMIRMIND_MTP_TEST") != nullptr) {
        return runMtpTest(engine, arch, cfg);
    }
    if (arch == "qwen35moe" &&
        std::getenv("MIMIRMIND_DFLASH_TEST") != nullptr) {
        return runDflashTest(engine, arch, cfg);
    }
    if (arch == "qwen35moe" &&
        std::getenv("MIMIRMIND_DFLASH_BATCH") != nullptr) {
        return runDflashBatch(engine, arch, cfg);
    }
    if (arch == "qwen35moe" &&
        std::getenv("MIMIRMIND_MTP_VERIFY_TEST") != nullptr) {
        return runMtpVerifyTest(engine, arch, cfg);
    }
    if (arch == "qwen35moe" &&
        std::getenv("MIMIRMIND_MTP_DRAFT_TEST") != nullptr) {
        return runMtpDraftTest(engine, arch, cfg);
    }
    if (arch == "qwen35moe" &&
        std::getenv("MIMIRMIND_MTP_BATCH_TEST") != nullptr) {
        return runMtpBatchTest(engine, arch, cfg);
    }
    if (arch == "qwen35moe" &&
        std::getenv("MIMIRMIND_MTP_ACCEPT") != nullptr) {
        return runMtpAccept(engine, arch, cfg);
    }
    if (arch == "qwen35moe" &&
        std::getenv("MIMIRMIND_MTP_PERF") != nullptr) {
        return runMtpPerf(engine, arch, cfg);
    }
    if (arch == "qwen35moe" &&
        std::getenv("MIMIRMIND_MTP_MULTI_TEST") != nullptr) {
        return runMtpMultiTest(engine, arch, cfg);
    }
    if (arch == "qwen35moe" &&
        std::getenv("MIMIRMIND_MTP_PERF_TEST") != nullptr) {
        return runMtpPerfTest(engine, arch, cfg);
    }
    return std::nullopt;
}


// M-Cuda.Batch D2e — batched decode throughput benchmark. With
// MIMIRMIND_BATCH_BENCH set, time generateBatch across batch sizes
// and report ms/step + generated-tokens/s, then exit. The batched
// forward processes all nSeq sequences in one pass, so gen-tok/s
// should scale with the batch while ms/step stays roughly flat —
// that is the serving-class throughput win. qwen35moe only.
int ServingBenchModes::runBatchBench(
    runtime::InferenceEngine& engine,
    [[maybe_unused]] const std::string& arch,
    [[maybe_unused]] const core::config::Config& cfg) {
    const auto& tok = engine.tokenizer();
    auto base = tok.encode("The capital of France is", /*addBos=*/false);
    if (base.empty()) base.push_back(1);
    const bool quick = std::getenv("MIMIRMIND_BENCH_QUICK") != nullptr;
    // MIMIRMIND_BENCH_NSEQ=N: profile-friendly mode — run ONLY nSeq=N
    // (skip the single-seq baseline and the sweep) with a small maxNew,
    // so an ncu run reaches the target regime with few prior kernels.
    const char* nseqEnv = std::getenv("MIMIRMIND_BENCH_NSEQ");
    // MIMIRMIND_BENCH_MAXNEW overrides the decode window (default 64
    // in nseq mode so decode dominates over prefill for a clean
    // gen-tok/s A/B); MIMIRMIND_BENCH_REPS repeats the sweep in-process
    // for a same-load median. nseqEnv accepts a comma-separated list.
    const char* maxNewEnv = std::getenv("MIMIRMIND_BENCH_MAXNEW");
    const std::size_t maxNew =
        quick ? 4
              : (maxNewEnv != nullptr
                     ? static_cast<std::size_t>(std::max<long>(
                           1, std::strtol(maxNewEnv, nullptr, 10)))
                     : (nseqEnv != nullptr ? 64 : 32));
    const char* repsEnv = std::getenv("MIMIRMIND_BENCH_REPS");
    const std::size_t reps = repsEnv != nullptr
        ? static_cast<std::size_t>(std::max<long>(1, std::strtol(repsEnv, nullptr, 10)))
        : 1;
    const std::size_t promptLen = base.size();
    std::vector<std::size_t> batchSizes;
    if (nseqEnv != nullptr) {
        const std::string spec(nseqEnv);
        std::size_t pos = 0;
        while (pos < spec.size()) {
            std::size_t comma = spec.find(',', pos);
            const std::string tok2 =
                spec.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
            if (!tok2.empty()) {
                batchSizes.push_back(static_cast<std::size_t>(
                    std::max<long>(1, std::strtol(tok2.c_str(), nullptr, 10))));
            }
            if (comma == std::string::npos) break;
            pos = comma + 1;
        }
        if (batchSizes.empty()) batchSizes.push_back(1);
    } else if (quick) {
        batchSizes = {1};
    } else {
        batchSizes = {1, 4, 8, 16};
    }
    std::cout << "\n[M-Cuda.Batch D2e bench] promptLen=" << promptLen
              << " maxNew=" << maxNew << "\n";
    // Single-session baseline on THIS box+model (apples-to-apples).
    if (nseqEnv == nullptr) {
        ::mimirmind::runtime::GenerateParams gpb{};
        gpb.maxNewTokens         = maxNew;
        gpb.sampling.temperature = 0.0F;
        engine.resetCache();
        const auto s0 = std::chrono::steady_clock::now();
        auto sref = engine.generate(base, gpb, {}, nullptr, {}, {});
        const auto s1 = std::chrono::steady_clock::now();
        const double sms =
            std::chrono::duration<double, std::milli>(s1 - s0).count();
        std::cout << "  single-seq generate(): " << sms << " ms  "
                  << (sms / static_cast<double>(sref.size()))
                  << " ms/tok  " << (1000.0 * static_cast<double>(sref.size()) / sms)
                  << " tok/s\n";
        std::cout.flush();
    }
    for (std::size_t nSeq : batchSizes) {
        std::vector<std::vector<std::int32_t>> prompts(nSeq, base);
        (void)engine.generateBatch(prompts, 4, -1);   // warm up (all nSeq)
        for (std::size_t rep = 0; rep < reps; ++rep) {
            const auto t0 = std::chrono::steady_clock::now();
            const auto benchOut = engine.generateBatch(prompts, maxNew, /*eosId=*/-1);
            const auto t1 = std::chrono::steady_clock::now();
            const double ms =
                std::chrono::duration<double, std::milli>(t1 - t0).count();
            const std::size_t steps   = promptLen + maxNew;
            const std::size_t genToks = nSeq * maxNew;
            std::cout << "  nSeq=" << nSeq << "  rep=" << rep
                      << "  total=" << ms << " ms  "
                      << (ms / static_cast<double>(steps)) << " ms/step  "
                      << (1000.0 * static_cast<double>(genToks) / ms)
                      << " gen-tok/s\n";
            // Parity probe: first output token ids (compare graph vs not).
            if (!benchOut.empty()) {
                std::cout << "  out[0] ids:";
                for (std::size_t ti = 0;
                     ti < benchOut[0].size() && ti < 12; ++ti) {
                    std::cout << " " << benchOut[0][ti];
                }
                std::cout << "\n";
            }
            std::cout.flush();
        }
    }
    return 0;
}


// M-Cuda.Batch D2d — batched serving parity gate (dev/CI hook).
// With MIMIRMIND_SERVING_PARITY set, run the batched decode path
// (generateServingParity) against single-seq greedy generate() on
// this freshly-loaded model, print the comparison, and exit before
// the HTTP server starts. qwen35moe only.
int ServingBenchModes::runServingParity(
    runtime::InferenceEngine& engine,
    [[maybe_unused]] const std::string& arch,
    [[maybe_unused]] const core::config::Config& cfg) {
    const auto& tok = engine.tokenizer();
    std::vector<std::int32_t> promptIds =
        tok.encode("The capital of France is", /*addBos=*/false);
    if (promptIds.empty()) {
        promptIds.push_back(1);
    }
    // MIMIRMIND_BATCH_NP=N truncates the prompt to N tokens. With a
    // 1-token prompt the single-session reference does a T=1 prefill,
    // i.e. the same per-token forward the batched path feeds — an
    // apples-to-apples parity gate. Larger N makes the single side do
    // a T=N prefill (a numerically distinct kernel path from N* T=1
    // decode), so the comparison then also reflects prefill-vs-feed.
    if (const char* np = std::getenv("MIMIRMIND_BATCH_NP")) {
        const long n = std::strtol(np, nullptr, 10);
        if (n > 0 && static_cast<std::size_t>(n) < promptIds.size()) {
            promptIds.resize(static_cast<std::size_t>(n));
        }
    }
    // MIMIRMIND_PARITY_MAXNEW=N lengthens the greedy comparison — used
    // as the BF16-TC default-on coherence gate (long generations across
    // diverse prompts must track the trusted scalar single-session path).
    std::size_t maxNew = 8;
    if (const char* mn = std::getenv("MIMIRMIND_PARITY_MAXNEW")) {
        const long v = std::strtol(mn, nullptr, 10);
        if (v > 0) maxNew = static_cast<std::size_t>(v);
    }
    // 5.27.11.1: conc configurable — qwen4_exp conc=1 parity uses NSEQ=1
    // (per-slot PLE for nSeq>1 is 5.27.11.2).
    std::size_t nSeq = 2;
    if (const char* ns = std::getenv("MIMIRMIND_PARITY_NSEQ")) {
        const long v = std::strtol(ns, nullptr, 10);
        if (v > 0) nSeq = static_cast<std::size_t>(v);
    }

    ::mimirmind::runtime::GenerateParams gp{};
    gp.maxNewTokens         = maxNew;
    gp.sampling.temperature = 0.0F;   // greedy argmax
    // Reset the KV+SSM state so the reference is a clean single-session
    // run — the prefix cache reuses KV across generate() calls but does
    // NOT restore the GatedDeltaNet recurrent state, which would
    // contaminate a later reference (lcp>0 keeps a stale SsmState).
    engine.resetCache();
    std::vector<std::int32_t> ref =
        engine.generate(promptIds, gp, {}, nullptr, {}, {});

    auto batched = engine.generateServingParity(promptIds, nSeq, maxNew);

    bool allSeqEqual = true;
    for (std::size_t s = 1; s < nSeq; ++s) {
        if (batched[s] != batched[0]) allSeqEqual = false;
    }
    std::size_t matchLen = 0;
    const std::size_t cmpN = std::min(batched[0].size(), ref.size());
    for (; matchLen < cmpN; ++matchLen) {
        if (batched[0][matchLen] != ref[matchLen]) break;
    }
    std::cout << "\n[M-Cuda.Batch D2d serving-parity] nSeq=" << nSeq
              << " maxNew=" << maxNew
              << " promptTokens=" << promptIds.size() << "\n"
              << "  batched[0] :";
    for (auto t : batched[0]) std::cout << ' ' << t;
    std::cout << "\n  single-seq :";
    for (auto t : ref) std::cout << ' ' << t;
    std::cout << "\n  all-seq-identical=" << (allSeqEqual ? "YES" : "NO")
              << "  ref-match-prefix=" << matchLen << "/" << ref.size()
              << ((matchLen == ref.size() && allSeqEqual)
                      ? "  => PASS"
                      : "  => CHECK")
              << "\n";

    // D2e.1 — generateBatch with DISTINCT prompts. Each batched
    // stream must equal its own single-session greedy generate().
    // 5.27.11.2: conc>1 per-slot PLE — qwen4_exp runs this too with each
    // prompt truncated to 1 token (distinct first tokens -> distinct
    // per-slot n-gram context; single-session also T=1 = apples-to-apples).
    if (arch == "qwen35moe" || arch == "qwen4_exp") {
    const char* multiPrompts[] = {
        "The capital of France is",
        "Once upon a time",
        "2 plus 2 equals",
        "Explain how a transformer neural network works:",
        "def quicksort(arr):",
        "The three laws of thermodynamics are",
        "Translate to German: The weather is nice today.",
        "List the planets of the solar system in order:",
    };
    std::vector<std::vector<std::int32_t>> bprompts;
    for (const char* mp : multiPrompts) {
        auto ids = tok.encode(mp, /*addBos=*/false);
        if (ids.empty()) ids.push_back(1);
        // 5.27.11.2: qwen4_exp — 1-token prompts isolate per-slot PLE
        // (distinct first tokens) from prefill-vs-decode numerics.
        if (arch == "qwen4_exp") ids.resize(1);
        bprompts.push_back(std::move(ids));
    }
    auto bout = engine.generateBatch(bprompts, maxNew, /*eosId=*/-1);
    std::cout << "[M-Cuda.Batch D2e generateBatch] "
              << bprompts.size() << " distinct prompts, maxNew="
              << maxNew << "\n";
    bool allBatchOk = true;
    for (std::size_t i = 0; i < bprompts.size(); ++i) {
        ::mimirmind::runtime::GenerateParams gpi{};
        gpi.maxNewTokens         = maxNew;
        gpi.sampling.temperature = 0.0F;
        engine.resetCache();   // clean single-session reference (see above)
        std::vector<std::int32_t> refi =
            engine.generate(bprompts[i], gpi, {}, nullptr, {}, {});
        std::size_t ml = 0;
        const std::size_t cn = std::min(bout[i].size(), refi.size());
        for (; ml < cn; ++ml) {
            if (bout[i][ml] != refi[ml]) break;
        }
        const bool ok = (ml == refi.size() && bout[i].size() == refi.size());
        allBatchOk = allBatchOk && ok;
        std::cout << "  prompt[" << i << "] match=" << ml << "/"
                  << refi.size() << (ok ? " OK" : " MISMATCH")
                  << "  batched:";
        for (auto t : bout[i]) std::cout << ' ' << t;
        std::cout << "\n";
    }
    std::cout << "  => generateBatch "
              << (allBatchOk ? "PASS" : "CHECK") << "\n";
    }  // end qwen35moe-only D2e distinct-prompt block (5.27.11.1)
    std::cout.flush();
    return 0;
}


// M-L0.Batch Phase 1 — L0 synchronized batched decode parity + perf.
// With MIMIRMIND_L0_BATCH set, run the Gemma 4 MoE batched decode
// path (generateBatchL0): first a greedy parity gate (nSeq identical
// prompts must produce identical streams AND match single-seq greedy
// generate()), then a throughput sweep over nSeq. Prints and exits
// before the HTTP server starts. Xe-LPG / Gemma 4 MoE only.
int ServingBenchModes::runL0Batch(
    runtime::InferenceEngine& engine,
    [[maybe_unused]] const std::string& arch,
    [[maybe_unused]] const core::config::Config& cfg) {
    const auto& tok = engine.tokenizer();
    auto base = tok.encode("The capital of France is", /*addBos=*/false);
    if (base.empty()) base.push_back(1);
    if (const char* np = std::getenv("MIMIRMIND_BATCH_NP")) {
        const long n = std::strtol(np, nullptr, 10);
        if (n > 0 && static_cast<std::size_t>(n) < base.size()) {
            base.resize(static_cast<std::size_t>(n));
        }
    }
    const bool quick = std::getenv("MIMIRMIND_BENCH_QUICK") != nullptr;
    const std::size_t maxNew    = quick ? 4 : 24;
    const std::size_t promptLen = base.size();

    try {
        // --- Q6_K prefill probe: a long prompt drives the block
        // matmuls at M=promptLen (>=16) so the Q6_K GEMM engages
        // (gemmMinM=16). Run under gemm-on vs gemm-disable configs
        // to A/B the prefill time; the greedy output must be
        // identical across configs (GEMM numerically correct in
        // situ). onPrefillDone isolates the prefill wall-time.
        {
            std::vector<std::int32_t> longP;
            while (longP.size() < 96) {
                longP.insert(longP.end(), base.begin(), base.end());
            }
            ::mimirmind::runtime::GenerateParams gpp{};
            gpp.maxNewTokens         = 8;
            gpp.sampling.temperature = 0.0F;
            double prefillMs = 0.0;
            auto onPre = [&](const auto& d) { prefillMs = d.prefillMs; };
            engine.resetCache();
            auto pout = engine.generate(longP, gpp, {}, nullptr, onPre, {});
            std::cout << "\n[M-L0.Batch prefill] promptLen=" << longP.size()
                      << " prefill_ms=" << prefillMs << " gen8=";
            for (auto t : pout) std::cout << ' ' << t;
            std::cout << "\n";
            std::cout.flush();
        }

        // --- (1c) greedy parity: batched nSeq in {1,2,4} vs single ---
        // nSeq=1 isolates the batched-attention MATH (M=1 -> same
        // vec kernels as single-seq) from the M>1 GEMM-vs-GEMV
        // numeric path. If nSeq=1 matches single-seq but nSeq>1
        // diverges, the divergence is kernel numerics breaking a
        // near-tie greedy pick, not a batched-attention bug.
        ::mimirmind::runtime::GenerateParams gp{};
        gp.maxNewTokens         = maxNew;
        gp.sampling.temperature = 0.0F;   // greedy argmax
        engine.resetCache();
        const std::vector<std::int32_t> ref =
            engine.generate(base, gp, {}, nullptr, {}, {});
        std::cout << "\n[M-L0.Batch parity] maxNew=" << maxNew
                  << " promptTokens=" << promptLen << "\n  single-seq :";
        for (auto t : ref) std::cout << ' ' << t;
        std::cout << "\n";

        const std::vector<std::size_t> parityN =
            quick ? std::vector<std::size_t>{1, 2}
                  : std::vector<std::size_t>{1, 2, 4};
        for (std::size_t nSeq : parityN) {
            std::vector<std::vector<std::int32_t>> pprompts(nSeq, base);
            auto batched =
                engine.generateBatchL0(pprompts, maxNew, /*eosId=*/-1);
            bool allSeqEqual = true;
            for (std::size_t s = 1; s < nSeq; ++s) {
                if (batched[s] != batched[0]) allSeqEqual = false;
            }
            std::size_t matchLen = 0;
            const std::size_t cmpN =
                std::min(batched[0].size(), ref.size());
            for (; matchLen < cmpN; ++matchLen) {
                if (batched[0][matchLen] != ref[matchLen]) break;
            }
            std::cout << "  nSeq=" << nSeq
                      << "  all-seq-identical="
                      << (allSeqEqual ? "YES" : "NO")
                      << "  ref-match=" << matchLen << "/" << ref.size()
                      << ((matchLen == ref.size() && allSeqEqual)
                              ? "  PASS" : "  CHECK")
                      << "  batched[0]:";
            for (auto t : batched[0]) std::cout << ' ' << t;
            std::cout << "\n";
        }

        // --- (1d) throughput sweep over nSeq ------------------------
        std::cout << "[M-L0.Batch bench] promptLen=" << promptLen
                  << " maxNew=" << maxNew << "\n";
        {
            engine.resetCache();
            const auto s0 = std::chrono::steady_clock::now();
            auto sref = engine.generate(base, gp, {}, nullptr, {}, {});
            const auto s1 = std::chrono::steady_clock::now();
            const double sms =
                std::chrono::duration<double, std::milli>(s1 - s0).count();
            std::cout << "  single-seq generate(): " << sms << " ms  "
                      << (sms / static_cast<double>(sref.size()))
                      << " ms/tok  "
                      << (1000.0 * static_cast<double>(sref.size()) / sms)
                      << " tok/s\n";
            std::cout.flush();
        }
        const std::vector<std::size_t> batchSizes =
            quick ? std::vector<std::size_t>{1}
                  : std::vector<std::size_t>{1, 2, 4, 8, 16, 32};
        for (std::size_t nSeq : batchSizes) {
            std::vector<std::vector<std::int32_t>> prompts(nSeq, base);
            (void)engine.generateBatchL0(prompts, quick ? 2 : 4, -1); // warm
            const auto t0 = std::chrono::steady_clock::now();
            (void)engine.generateBatchL0(prompts, maxNew, /*eosId=*/-1);
            const auto t1 = std::chrono::steady_clock::now();
            const double ms =
                std::chrono::duration<double, std::milli>(t1 - t0).count();
            const std::size_t steps   = promptLen + maxNew;
            const std::size_t genToks = nSeq * maxNew;
            std::cout << "  nSeq=" << nSeq
                      << "  total=" << ms << " ms  "
                      << (ms / static_cast<double>(steps)) << " ms/step  "
                      << (1000.0 * static_cast<double>(genToks) / ms)
                      << " gen-tok/s\n";
            std::cout.flush();
        }
    } catch (const std::exception& ex) {
        std::cout << "\n[M-L0.Batch] unavailable for this model: "
                  << ex.what() << "\n";
    }
    return 0;
}


// M-Cuda.Batch D2e.2 — continuous-batching STEP-loop validation.
// With MIMIRMIND_SERVING_LOOP set, drive the persistent per-slot
// stepServing() interface as a hand-rolled continuous batcher:
// admit 3 distinct prompts at STAGGERED iterations (each pinned to
// its own slot, stepping at its OWN position), collect each stream,
// and compare to single-seq greedy generate(). This is the engine
// primitive the ContinuousBatcher wraps with a worker thread + HTTP.
int ServingBenchModes::runServingLoop(
    runtime::InferenceEngine& engine,
    [[maybe_unused]] const std::string& arch,
    [[maybe_unused]] const core::config::Config& cfg) {
    const auto& tok = engine.tokenizer();
    const char* prompts[] = {
        "The capital of France is",
        "Once upon a time",
        "2 plus 2 equals",
    };
    const std::size_t nReq   = 3;
    const std::size_t maxNew  = 8;
    const std::size_t admitAt[nReq] = {0, 2, 4};   // staggered arrival

    std::vector<std::vector<std::int32_t>> pids(nReq);
    std::size_t maxLen = 0;
    for (std::size_t r = 0; r < nReq; ++r) {
        pids[r] = tok.encode(prompts[r], /*addBos=*/false);
        if (pids[r].empty()) pids[r].push_back(1);
        maxLen = std::max(maxLen, pids[r].size());
    }
    const std::size_t maxContext = maxLen + maxNew + 8;
    engine.ensureServingState(/*maxBatch=*/nReq, maxContext);

    struct Req {
        std::size_t promptLen{0};
        std::size_t pos{0};
        std::int32_t lastTok{0};
        bool admitted{false};
        bool done{false};
        std::vector<std::int32_t> out;
    };
    std::vector<Req> req(nReq);
    for (std::size_t r = 0; r < nReq; ++r) req[r].promptLen = pids[r].size();

    // Slot r == request r (pinned): admit in request order so the
    // active set is always the contiguous prefix [0, nAdmitted).
    std::size_t nAdmitted = 0;
    using Step = ::mimirmind::runtime::InferenceEngine::ServingSlotStep;
    for (std::size_t g = 0; g < maxLen + maxNew + nReq * 4; ++g) {
        for (std::size_t r = 0; r < nReq; ++r) {
            if (!req[r].admitted && admitAt[r] <= g) {
                req[r].admitted = true;
                nAdmitted = std::max(nAdmitted, r + 1);
            }
        }
        if (nAdmitted == 0) continue;

        std::vector<Step> steps(nAdmitted);
        for (std::size_t i = 0; i < nAdmitted; ++i) {
            Step s{};
            s.slot = static_cast<std::uint32_t>(i);
            if (!req[i].admitted || req[i].done) {
                // Idle slot: fresh 1-token dummy (output discarded).
                s.token = 0; s.pos = 0; s.seqStart = true;
            } else {
                const std::size_t p = req[i].pos;
                s.token = (p < req[i].promptLen) ? pids[i][p]
                                                 : req[i].lastTok;
                s.pos      = static_cast<std::int32_t>(p);
                s.seqStart = (p == 0);
            }
            steps[i] = s;
        }
        std::vector<std::int32_t> toks(nAdmitted, 0);
        engine.stepServing(steps, toks);

        bool allDone = true;
        for (std::size_t i = 0; i < nAdmitted; ++i) {
            if (!req[i].admitted || req[i].done) continue;
            if (req[i].pos + 1 >= req[i].promptLen) {
                req[i].out.push_back(toks[i]);
                req[i].lastTok = toks[i];
                if (req[i].out.size() >= maxNew) req[i].done = true;
            }
            req[i].pos++;
            if (!req[i].done) allDone = false;
        }
        bool allAdmitted = true;
        for (std::size_t r = 0; r < nReq; ++r)
            if (!req[r].admitted) allAdmitted = false;
        if (allAdmitted && allDone) break;
    }

    std::cout << "\n[M-Cuda.Batch D2e.2 serving-loop] " << nReq
              << " staggered requests (admitAt 0/2/4), maxNew="
              << maxNew << "\n";
    bool allOk = true;
    for (std::size_t r = 0; r < nReq; ++r) {
        ::mimirmind::runtime::GenerateParams gpr{};
        gpr.maxNewTokens         = maxNew;
        gpr.sampling.temperature = 0.0F;
        engine.resetCache();
        std::vector<std::int32_t> ref =
            engine.generate(pids[r], gpr, {}, nullptr, {}, {});
        std::size_t ml = 0;
        const std::size_t cn = std::min(req[r].out.size(), ref.size());
        for (; ml < cn; ++ml) if (req[r].out[ml] != ref[ml]) break;
        const bool ok = (ml == ref.size() &&
                         req[r].out.size() == ref.size());
        allOk = allOk && ok;
        std::cout << "  req[" << r << "] admitAt=" << admitAt[r]
                  << " match=" << ml << "/" << ref.size()
                  << (ok ? " OK" : " MISMATCH") << "  loop:";
        for (auto t : req[r].out) std::cout << ' ' << t;
        std::cout << "\n";
    }
    std::cout << "  => serving-loop " << (allOk ? "PASS" : "CHECK")
              << "\n";
    std::cout.flush();
    return 0;
}


// M-Cuda.Batch D2e.2 — threaded ContinuousBatcher end-to-end test.
// With MIMIRMIND_BATCHER_TEST set, submit MORE requests than there
// are slots so later ones must reuse freed slots, then verify each
// stream == its single-seq greedy generate(). Exercises the worker
// thread + admit/complete/slot-reuse path the HTTP server uses.
int ServingBenchModes::runBatcherTest(
    runtime::InferenceEngine& engine,
    [[maybe_unused]] const std::string& arch,
    [[maybe_unused]] const core::config::Config& cfg) {
    const auto& tok = engine.tokenizer();
    // Include LONG prompts (>16 tokens => cross paged-KV block
    // boundaries, blockSize=16) so the block-table walk in the
    // batched paged path is exercised, plus short ones for slot
    // reuse. Each stream must still equal single-seq generate().
    const char* prompts[] = {
        "The capital of France is",
        "Once upon a time in a small village nestled deep between two "
        "great mountains, there lived an old clockmaker who believed "
        "that every second carried a secret worth keeping, and so he",
        "2 plus 2 equals",
        "Write a detailed explanation of how photosynthesis converts "
        "sunlight, water and carbon dioxide into glucose and oxygen "
        "inside the chloroplasts of a green plant leaf, step by step:",
        "In the beginning",
    };
    const std::size_t nReq  = 5;
    const std::size_t maxNew = 12;
    std::vector<std::vector<std::int32_t>> pids(nReq);
    std::size_t maxLen = 0;
    for (std::size_t r = 0; r < nReq; ++r) {
        pids[r] = tok.encode(prompts[r], /*addBos=*/false);
        if (pids[r].empty()) pids[r].push_back(1);
        maxLen = std::max(maxLen, pids[r].size());
    }
    // References first (before the batcher builds serving state), so
    // the single-session KV/SSM path is untouched by the batcher.
    std::vector<std::vector<std::int32_t>> refs(nReq);
    for (std::size_t r = 0; r < nReq; ++r) {
        ::mimirmind::runtime::GenerateParams gpr{};
        gpr.maxNewTokens         = maxNew;
        gpr.sampling.temperature = 0.0F;
        engine.resetCache();
        refs[r] = engine.generate(pids[r], gpr, {}, nullptr, {}, {});
    }

    const std::size_t maxBatch   = 3;   // < nReq => forces slot reuse
    const std::size_t maxContext = maxLen + maxNew + 8;
    ::mimirmind::runtime::serving::ContinuousBatcher batcher(
        engine, maxBatch, maxContext, /*eosId=*/-1,
        /*maxInflight=*/nReq);   // accept all; validate queueing/reuse

    std::vector<std::shared_ptr<
        ::mimirmind::runtime::serving::ServingRequest>> handles(nReq);
    for (std::size_t r = 0; r < nReq; ++r) {
        handles[r] = batcher.submit(pids[r], maxNew, {});
    }
    std::cout << "\n[M-Cuda.Batch D2e.2 batcher-test] " << nReq
              << " requests, maxBatch=" << maxBatch
              << " (slot reuse), maxNew=" << maxNew << "\n";
    bool allOk = true;
    for (std::size_t r = 0; r < nReq; ++r) {
        std::vector<std::int32_t> out = handles[r]->waitAll();
        std::size_t ml = 0;
        const std::size_t cn = std::min(out.size(), refs[r].size());
        for (; ml < cn; ++ml) if (out[ml] != refs[r][ml]) break;
        const bool ok = (ml == refs[r].size() &&
                         out.size() == refs[r].size() &&
                         handles[r]->error.empty());
        allOk = allOk && ok;
        std::cout << "  req[" << r << "] promptLen=" << pids[r].size()
                  << " match=" << ml << "/"
                  << refs[r].size() << (ok ? " OK" : " MISMATCH");
        if (!handles[r]->error.empty())
            std::cout << " err=" << handles[r]->error;
        std::cout << "  out:";
        for (auto t : out) std::cout << ' ' << t;
        std::cout << "\n";
    }
    std::cout << "  => batcher-test " << (allOk ? "PASS" : "CHECK")
              << "\n";
    std::cout.flush();
    return 0;
}


// M-Cuda.MTP — native multi-token-prediction validation + speedup.
// MIMIRMIND_MTP_TEST: run generateMtp (depth from MIMIRMIND_MTP_DEPTH,
// default 2) vs plain greedy generate() on the same prompt. Output MUST
// be bit-identical (verify guarantees correctness); report accept-rate
// and decode speedup.
int ServingBenchModes::runMtpTest(
    runtime::InferenceEngine& engine,
    [[maybe_unused]] const std::string& arch,
    [[maybe_unused]] const core::config::Config& cfg) {
    if (!engine.mtpAvailable()) {
        std::cout << "\n[M-Cuda.MTP] model has no nextn head — skipped\n";
        std::cout.flush();
        return 0;
    }
    const auto& tok = engine.tokenizer();
    const char* promptEnv = std::getenv("MIMIRMIND_MTP_PROMPT");
    std::vector<std::int32_t> pids = tok.encode(
        promptEnv != nullptr
            ? promptEnv
            : "The history of artificial intelligence began when",
        /*addBos=*/false);
    if (pids.empty()) pids.push_back(1);
    std::size_t depth = 2;
    if (const char* dv = std::getenv("MIMIRMIND_MTP_DEPTH")) {
        const long v = std::strtol(dv, nullptr, 10);
        if (v >= 1) depth = static_cast<std::size_t>(v);
    }
    std::size_t maxNew = 64;
    if (const char* mv = std::getenv("MIMIRMIND_MTP_MAXNEW")) {
        const long v = std::strtol(mv, nullptr, 10);
        if (v >= 1) maxNew = static_cast<std::size_t>(v);
    }
    std::cout << "\n[M-Cuda.MTP] promptTokens=" << pids.size();
    using clk = std::chrono::steady_clock;

    // Baseline greedy generate().
    ::mimirmind::runtime::GenerateParams gp{};
    gp.maxNewTokens         = maxNew;
    gp.sampling.temperature = 0.0F;
    engine.resetCache();
    const auto tb0 = clk::now();
    std::vector<std::int32_t> ref =
        engine.generate(pids, gp, {}, nullptr, {}, {});
    const double baseMs =
        std::chrono::duration<double, std::milli>(clk::now() - tb0).count();

    // MTP greedy generate.
    std::size_t drafted = 0, accepted = 0;
    const auto tm0 = clk::now();
    std::vector<std::int32_t> mtp =
        engine.generateMtp(pids, maxNew, depth, tok.eosId(),
                       &drafted, &accepted);
    const double mtpMs =
        std::chrono::duration<double, std::milli>(clk::now() - tm0).count();

    std::size_t matchLen = 0;
    const std::size_t cn = std::min(ref.size(), mtp.size());
    for (; matchLen < cn; ++matchLen)
        if (ref[matchLen] != mtp[matchLen]) break;
    const bool identical =
        (ref.size() == mtp.size()) && (matchLen == ref.size());
    const double acceptRate =
        drafted > 0 ? static_cast<double>(accepted) / drafted : 0.0;

    std::cout << "\n[M-Cuda.MTP] depth=" << depth << " maxNew=" << maxNew
              << "\n  output-identical=" << (identical ? "YES" : "NO")
              << " (match " << matchLen << "/" << ref.size() << ", mtp "
              << mtp.size() << ")\n"
              << "  accept-rate=" << acceptRate << " (" << accepted << "/"
              << drafted << ")\n"
              << "  baseline=" << baseMs << " ms  mtp=" << mtpMs
              << " ms  speedup=" << (mtpMs > 0 ? baseMs / mtpMs : 0.0)
              << "x\n"
              << "  => MTP " << (identical ? "PASS" : "MISMATCH") << "\n";
    std::cout.flush();
    return 0;
}


// M-Cuda.DFlash Phase 3.4 — DFlash block-diffusion draft validation.
// MIMIRMIND_DFLASH_TEST: run generateDflash (block size from
// MIMIRMIND_DFLASH_N, default 7 => block_size 8) vs plain greedy
// generate() on the same prompt. Output MUST be bit-identical (verify
// guarantees correctness); report mean accept-length + decode speedup.
// The drafter checkpoint dir comes from MIMIRMIND_DFLASH_DIR.
int ServingBenchModes::runDflashTest(
    runtime::InferenceEngine& engine,
    [[maybe_unused]] const std::string& arch,
    [[maybe_unused]] const core::config::Config& cfg) {
    if (!engine.dflashAvailable()) {
        std::cout << "\n[M-Cuda.DFlash] target not DFlash-capable — skipped\n";
        std::cout.flush();
        return 0;
    }
    const char* dirEnv = std::getenv("MIMIRMIND_DFLASH_DIR");
    if (dirEnv == nullptr) {
        std::cout << "\n[M-Cuda.DFlash] set MIMIRMIND_DFLASH_DIR to the "
                     "drafter checkpoint dir — skipped\n";
        std::cout.flush();
        return 0;
    }
    const auto& tok = engine.tokenizer();
    const char* promptEnv = std::getenv("MIMIRMIND_DFLASH_PROMPT");
    std::vector<std::int32_t> pids = tok.encode(
        promptEnv != nullptr
            ? promptEnv
            : "The history of artificial intelligence began when",
        /*addBos=*/false);
    if (pids.empty()) pids.push_back(1);
    std::size_t draftN = 7;
    if (const char* dv = std::getenv("MIMIRMIND_DFLASH_N")) {
        const long v = std::strtol(dv, nullptr, 10);
        if (v >= 1) draftN = static_cast<std::size_t>(v);
    }
    std::size_t maxNew = 128;
    if (const char* mv = std::getenv("MIMIRMIND_DFLASH_MAXNEW")) {
        const long v = std::strtol(mv, nullptr, 10);
        if (v >= 1) maxNew = static_cast<std::size_t>(v);
    }
    std::cout << "\n[M-Cuda.DFlash] promptTokens=" << pids.size()
              << " drafter=" << dirEnv;
    using clk = std::chrono::steady_clock;

    // Baseline greedy generate().
    ::mimirmind::runtime::GenerateParams gp{};
    gp.maxNewTokens         = maxNew;
    gp.sampling.temperature = 0.0F;
    engine.resetCache();
    const auto tb0 = clk::now();
    std::vector<std::int32_t> ref =
        engine.generate(pids, gp, {}, nullptr, {}, {});
    const double baseMs =
        std::chrono::duration<double, std::milli>(clk::now() - tb0).count();

    // DFlash greedy generate.
    std::size_t drafted = 0, accepted = 0;
    const auto td0 = clk::now();
    std::vector<std::int32_t> df =
        engine.generateDflash(pids, maxNew, draftN, tok.eosId(), dirEnv,
                          &drafted, &accepted);
    const double dfMs =
        std::chrono::duration<double, std::milli>(clk::now() - td0).count();

    std::size_t matchLen = 0;
    const std::size_t cn = std::min(ref.size(), df.size());
    for (; matchLen < cn; ++matchLen)
        if (ref[matchLen] != df[matchLen]) break;
    const bool identical =
        (ref.size() == df.size()) && (matchLen == ref.size());
    const double acceptRate =
        drafted > 0 ? static_cast<double>(accepted) / drafted : 0.0;
    // Each round emits (accepted_r + 1) tokens => rounds = emitted - accepted.
    const std::size_t rounds =
        df.size() > accepted ? df.size() - accepted : 0;
    const double acceptLen =
        rounds > 0 ? static_cast<double>(accepted) / rounds : 0.0;

    std::cout << "\n[M-Cuda.DFlash] draftN=" << draftN
              << " (block " << (draftN + 1) << ") maxNew=" << maxNew
              << "\n  output-identical=" << (identical ? "YES" : "NO")
              << " (match " << matchLen << "/" << ref.size() << ", dflash "
              << df.size() << ")\n"
              << "  accept-len=" << acceptLen << " (accepted " << accepted
              << " / rounds " << rounds << "; accept-rate=" << acceptRate
              << ", drafted " << drafted << ")\n"
              << "  baseline=" << baseMs << " ms  dflash=" << dfMs
              << " ms  speedup=" << (dfMs > 0 ? baseMs / dfMs : 0.0)
              << "x\n"
              << "  => DFlash " << (identical ? "PASS" : "MISMATCH") << "\n";
    std::cout.flush();
    return 0;
}


// 5.9.1 — DFlash SERVING-batched spec decode: shared prompt over N slots,
// per-slot block-draft -> ONE batched verify over M=N*(K+1) (expert-read
// amortization) -> per-slot accept via the per-timestep SSM export. The
// >1x lever: speedup vs plain batched decode should rise with N. Also a
// correctness signal: all N slots (identical prompt) must produce the
// same stream.
int ServingBenchModes::runDflashBatch(
    runtime::InferenceEngine& engine,
    [[maybe_unused]] const std::string& arch,
    [[maybe_unused]] const core::config::Config& cfg) {
    const char* dirEnv = std::getenv("MIMIRMIND_DFLASH_DIR");
    if (dirEnv == nullptr) {
        std::cout << "\n[DFlash.Batch] set MIMIRMIND_DFLASH_DIR\n";
        std::cout.flush();
        return 0;
    }
    const auto& tok = engine.tokenizer();
    const char* promptEnv = std::getenv("MIMIRMIND_DFLASH_PROMPT");
    std::vector<std::int32_t> pids = tok.encode(
        promptEnv != nullptr
            ? promptEnv
            : "The history of artificial intelligence began when",
        /*addBos=*/false);
    if (pids.empty()) pids.push_back(1);
    std::size_t K = 7, N = 8, maxNew = 96;
    if (const char* v = std::getenv("MIMIRMIND_DFLASH_N")) {
        const long x = std::strtol(v, nullptr, 10); if (x >= 1) K = static_cast<std::size_t>(x);
    }
    if (const char* v = std::getenv("MIMIRMIND_DFLASH_NSEQ")) {
        const long x = std::strtol(v, nullptr, 10); if (x >= 1) N = static_cast<std::size_t>(x);
    }
    if (const char* v = std::getenv("MIMIRMIND_DFLASH_MAXNEW")) {
        const long x = std::strtol(v, nullptr, 10); if (x >= 1) maxNew = static_cast<std::size_t>(x);
    }
    using clk = std::chrono::steady_clock;

    std::vector<std::vector<std::int32_t>> prompts(N, pids);
    const auto tb0 = clk::now();
    const auto base = engine.generateBatch(prompts, maxNew, tok.eosId());
    const double baseMs =
        std::chrono::duration<double, std::milli>(clk::now() - tb0).count();

    std::size_t drafted = 0, accepted = 0;
    const auto td0 = clk::now();
    const auto df = engine.generateBatchDflash(pids, N, maxNew, K, tok.eosId(),
                                           dirEnv, &drafted, &accepted);
    const double dfMs =
        std::chrono::duration<double, std::milli>(clk::now() - td0).count();

    bool allEq = true;
    for (std::size_t s = 1; s < df.size(); ++s) {
        if (df[s] != df[0]) allEq = false;
    }
    const std::size_t roundsTot = (K > 0 ? drafted / K : 0);
    const double acceptLen =
        roundsTot > 0 ? static_cast<double>(accepted) / roundsTot : 0.0;
    const double acceptRate =
        drafted > 0 ? static_cast<double>(accepted) / drafted : 0.0;
    std::cout << "\n[DFlash.Batch] N=" << N << " K=" << K
              << " (block " << (K + 1) << ") maxNew=" << maxNew
              << " promptTokens=" << pids.size()
              << "\n  slots-identical=" << (allEq ? "YES" : "NO")
              << " (df[0].len=" << (df.empty() ? 0 : df[0].size()) << ")"
              << "\n  accept-len=" << acceptLen
              << " accept-rate=" << acceptRate
              << " (accepted " << accepted << " / drafted " << drafted << ")"
              << "\n  baseline(batch)=" << baseMs << " ms  dflash(batch)="
              << dfMs << " ms  speedup=" << (dfMs > 0 ? baseMs / dfMs : 0.0)
              << "x\n";
    std::cout.flush();
    return 0;
}


// M-Cuda.MTP Increment E1 — batched-verify parity gate. With a
// single slot (N=1), stepServingVerify (full-attention over virtual
// paged slots + K+1 sequential GatedDeltaNet steps + per-step SSM
// snapshot) MUST reproduce single-session forwardVerify on the SAME
// committed prefix and the SAME K+1 verify tokens, position for
// position. This isolates the new verify orchestration from the
// paged-vs-contiguous substrate parity already proven by
// generateServingParity.
int ServingBenchModes::runMtpVerifyTest(
    runtime::InferenceEngine& engine,
    [[maybe_unused]] const std::string& arch,
    [[maybe_unused]] const core::config::Config& cfg) {
    const auto& tok = engine.tokenizer();
    const char* promptEnv = std::getenv("MIMIRMIND_MTP_PROMPT");
    std::vector<std::int32_t> pids = tok.encode(
        promptEnv != nullptr
            ? promptEnv
            : "The history of artificial intelligence began when",
        /*addBos=*/false);
    if (pids.empty()) pids.push_back(1);
    std::size_t K = 2;
    if (const char* dv = std::getenv("MIMIRMIND_MTP_DEPTH")) {
        const long v = std::strtol(dv, nullptr, 10);
        if (v >= 1) K = static_cast<std::size_t>(v);
    }
    const std::size_t P = pids.size();

    // K+1 in-distribution verify tokens (token0 + K greedy drafts).
    // Their VALUES only need to agree across the two forward paths —
    // the gate compares logits, not correctness — so a short greedy
    // generate() is a convenient, realistic source.
    ::mimirmind::runtime::GenerateParams gp{};
    gp.maxNewTokens         = K + 1;
    gp.sampling.temperature = 0.0F;
    engine.resetCache();
    std::vector<std::int32_t> vtoks =
        engine.generate(pids, gp, {}, nullptr, {}, {});
    if (vtoks.empty()) vtoks.push_back(1);
    vtoks.resize(K + 1, vtoks.back());

    // Reference: single-session forwardVerify over the committed
    // prompt, then the K+1 verify tokens (provisional, not committed).
    engine.resetCache();
    (void)engine.forwardVerify(pids);
    engine.commitVerified(pids);
    const std::vector<std::vector<float>> ref = engine.forwardVerify(vtoks);

    // Test: serving path. Prefill the prompt through stepServing to
    // build the paged KV + recurrent state to position P, then verify.
    using Step = ::mimirmind::runtime::InferenceEngine::ServingSlotStep;
    using VSlot = ::mimirmind::runtime::InferenceEngine::VerifySlot;
    engine.ensureServingState(/*maxBatch=*/1, /*maxContext=*/P + K + 8);
    for (std::size_t g = 0; g < P; ++g) {
        std::vector<Step> steps(1);
        steps[0].slot     = 0;
        steps[0].token    = pids[g];
        steps[0].pos      = static_cast<std::int32_t>(g);
        steps[0].seqStart = (g == 0);
        std::vector<std::int32_t> outTok(1, 0);
        engine.stepServing(steps, outTok);
    }
    std::vector<VSlot> vs(1);
    vs[0].slot    = 0;
    vs[0].basePos = static_cast<std::int32_t>(P);
    const std::vector<std::vector<float>> test =
        engine.stepServingVerify(vs, vtoks, K);

    bool   allMatch = (ref.size() == test.size());
    double maxDelta = 0.0;
    const std::size_t rows = std::min(ref.size(), test.size());
    std::cout << "\n[M-Cuda.MTP.E1] depth=" << K << " promptTokens=" << P;
    for (std::size_t j = 0; j < rows; ++j) {
        std::size_t aRef = 0, aTst = 0;
        float bRef = ref[j][0], bTst = test[j][0];
        const std::size_t V = std::min(ref[j].size(), test[j].size());
        for (std::size_t v = 1; v < V; ++v) {
            if (ref[j][v]  > bRef) { bRef = ref[j][v];  aRef = v; }
            if (test[j][v] > bTst) { bTst = test[j][v]; aTst = v; }
            double d = static_cast<double>(ref[j][v]) - test[j][v];
            if (d < 0.0) d = -d;
            if (d > maxDelta) maxDelta = d;
        }
        if (aRef != aTst) allMatch = false;
        std::cout << "\n  pos " << j << " argmax ref=" << aRef
                  << " test=" << aTst
                  << (aRef == aTst ? "  match" : "  MISMATCH");
    }
    std::cout << "\n  maxLogitDelta=" << maxDelta
              << "\n  => E1 verify " << (allMatch ? "PASS" : "MISMATCH")
              << "\n";
    std::cout.flush();
    return 0;
}


// M-Cuda.MTP Increment E2 — per-slot MTP draft parity gate. Drafts
// `depth` tokens for `nSeq` slots, each on its own nextn KV cache,
// and checks (a) all slots agree (per-slot KV isolation) and (b)
// they match an independent single-sequence reference draft chain
// (the serving per-slot KV path == single-session draft).
int ServingBenchModes::runMtpDraftTest(
    runtime::InferenceEngine& engine,
    [[maybe_unused]] const std::string& arch,
    [[maybe_unused]] const core::config::Config& cfg) {
    if (!engine.mtpAvailable()) {
        std::cout << "\n[M-Cuda.MTP.E2] model has no nextn head — skipped\n";
        std::cout.flush();
        return 0;
    }
    const auto& tok = engine.tokenizer();
    const char* promptEnv = std::getenv("MIMIRMIND_MTP_PROMPT");
    std::vector<std::int32_t> pids = tok.encode(
        promptEnv != nullptr
            ? promptEnv
            : "The history of artificial intelligence began when",
        /*addBos=*/false);
    if (pids.empty()) pids.push_back(1);
    std::size_t K = 4;
    if (const char* dv = std::getenv("MIMIRMIND_MTP_DEPTH")) {
        const long v = std::strtol(dv, nullptr, 10);
        if (v >= 1) K = static_cast<std::size_t>(v);
    }
    std::size_t N = 4;
    if (const char* nv = std::getenv("MIMIRMIND_MTP_NSEQ")) {
        const long v = std::strtol(nv, nullptr, 10);
        if (v >= 1) N = static_cast<std::size_t>(v);
    }
    const auto r = engine.mtpDraftParity(pids, N, K);
    std::cout << "\n[M-Cuda.MTP.E2] nSeq=" << N << " depth=" << K
              << " promptTokens=" << pids.size() << "\n  ref  drafts:";
    for (const std::int32_t t : r.refDrafts) std::cout << " " << t;
    for (std::size_t s = 0; s < r.slotDrafts.size(); ++s) {
        std::cout << "\n  slot " << s << " drafts:";
        for (const std::int32_t t : r.slotDrafts[s]) std::cout << " " << t;
    }
    const bool pass = r.allSlotsAgree && r.matchesReference;
    std::cout << "\n  allSlotsAgree=" << (r.allSlotsAgree ? "YES" : "NO")
              << " matchesReference=" << (r.matchesReference ? "YES" : "NO")
              << "\n  => E2 draft " << (pass ? "PASS" : "MISMATCH") << "\n";
    std::cout.flush();
    return 0;
}


// M-Cuda.MTP Increment E3 — batched native MTP decode parity gate.
// generateBatchMtp (per-slot draft -> batched verify -> per-slot
// accept + snapshot restore, no re-forward) over nSeq identical
// prompts MUST (a) produce identical streams and (b) match single-
// session generateMtp bit-for-bit — the end-to-end Increment E gate.
int ServingBenchModes::runMtpBatchTest(
    runtime::InferenceEngine& engine,
    [[maybe_unused]] const std::string& arch,
    [[maybe_unused]] const core::config::Config& cfg) {
    if (!engine.mtpAvailable()) {
        std::cout << "\n[M-Cuda.MTP.E3] model has no nextn head — skipped\n";
        std::cout.flush();
        return 0;
    }
    const auto& tok = engine.tokenizer();
    const char* promptEnv = std::getenv("MIMIRMIND_MTP_PROMPT");
    std::vector<std::int32_t> pids = tok.encode(
        promptEnv != nullptr
            ? promptEnv
            : "The history of artificial intelligence began when",
        /*addBos=*/false);
    if (pids.empty()) pids.push_back(1);
    std::size_t K = 2;
    if (const char* dv = std::getenv("MIMIRMIND_MTP_DEPTH")) {
        const long v = std::strtol(dv, nullptr, 10);
        if (v >= 1) K = static_cast<std::size_t>(v);
    }
    std::size_t N = 4;
    if (const char* nv = std::getenv("MIMIRMIND_MTP_NSEQ")) {
        const long v = std::strtol(nv, nullptr, 10);
        if (v >= 1) N = static_cast<std::size_t>(v);
    }
    std::size_t maxNew = 32;
    if (const char* mv = std::getenv("MIMIRMIND_MTP_MAXNEW")) {
        const long v = std::strtol(mv, nullptr, 10);
        if (v >= 1) maxNew = static_cast<std::size_t>(v);
    }

    const auto batched = engine.generateBatchMtp(pids, N, maxNew, K, tok.eosId());
    std::size_t drafted = 0, accepted = 0;
    const auto ref = engine.generateMtp(pids, maxNew, K, tok.eosId(),
                                    &drafted, &accepted);

    bool allEq = true;
    for (std::size_t s = 1; s < batched.size(); ++s) {
        if (batched[s] != batched[0]) allEq = false;
    }
    const std::vector<std::int32_t>& b0 =
        batched.empty() ? ref : batched[0];
    std::size_t matchLen = 0;
    const std::size_t cn = std::min(ref.size(), b0.size());
    for (; matchLen < cn; ++matchLen) {
        if (ref[matchLen] != b0[matchLen]) break;
    }
    const bool slot0Exact =
        (ref.size() == b0.size()) && (matchLen == ref.size());
    const bool pass = allEq && slot0Exact;

    std::cout << "\n[M-Cuda.MTP.E3] nSeq=" << N << " depth=" << K
              << " maxNew=" << maxNew << " promptTokens=" << pids.size()
              << "\n  ref    (" << ref.size() << "):";
    for (const std::int32_t t : ref) std::cout << " " << t;
    std::cout << "\n  slot 0 (" << b0.size() << "):";
    for (const std::int32_t t : b0) std::cout << " " << t;
    std::cout << "\n  allSlotsAgree=" << (allEq ? "YES" : "NO")
              << " slot0==single-session=" << (slot0Exact ? "YES" : "NO")
              << " (match " << matchLen << "/" << ref.size() << ")"
              << "\n  => E3 batch-MTP " << (pass ? "PASS" : "MISMATCH")
              << "\n";
    std::cout.flush();
    return 0;
}


// M-Cuda.MTP accept-rate sweep — measures single-session MTP accept over
// depths 1..4 on a few prompts (no throughput sweep, one load). SpecLA
// shows spec-decode on linear-attention loses below ~0.7 accept; this
// checks whether lower depth reaches that on qwen3.6.
int ServingBenchModes::runMtpAccept(
    runtime::InferenceEngine& engine,
    [[maybe_unused]] const std::string& arch,
    [[maybe_unused]] const core::config::Config& cfg) {
    if (!engine.mtpAvailable()) {
        std::cout << "\n[MTP-ACCEPT] model has no nextn head — skipped\n";
        std::cout.flush();
        return 0;
    }
    const auto& tok = engine.tokenizer();
    const char* prompts[] = {
        "The history of artificial intelligence began when",
        "Write a Python function that returns the nth Fibonacci number.",
        "Explain the theory of relativity in simple terms."
    };
    std::size_t maxNew = 128;
    if (const char* mv = std::getenv("MIMIRMIND_MTP_MAXNEW")) {
        const long v = std::strtol(mv, nullptr, 10);
        if (v >= 1) maxNew = static_cast<std::size_t>(v);
    }
    std::cout << "\n[MTP-ACCEPT] maxNew=" << maxNew
              << " prompts=3 (per-token accept = accepted/drafted)\n";
    for (std::size_t K = 1; K <= 4; ++K) {
        std::size_t totD = 0, totA = 0;
        for (const char* p : prompts) {
            std::vector<std::int32_t> pids = tok.encode(p, /*addBos=*/false);
            if (pids.empty()) pids.push_back(1);
            std::size_t d = 0, a = 0;
            (void)engine.generateMtp(pids, maxNew, K, tok.eosId(), &d, &a);
            totD += d;
            totA += a;
        }
        const double acc =
            totD > 0 ? static_cast<double>(totA) / totD : 0.0;
        // Expected accepted tokens per draft round (geometric, per-token p):
        const double eAcc =
            (acc < 1.0) ? (acc * (1.0 - std::pow(acc, static_cast<double>(K))) /
                           (1.0 - acc))
                        : static_cast<double>(K);
        std::cout << "  depth=" << K << "  accept=" << acc << "  (" << totA
                  << "/" << totD << ")  E[accepted/round]=" << eAcc
                  << "  tokens/step=" << (1.0 + eAcc) << "\n";
        std::cout.flush();
    }
    return 0;
}


// M-Cuda.MTP re-test — MTP-at-serving-scale THROUGHPUT. Times batched
// native MTP (generateBatchMtp) vs plain batched decode (generateBatch)
// at nSeq {16,32,64}, depth K. The Increment-E "net-loss" verdict was
// measured at nSeq16; this checks whether a larger batch amortises the
// MTP verify (esp. the GDN K+1 sequential passes) and flips it to a win.
int ServingBenchModes::runMtpPerf(
    runtime::InferenceEngine& engine,
    [[maybe_unused]] const std::string& arch,
    [[maybe_unused]] const core::config::Config& cfg) {
    if (!engine.mtpAvailable()) {
        std::cout << "\n[MTP-PERF] model has no nextn head — skipped\n";
        std::cout.flush();
        return 0;
    }
    using clk = std::chrono::steady_clock;
    const auto& tok = engine.tokenizer();
    const char* promptEnv = std::getenv("MIMIRMIND_MTP_PROMPT");
    std::vector<std::int32_t> pids = tok.encode(
        promptEnv != nullptr
            ? promptEnv
            : "The history of artificial intelligence began when",
        /*addBos=*/false);
    if (pids.empty()) pids.push_back(1);
    std::size_t K = 3;
    if (const char* dv = std::getenv("MIMIRMIND_MTP_DEPTH")) {
        const long v = std::strtol(dv, nullptr, 10);
        if (v >= 1) K = static_cast<std::size_t>(v);
    }
    std::size_t maxNew = 128;
    if (const char* mv = std::getenv("MIMIRMIND_MTP_MAXNEW")) {
        const long v = std::strtol(mv, nullptr, 10);
        if (v >= 1) maxNew = static_cast<std::size_t>(v);
    }
    // Single-session accept-rate for context (drafted/accepted).
    std::size_t drafted = 0, accepted = 0;
    (void)engine.generateMtp(pids, maxNew, K, tok.eosId(), &drafted, &accepted);
    const double acc =
        drafted > 0 ? static_cast<double>(accepted) / drafted : 0.0;
    std::cout << "\n[MTP-PERF] depth=" << K << " maxNew=" << maxNew
              << " promptTokens=" << pids.size()
              << " single-session accept=" << acc << " (" << accepted
              << "/" << drafted << ")\n";
    auto genCount = [](const std::vector<std::vector<std::int32_t>>& o) {
        std::size_t n = 0;
        for (const auto& s : o) n += s.size();
        return n;
    };
    for (std::size_t N : {std::size_t{16}, std::size_t{32}, std::size_t{64}}) {
        std::vector<std::vector<std::int32_t>> prompts(N, pids);
        (void)engine.generateBatchMtp(pids, N, 4, K, tok.eosId());   // warm
        (void)engine.generateBatch(prompts, 4, /*eosId=*/-1);        // warm
        const auto m0 = clk::now();
        const auto mout = engine.generateBatchMtp(pids, N, maxNew, K, tok.eosId());
        const auto m1 = clk::now();
        const auto p0 = clk::now();
        const auto pout = engine.generateBatch(prompts, maxNew, /*eosId=*/-1);
        const auto p1 = clk::now();
        const double mtpS = std::chrono::duration<double>(m1 - m0).count();
        const double plnS = std::chrono::duration<double>(p1 - p0).count();
        const double mtpTps = static_cast<double>(genCount(mout)) / mtpS;
        const double plnTps = static_cast<double>(genCount(pout)) / plnS;
        std::cout << "  nSeq=" << N << "  plain=" << plnTps << " tok/s ("
                  << plnS << "s)  MTP=" << mtpTps << " tok/s (" << mtpS
                  << "s)  ratio=" << (plnTps > 0 ? mtpTps / plnTps : 0.0)
                  << "x\n";
        std::cout.flush();
    }
    return 0;
}


// M-Cuda.MTP Increment E4 — heterogeneous batched MTP parity gate.
// generateBatchMtpMulti over DIFFERENT prompts (different content and
// length, slots diverging in position + finishing independently) MUST
// reproduce each prompt's single-session generateMtp bit-for-bit —
// the multi-tenant serving-correctness gate for native MTP.
int ServingBenchModes::runMtpMultiTest(
    runtime::InferenceEngine& engine,
    [[maybe_unused]] const std::string& arch,
    [[maybe_unused]] const core::config::Config& cfg) {
    if (!engine.mtpAvailable()) {
        std::cout << "\n[M-Cuda.MTP.E4] model has no nextn head — skipped\n";
        std::cout.flush();
        return 0;
    }
    const auto& tok = engine.tokenizer();
    const std::vector<std::string> texts = {
        "The history of artificial intelligence began when",
        "In a distant galaxy far beyond the reach of",
        "Python is a programming language",
        "Once upon a time there lived a wise old king who ruled",
    };
    std::vector<std::vector<std::int32_t>> prompts;
    for (const auto& t : texts) {
        auto ids = tok.encode(t, /*addBos=*/false);
        if (ids.empty()) ids.push_back(1);
        prompts.push_back(std::move(ids));
    }
    std::size_t K = 2;
    if (const char* dv = std::getenv("MIMIRMIND_MTP_DEPTH")) {
        const long v = std::strtol(dv, nullptr, 10);
        if (v >= 1) K = static_cast<std::size_t>(v);
    }
    std::size_t maxNew = 32;
    if (const char* mv = std::getenv("MIMIRMIND_MTP_MAXNEW")) {
        const long v = std::strtol(mv, nullptr, 10);
        if (v >= 1) maxNew = static_cast<std::size_t>(v);
    }

    const auto batched =
        engine.generateBatchMtpMulti(prompts, maxNew, K, tok.eosId());

    bool allExact = (batched.size() == prompts.size());
    std::cout << "\n[M-Cuda.MTP.E4] nSeq=" << prompts.size() << " depth=" << K
              << " maxNew=" << maxNew;
    for (std::size_t s = 0; s < prompts.size(); ++s) {
        std::size_t drafted = 0, accepted = 0;
        const auto ref = engine.generateMtp(prompts[s], maxNew, K,
                                        tok.eosId(), &drafted, &accepted);
        const auto& b = batched[s];
        std::size_t ml = 0;
        const std::size_t cn = std::min(ref.size(), b.size());
        for (; ml < cn; ++ml) if (ref[ml] != b[ml]) break;
        const bool exact = (ref.size() == b.size()) && (ml == ref.size());
        if (!exact) allExact = false;
        std::cout << "\n  slot " << s << " promptTokens=" << prompts[s].size()
                  << " match " << ml << "/" << ref.size()
                  << (exact ? "  exact" : "  MISMATCH");
    }
    std::cout << "\n  => E4 multi-MTP " << (allExact ? "PASS" : "MISMATCH")
              << "\n";
    std::cout.flush();
    return 0;
}


// M-Cuda.MTP Increment E5 — throughput bench: batched native MTP
// (generateBatchMtp, depth 2/3) vs the non-speculative batched
// baseline (generateBatch) across nSeq, reporting gen-tok/s and the
// MTP speedup. All streams run eosId=-1 so every slot produces
// exactly maxNew tokens (clean token counting).
int ServingBenchModes::runMtpPerfTest(
    runtime::InferenceEngine& engine,
    [[maybe_unused]] const std::string& arch,
    [[maybe_unused]] const core::config::Config& cfg) {
    if (!engine.mtpAvailable()) {
        std::cout << "\n[M-Cuda.MTP.E5] model has no nextn head — skipped\n";
        std::cout.flush();
        return 0;
    }
    const auto& tok = engine.tokenizer();
    const char* promptEnv = std::getenv("MIMIRMIND_MTP_PROMPT");
    std::vector<std::int32_t> pids = tok.encode(
        promptEnv != nullptr
            ? promptEnv
            : "The history of artificial intelligence began when",
        /*addBos=*/false);
    if (pids.empty()) pids.push_back(1);
    std::size_t maxNew = 128;
    if (const char* mv = std::getenv("MIMIRMIND_MTP_MAXNEW")) {
        const long v = std::strtol(mv, nullptr, 10);
        if (v >= 1) maxNew = static_cast<std::size_t>(v);
    }
    std::vector<std::size_t> nSeqs = {1, 4, 8, 16};
    if (const char* nv = std::getenv("MIMIRMIND_MTP_NSEQ")) {
        const long v = std::strtol(nv, nullptr, 10);
        if (v >= 1) nSeqs = {static_cast<std::size_t>(v)};
    }
    // Runbook (MTP depth1/nSeq1 single-user cell): allow selecting a
    // single depth so the untested depth1 point can be measured. The
    // vLLM oracle shows +20% at depth1/single-stream; our net-loss
    // evidence is all depth2-3/nSeq>=16. MIMIRMIND_MTP_DEPTH=1 -> {1}.
    std::vector<std::size_t> depths = {2, 3};
    if (const char* dv = std::getenv("MIMIRMIND_MTP_DEPTH")) {
        const long v = std::strtol(dv, nullptr, 10);
        if (v >= 1) depths = {static_cast<std::size_t>(v)};
    }
    using clk = std::chrono::steady_clock;

    // Representative accept-rate (prompt-dependent, batch-invariant).
    std::cout << "\n[M-Cuda.MTP.E5] maxNew=" << maxNew
              << " promptTokens=" << pids.size() << "\n";
    for (const std::size_t D : depths) {
        std::size_t drafted = 0, accepted = 0;
        (void)engine.generateMtp(pids, 32, D, /*eos=*/-1, &drafted, &accepted);
        std::cout << "  accept-rate depth=" << D << ": "
                  << (drafted ? double(accepted) / double(drafted) : 0.0)
                  << " (" << accepted << "/" << drafted << ")\n";
    }

    // Single-user (nSeq==1) SINGLE-SESSION ratio: plain generate() vs
    // generateMtp(), timed with warmup + resetCache. This is the true
    // single-user decode path (the batched generateBatchMtp path is
    // untested/aborts at depth1). MIMIRMIND_MTP_SINGLE_ONLY=1 reports
    // just this and returns (skips the batched loop below).
    if (std::getenv("MIMIRMIND_MTP_SINGLE_ONLY") != nullptr) {
        ::mimirmind::runtime::GenerateParams sgp{};
        sgp.maxNewTokens         = maxNew;
        sgp.sampling.temperature = 0.0F;
        auto nowMs = []() {
            return std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
        };
        engine.resetCache(); (void)engine.generate(pids, sgp, {}, nullptr, {}, {});
        engine.resetCache(); (void)engine.generateMtp(pids, 8, depths.front(), -1);
        engine.resetCache();
        const double p0 = nowMs();
        const auto plain = engine.generate(pids, sgp, {}, nullptr, {}, {});
        const double plMs = nowMs() - p0;
        const double plTps = double(plain.size()) / (plMs / 1000.0);
        std::cout << "  [single-session nSeq=1] plain=" << plTps
                  << " tok/s (" << plain.size() << " tok, " << plMs << " ms)\n";
        for (const std::size_t D : depths) {
            engine.resetCache();
            const double m0 = nowMs();
            const auto mo = engine.generateMtp(pids, maxNew, D, -1);
            const double mMs = nowMs() - m0;
            const double mTps = double(mo.size()) / (mMs / 1000.0);
            std::cout << "  [single-session nSeq=1] depth=" << D
                      << " MTP=" << mTps << " tok/s (" << mo.size()
                      << " tok, " << mMs << " ms) ratio=" << (mTps / plTps)
                      << "\n";
        }
        std::cout.flush();
        return 0;
    }

    auto timeCall = [&](auto&& fn) -> double {
        const auto t0 = clk::now();
        fn();
        return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    };
    // Global warmup at the largest shape (lazy-allocs all serving +
    // verify buffers; ramps GPU clocks) so timed runs are steady.
    {
        const std::size_t wn = nSeqs.back();
        std::vector<std::vector<std::int32_t>> wp(wn, pids);
        (void)engine.generateBatch(wp, 8, -1);
        (void)engine.generateBatchMtp(pids, wn, 8, depths.back(), -1);
    }

    std::cout << "  nSeq | baseline tok/s | depth2 tok/s (x) | depth3 tok/s (x)\n";
    for (const std::size_t nSeq : nSeqs) {
        const double toks = double(nSeq) * double(maxNew);
        std::vector<std::vector<std::int32_t>> bp(nSeq, pids);
        const double baseMs =
            timeCall([&]{ (void)engine.generateBatch(bp, maxNew, /*eos=*/-1); });
        const double baseTps = toks / (baseMs / 1000.0);

        std::cout << "  " << nSeq << "    | " << baseTps;
        for (const std::size_t D : depths) {
            const double mtpMs = timeCall(
                [&]{ (void)engine.generateBatchMtp(pids, nSeq, maxNew, D, -1); });
            const double mtpTps = toks / (mtpMs / 1000.0);
            std::cout << " | " << mtpTps << " (" << (mtpTps / baseTps) << ")";
        }
        std::cout << "\n";
        std::cout.flush();
    }
    std::cout << "  => E5 perf done\n";
    std::cout.flush();
    return 0;
}

} // namespace mimirmind::cli
