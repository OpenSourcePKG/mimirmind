// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "mimirmind/ServeMode.hpp"

#include "mimirmind/CliArgs.hpp"
#include "mimirmind/CliParser.hpp"

#ifdef MIMIRMIND_HAVE_L0
#include "compute/l0/GpuOps.hpp"
#endif
#include "core/backend/BackendPool.hpp"
#include "core/backend/BackendRegistry.hpp"
#include "core/config/Config.hpp"
#ifdef MIMIRMIND_HAVE_L0
#include "core/ipc/MuninClient.hpp"
#endif
#include "core/log/Log.hpp"
#include "core/os/GovernorLock.hpp"
#include "model/Tokenizer.hpp"
#include "runtime/InferenceEngine.hpp"
#include "runtime/serving/ContinuousBatcher.hpp"
#include "runtime/nvfp4/ModelFormatResolver.hpp"
#include "runtime/perf/PerfRegressionDetector.hpp"
#include "runtime/spec/Drafter.hpp"
#include "runtime/spec/ModelDrafter.hpp"
#include "runtime/spec/NGramDrafter.hpp"
#include "runtime/thermal/FanController.hpp"
#include "runtime/thermal/GpuClockGovernor.hpp"
#include "runtime/thermal/PowerMonitor.hpp"
#include "runtime/thermal/SystemMonitor.hpp"
#include "runtime/thermal/ThermalGuard.hpp"
#include "runtime/thermal/ThermalProfile.hpp"
#include "server/ApiServer.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace {

// Held in the SIGINT handler so a Ctrl-C asks the listener to drain.
std::atomic<::mimirmind::server::ApiServer*> g_runningServer{nullptr};

} // namespace

extern "C" void signalStop(int /*sig*/) {
    if (auto* s = g_runningServer.load(std::memory_order_acquire)) {
        s->stop();
    }
}

namespace mimirmind::cli {

int runServe(const CliArgs& args, const ::mimirmind::core::config::Config& cfg) {
    std::cout << kBanner;
    std::cout.flush();

    if (args.modelPath.empty()) {
        std::cerr << "serve: models[<defaultModel>].path is required "
                     "(fill it in config.json or pass --model PATH)\n";
        return 2;
    }

    // ---- M-Munin attached mode -----------------------------------------
    // Two things happen up-front:
    //   1. Standalone workers acquire the governor flock so a second
    //      standalone process on the same host fails fast rather than
    //      dueling over sysfs writes.
    //   2. Attached workers instead probe Munin's healthz to confirm the
    //      daemon is up and the models Munin holds cover our loadOnStart
    //      list. We refuse to start if any expected model is missing —
    //      failing at boot beats failing on the first request.
    const bool attachedMode = !args.attachSocket.empty();

#ifndef MIMIRMIND_HAVE_L0
    if (attachedMode) {
        std::cerr << "serve: --attach requested but this build has no "
                     "L0 backend compiled in — Munin's IPC surface uses "
                     "L0 handles, so attached mode is L0-only. Rebuild "
                     "with -DMIMIRMIND_ENABLE_L0=ON or drop --attach.\n";
        return 2;
    }
#endif

    std::optional<::mimirmind::core::os::GovernorLock> governorLock;
    // These dev/bench hooks exit before the HTTP server starts, so they
    // need neither the thermal governor nor exclusive ownership — skip the
    // flock so they can run alongside a live serve/Munin worker (subject to
    // host memory). MIMIRMIND_L0_BATCH (the L0 synchronized batched-decode
    // parity+perf hook) is one of them: without this it would fail the
    // GovernorLock::tryAcquire while Munin holds the lock and never reach
    // the bench block below.
    const bool servingParity =
        std::getenv("MIMIRMIND_SERVING_PARITY") != nullptr ||
        std::getenv("MIMIRMIND_BATCH_BENCH")    != nullptr ||
        std::getenv("MIMIRMIND_L0_BATCH")       != nullptr;
    if (!attachedMode && !servingParity) {
        auto lk = ::mimirmind::core::os::GovernorLock::tryAcquire();
        if (!lk) {
            std::cerr << "serve: " << lk.error()
                      << "\nHint: if Munin is running, start this "
                         "worker with --attach unix:/var/run/munin/munin.sock "
                         "so it does not compete for governor ownership.\n";
            return 2;
        }
        governorLock = std::move(*lk);
        MM_LOG_INFO("main",
                    "serve: acquired governor flock at '{}'",
                    governorLock->path());
    }

    // Probed once before the per-model attach loop so a dead Munin does
    // not manifest as N confusing per-model attach errors. The whole
    // Munin/attach chain is L0-only; a HIP-only build already errored
    // out above if `--attach` was set, so this block is unreachable
    // and its `MuninClient` references would fail to compile without
    // the guard.
#ifdef MIMIRMIND_HAVE_L0
    if (attachedMode) {
        MM_LOG_INFO("main",
                    "serve: attached mode — probing Munin at '{}'",
                    args.attachSocket);
        auto hz = ::mimirmind::core::ipc::MuninClient::healthz(args.attachSocket);
        if (!hz) {
            std::cerr << "serve: Munin healthz failed at '"
                      << args.attachSocket << "': " << hz.error() << "\n";
            return 2;
        }
        if (hz->governorOwner != "munin") {
            std::cerr << "serve: refusing to attach — Munin reports "
                         "governor_owner='" << hz->governorOwner
                      << "', expected 'munin'. Standalone-worker "
                         "handoff back to Munin is not part of "
                         "Schritt 8-minimal (M-Munin ADR).\n";
            return 2;
        }
        MM_LOG_INFO("main",
                    "serve: Munin healthz ok — pid={} models={} owner={}",
                    hz->pid, hz->models.size(), hz->governorOwner);
        for (const auto& m : hz->models) {
            MM_LOG_INFO("main",
                        "  munin-model id='{}' fingerprint='{}' bytes={}",
                        m.id, m.fingerprint, m.totalBytes);
        }
    }
#endif

    // Load every loadOnStart:true model. Each gets its own InferenceEngine
    // (own L0 context, USM, autotune) — request dispatch picks the target
    // via `req.model`. Startup cost scales linearly with N (each model
    // runs its own selfTest + autotune pass), and USM is shared UMA-style
    // so N models × their footprint × ~1.2 must fit under
    // runtime.usmProbeTotalGib.
    auto applyRuntimeOverrides = [&](::mimirmind::runtime::InferenceEngine& e,
                                     const ::mimirmind::core::config::RuntimeSettings& rt) {
        if (rt.maxContextTokens.has_value() && *rt.maxContextTokens > 0) {
            e.setMaxContextTokens(*rt.maxContextTokens);
        }
        if (rt.kvDtype.has_value()) {
            const std::string_view v{*rt.kvDtype};
            if (v == "fp16")           e.setKvDtype(::mimirmind::runtime::KvDtype::FP16);
            else if (v == "q8_0")      e.setKvDtype(::mimirmind::runtime::KvDtype::Q8_0);
            else if (v == "f32" || v.empty())
                                       e.setKvDtype(::mimirmind::runtime::KvDtype::F32);
            else {
                MM_LOG_WARN("main",
                            "runtime.kvDtype='{}' unrecognised — falling "
                            "back to f32", v);
            }
        }
    };

    std::vector<std::unique_ptr<::mimirmind::runtime::InferenceEngine>> ownedEngines;
    std::vector<::mimirmind::server::LoadedEngine> loadedEngines;
#ifdef MIMIRMIND_HAVE_L0
    // In attached mode: one MuninClient per loaded model, kept alive
    // for the whole worker lifetime so Munin's implicit-detach logic
    // sees the peer-close only when the worker actually shuts down.
    // L0-only — Munin's IPC surface uses `zeMemOpenIpcHandle`.
    std::vector<std::unique_ptr<::mimirmind::core::ipc::MuninClient>> attachedClients;
#endif

    // Enumerate every compiled-in + runtime-available backend/device
    // once, up-front. Per-model config gets to pick its entry by token
    // (`models[].backend`) — enables dual-GPU deployments (target on
    // dGPU, draft on iGPU) without spawning two worker processes.
    ::mimirmind::core::backend::BackendPool backendPool;
    backendPool.discoverAll();

    for (const auto& m : cfg.models) {
        if (!m.loadOnStart) continue;
        ::mimirmind::core::backend::BackendKind engineKind{};
        try {
            const std::string token = m.backend.empty() ? std::string{"auto"} : m.backend;
            auto& entry = backendPool.selectByToken(token);
            engineKind  = entry.kind;
            MM_LOG_INFO("main",
                        "serve: model '{}' bound to backend '{}' via token '{}'",
                        m.id,
                        ::mimirmind::core::backend::BackendRegistry::name(entry.kind),
                        entry.token);
        } catch (const std::exception& x) {
            std::cerr << "serve: model '" << m.id
                      << "' backend='" << m.backend << "' cannot be "
                      << "resolved: " << x.what() << "\n";
            return 2;
        }
        auto e = std::make_unique<::mimirmind::runtime::InferenceEngine>(
            cfg, engineKind);

#ifdef MIMIRMIND_HAVE_L0
        if (attachedMode) {
            MM_LOG_INFO("main",
                        "serve: attaching to Munin for model '{}' "
                        "(local header from '{}')", m.id, m.path);
            auto client = std::make_unique<::mimirmind::core::ipc::MuninClient>(
                e->ctx());
            auto result = client->attach(args.attachSocket, m.id);
            if (!result) {
                std::cerr << "serve: MuninClient::attach for id='"
                          << m.id << "' failed: " << result.error() << "\n";
                return 2;
            }
            try {
                e->loadModelAttached(m.path,
                                     result->manifest,
                                     std::span<void* const>{result->chunkBases});
            } catch (const std::exception& x) {
                std::cerr << "serve: loadModelAttached('" << m.id
                          << "') failed: " << x.what() << "\n";
                return 2;
            }
            attachedClients.push_back(std::move(client));
        } else
#endif
        {
            if (runtime::nvfp4::resolveModelFormat(m.format, m.path)
                == core::config::ModelFormat::Nvfp4) {
                MM_LOG_INFO("main", "serve: loading NVFP4 model '{}' (id='{}')",
                            m.path, m.id);
                e->loadModelNvfp4(m.path, m.tokenizerGguf);
            } else {
                MM_LOG_INFO("main", "serve: loading model '{}' (id='{}')",
                            m.path, m.id);
                e->loadModel(m.path);
            }
        }

        const auto& arch = e->config().architecture;
        if (arch != "qwen2" && arch != "gemma4" && arch != "qwen35moe") {
            const std::string msg =
                "serve: architecture '" + arch + "' (model id '" + m.id +
                "') is not implemented yet. See "
                "Memory/mimirmind/research/m8-gemma4-staging.md.";
            MM_LOG_ERROR("main", "{}", msg);
            std::cerr << msg << "\n";
            return 2;
        }
        // setKvDtype + setMaxContextTokens inspect loaded model state
        // (fused-QKV coverage, attn_k/v.bias presence per block), so
        // apply the per-model runtime overrides AFTER loadModel.
        applyRuntimeOverrides(*e, cfg.effectiveRuntime(m.id));

        // M-Cuda.Batch D2e — batched decode throughput benchmark. With
        // MIMIRMIND_BATCH_BENCH set, time generateBatch across batch sizes
        // and report ms/step + generated-tokens/s, then exit. The batched
        // forward processes all nSeq sequences in one pass, so gen-tok/s
        // should scale with the batch while ms/step stays roughly flat —
        // that is the serving-class throughput win. qwen35moe only.
        if (arch == "qwen35moe" &&
            std::getenv("MIMIRMIND_BATCH_BENCH") != nullptr) {
            const auto& tok = e->tokenizer();
            auto base = tok.encode("The capital of France is", /*addBos=*/false);
            if (base.empty()) base.push_back(1);
            const bool quick = std::getenv("MIMIRMIND_BENCH_QUICK") != nullptr;
            const std::size_t maxNew    = quick ? 4 : 32;
            const std::size_t promptLen = base.size();
            const std::vector<std::size_t> batchSizes =
                quick ? std::vector<std::size_t>{1}
                      : std::vector<std::size_t>{1, 4, 8, 16};
            std::cout << "\n[M-Cuda.Batch D2e bench] promptLen=" << promptLen
                      << " maxNew=" << maxNew << "\n";
            // Single-session baseline on THIS box+model (apples-to-apples).
            {
                ::mimirmind::runtime::GenerateParams gpb{};
                gpb.maxNewTokens         = maxNew;
                gpb.sampling.temperature = 0.0F;
                e->resetCache();
                const auto s0 = std::chrono::steady_clock::now();
                auto sref = e->generate(base, gpb, {}, nullptr, {}, {});
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
                if (nSeq == 1) {
                    (void)e->generateBatch(prompts, 2, -1);   // warm up
                }
                const auto t0 = std::chrono::steady_clock::now();
                (void)e->generateBatch(prompts, maxNew, /*eosId=*/-1);
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
            return 0;
        }

        // M-Cuda.Batch D2d — batched serving parity gate (dev/CI hook).
        // With MIMIRMIND_SERVING_PARITY set, run the batched decode path
        // (generateServingParity) against single-seq greedy generate() on
        // this freshly-loaded model, print the comparison, and exit before
        // the HTTP server starts. qwen35moe only.
        if (arch == "qwen35moe" &&
            std::getenv("MIMIRMIND_SERVING_PARITY") != nullptr) {
            const auto& tok = e->tokenizer();
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
            const std::size_t maxNew = 8;
            const std::size_t nSeq   = 2;

            ::mimirmind::runtime::GenerateParams gp{};
            gp.maxNewTokens         = maxNew;
            gp.sampling.temperature = 0.0F;   // greedy argmax
            // Reset the KV+SSM state so the reference is a clean single-session
            // run — the prefix cache reuses KV across generate() calls but does
            // NOT restore the GatedDeltaNet recurrent state, which would
            // contaminate a later reference (lcp>0 keeps a stale SsmState).
            e->resetCache();
            std::vector<std::int32_t> ref =
                e->generate(promptIds, gp, {}, nullptr, {}, {});

            auto batched = e->generateServingParity(promptIds, nSeq, maxNew);

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
            const char* multiPrompts[] = {
                "The capital of France is",
                "Once upon a time",
                "2 plus 2 equals",
            };
            std::vector<std::vector<std::int32_t>> bprompts;
            for (const char* mp : multiPrompts) {
                auto ids = tok.encode(mp, /*addBos=*/false);
                if (ids.empty()) ids.push_back(1);
                bprompts.push_back(std::move(ids));
            }
            auto bout = e->generateBatch(bprompts, maxNew, /*eosId=*/-1);
            std::cout << "[M-Cuda.Batch D2e generateBatch] "
                      << bprompts.size() << " distinct prompts, maxNew="
                      << maxNew << "\n";
            bool allBatchOk = true;
            for (std::size_t i = 0; i < bprompts.size(); ++i) {
                ::mimirmind::runtime::GenerateParams gpi{};
                gpi.maxNewTokens         = maxNew;
                gpi.sampling.temperature = 0.0F;
                e->resetCache();   // clean single-session reference (see above)
                std::vector<std::int32_t> refi =
                    e->generate(bprompts[i], gpi, {}, nullptr, {}, {});
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
            std::cout.flush();
            return 0;
        }

        // M-L0.Batch Phase 1 — L0 synchronized batched decode parity + perf.
        // With MIMIRMIND_L0_BATCH set, run the Gemma 4 MoE batched decode
        // path (generateBatchL0): first a greedy parity gate (nSeq identical
        // prompts must produce identical streams AND match single-seq greedy
        // generate()), then a throughput sweep over nSeq. Prints and exits
        // before the HTTP server starts. Xe-LPG / Gemma 4 MoE only.
        if (std::getenv("MIMIRMIND_L0_BATCH") != nullptr) {
            const auto& tok = e->tokenizer();
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
                // --- (1c) greedy parity: nSeq=2 identical vs single-seq ----
                ::mimirmind::runtime::GenerateParams gp{};
                gp.maxNewTokens         = maxNew;
                gp.sampling.temperature = 0.0F;   // greedy argmax
                e->resetCache();
                const std::vector<std::int32_t> ref =
                    e->generate(base, gp, {}, nullptr, {}, {});

                const std::size_t nParity = 2;
                std::vector<std::vector<std::int32_t>> pprompts(nParity, base);
                auto batched = e->generateBatchL0(pprompts, maxNew, /*eosId=*/-1);

                bool allSeqEqual = true;
                for (std::size_t s = 1; s < nParity; ++s) {
                    if (batched[s] != batched[0]) allSeqEqual = false;
                }
                std::size_t matchLen = 0;
                const std::size_t cmpN = std::min(batched[0].size(), ref.size());
                for (; matchLen < cmpN; ++matchLen) {
                    if (batched[0][matchLen] != ref[matchLen]) break;
                }
                std::cout << "\n[M-L0.Batch parity] nSeq=" << nParity
                          << " maxNew=" << maxNew
                          << " promptTokens=" << promptLen << "\n"
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

                // --- (1d) throughput sweep over nSeq ------------------------
                std::cout << "[M-L0.Batch bench] promptLen=" << promptLen
                          << " maxNew=" << maxNew << "\n";
                {
                    e->resetCache();
                    const auto s0 = std::chrono::steady_clock::now();
                    auto sref = e->generate(base, gp, {}, nullptr, {}, {});
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
                          : std::vector<std::size_t>{1, 2, 4, 8};
                for (std::size_t nSeq : batchSizes) {
                    std::vector<std::vector<std::int32_t>> prompts(nSeq, base);
                    (void)e->generateBatchL0(prompts, quick ? 2 : 4, -1); // warm
                    const auto t0 = std::chrono::steady_clock::now();
                    (void)e->generateBatchL0(prompts, maxNew, /*eosId=*/-1);
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
        if (arch == "qwen35moe" &&
            std::getenv("MIMIRMIND_SERVING_LOOP") != nullptr) {
            const auto& tok = e->tokenizer();
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
            e->ensureServingState(/*maxBatch=*/nReq, maxContext);

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
                e->stepServing(steps, toks);

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
                e->resetCache();
                std::vector<std::int32_t> ref =
                    e->generate(pids[r], gpr, {}, nullptr, {}, {});
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
        if (arch == "qwen35moe" &&
            std::getenv("MIMIRMIND_BATCHER_TEST") != nullptr) {
            const auto& tok = e->tokenizer();
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
                e->resetCache();
                refs[r] = e->generate(pids[r], gpr, {}, nullptr, {}, {});
            }

            const std::size_t maxBatch   = 3;   // < nReq => forces slot reuse
            const std::size_t maxContext = maxLen + maxNew + 8;
            ::mimirmind::runtime::serving::ContinuousBatcher batcher(
                *e, maxBatch, maxContext, /*eosId=*/-1);

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
        if (arch == "qwen35moe" &&
            std::getenv("MIMIRMIND_MTP_TEST") != nullptr) {
            if (!e->mtpAvailable()) {
                std::cout << "\n[M-Cuda.MTP] model has no nextn head — skipped\n";
                std::cout.flush();
                return 0;
            }
            const auto& tok = e->tokenizer();
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
            e->resetCache();
            const auto tb0 = clk::now();
            std::vector<std::int32_t> ref =
                e->generate(pids, gp, {}, nullptr, {}, {});
            const double baseMs =
                std::chrono::duration<double, std::milli>(clk::now() - tb0).count();

            // MTP greedy generate.
            std::size_t drafted = 0, accepted = 0;
            const auto tm0 = clk::now();
            std::vector<std::int32_t> mtp =
                e->generateMtp(pids, maxNew, depth, tok.eosId(),
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

        // M9.8b — cross-block sanity check on the effective runtime.
        // The plain-attention fallback in kernels/attention.cl holds
        // scores[ATTN_MAX_TK] in 64 KiB SLM, so if a caller forces the
        // plain path (features.prefillFlash: false) at a context length
        // above kAttentionMaxTk, the very first request will throw
        // deep in GpuOps::attentionPlainAsync. Catch that combination
        // at startup so the operator sees a clear message during boot,
        // not a stack trace during the first prod-facing request.
        {
            const auto effMaxCtx = e->maxContextTokens();
            // The plain-attention SLM-cap check is L0-only — the HIP
            // backend has no plain-attention path (flash is the only
            // impl there).
#ifdef MIMIRMIND_HAVE_L0
            if (effMaxCtx > ::mimirmind::compute::l0::GpuOps::kAttentionMaxTk
                && !cfg.features.flashPrefill) {
                const std::string msg =
                    "serve: model '" + m.id + "' has effective "
                    "runtime.maxContextTokens=" + std::to_string(effMaxCtx) +
                    " > kAttentionMaxTk=" +
                    std::to_string(
                        ::mimirmind::compute::l0::GpuOps::kAttentionMaxTk) +
                    " while features.prefillFlash=false — the "
                    "plain-attention fallback cannot hold "
                    "scores[ATTN_MAX_TK] in SLM at that context "
                    "length. Set features.prefillFlash=true (default) "
                    "OR reduce runtime.maxContextTokens below " +
                    std::to_string(
                        ::mimirmind::compute::l0::GpuOps::kAttentionMaxTk) + ".";
                MM_LOG_ERROR("main", "{}", msg);
                std::cerr << msg << "\n";
                return 2;
            }
#else
            (void)effMaxCtx;
#endif
            // Informational warn — long context + wide KV storage
            // pressures a 24 GiB DRAM host running Gemma 4 26B-A4B
            // weights (~22 GiB) alongside the KV cache. Rough per-token
            // KV size at F32 is ~430 KiB across all 30 layers; Q8_0 is
            // ~4× smaller. This is a warning, not an error — smaller
            // architectures (E4B / dense 4B) fit F32 KV comfortably.
            if (effMaxCtx > 24576
                && e->kvDtype() == ::mimirmind::runtime::KvDtype::F32) {
                MM_LOG_WARN("main",
                            "runtime.maxContextTokens={} for model '{}' "
                            "with kvDtype=f32: the KV cache will consume "
                            "several GiB on Gemma-4-class geometries. "
                            "Consider kvDtype=q8_0 or kvDtype=fp16 on "
                            "shared-24 GiB hosts.",
                            effMaxCtx, m.id);
            }
        }

        const auto d = e->kvDtype();
        const char* dName = (d == ::mimirmind::runtime::KvDtype::FP16 ? "fp16"
                           : d == ::mimirmind::runtime::KvDtype::Q8_0 ? "q8_0"
                                                                      : "f32");
        MM_LOG_INFO("main",
                    "KV cache dtype for '{}': {} (block {} B × {} elem)",
                    m.id, dName,
                    ::mimirmind::runtime::kvBlockBytes(d),
                    ::mimirmind::runtime::kvBlockElements(d));

        ::mimirmind::server::LoadedEngine le{};
        le.id     = m.id;
        le.title  = m.title;
        le.engine = e.get();
        loadedEngines.push_back(std::move(le));
        ownedEngines.push_back(std::move(e));
    }

    if (ownedEngines.empty()) {
        std::cerr << "serve: no model with loadOnStart:true in config.json — "
                     "nothing to serve\n";
        return 2;
    }

    // The default engine drives all the per-process ancillaries below
    // (thermal guard, power monitor, governor, fan, perf-regression).
    // Additional engines share those same monitors transparently — the
    // hooks are stateless getters that any engine's generate() consults.
    const std::string defaultId = cfg.defaultModel.empty()
        ? cfg.defaultModelEntry().id
        : cfg.defaultModel;
    ::mimirmind::runtime::InferenceEngine* defaultEnginePtr = nullptr;
    for (auto& le : loadedEngines) {
        if (le.id == defaultId) { defaultEnginePtr = le.engine; break; }
    }
    if (defaultEnginePtr == nullptr) {
        std::cerr << "serve: defaultModel='" << defaultId
                  << "' has no loadOnStart:true entry\n";
        return 2;
    }
    auto& engine = *defaultEnginePtr;
    // Re-use the effective runtime for the DEFAULT model when reporting
    // to the user later on (preserve_thinking flag).
    const auto effRuntime = cfg.effectiveRuntime(defaultId);

    // M9.11.1 — Optional speculative decoding. Two drafter variants:
    //   * `speculative.drafter == "model"` loads a second, smaller
    //     InferenceEngine and wraps it in `ModelDrafter`. Requires a
    //     vocab-compatible model resolved via `speculative.draft`.
    //   * `speculative.drafter == "ngram"` uses in-context Prompt-Lookup
    //     Decoding — no second model, no vocab check, zero USM cost.
    // Both variants only kick in when `speculative.enabled` is true.
    std::unique_ptr<::mimirmind::runtime::InferenceEngine> draftEngine;
    std::unique_ptr<::mimirmind::runtime::Drafter>         drafter;
    if (cfg.speculative.enabled) {
        using DrafterKind = ::mimirmind::core::config::SpeculativeSettings::Drafter;
        if (cfg.speculative.drafter == DrafterKind::NGram) {
            ::mimirmind::runtime::NGramDrafter::Config nc{};
            nc.minK = static_cast<std::size_t>(cfg.speculative.ngramMinK);
            nc.maxK = static_cast<std::size_t>(cfg.speculative.ngramMaxK);
            drafter = std::make_unique<::mimirmind::runtime::NGramDrafter>(nc);
            MM_LOG_INFO("main",
                        "serve: speculative decoding ready — "
                        "drafter=ngram minK={} maxK={}",
                        nc.minK, nc.maxK);
        } else if (!cfg.speculative.draft.empty()) {
            std::string draftPath;
            try {
                draftPath = cfg.model(cfg.speculative.draft).path;
            } catch (const std::exception& e) {
                MM_LOG_WARN("main",
                            "serve: speculative.draft='{}' unresolved ({}) — "
                            "speculative decoding disabled",
                            cfg.speculative.draft, e.what());
            }
            if (!draftPath.empty()) {
                MM_LOG_INFO("main",
                            "serve: loading draft model '{}'", draftPath);
                try {
                    draftEngine = std::make_unique<::mimirmind::runtime::InferenceEngine>(cfg);
                    draftEngine->loadModel(draftPath);

                    // Vocab compatibility. Modified rejection sampling
                    // only works when draft token-id N and target token-
                    // id N mean the same subword. vocabSize alone
                    // doesn't guarantee it, but a mismatch there is a
                    // hard disqualification. bos/eos must match too
                    // because we replay the same prompt-id stream
                    // through both engines.
                    const auto& tTok = engine.tokenizer();
                    const auto& dTok = draftEngine->tokenizer();
                    const bool sizeMatch = tTok.vocabSize() == dTok.vocabSize();
                    const bool bosMatch  = tTok.bosId()     == dTok.bosId();
                    const bool eosMatch  = tTok.eosId()     == dTok.eosId();
                    if (!sizeMatch || !bosMatch || !eosMatch) {
                        MM_LOG_WARN("main",
                                    "serve: draft model vocab incompatible with "
                                    "target — disabling speculative decoding. "
                                    "target(vocab={}, bos={}, eos={}) vs "
                                    "draft(vocab={}, bos={}, eos={})",
                                    tTok.vocabSize(), tTok.bosId(), tTok.eosId(),
                                    dTok.vocabSize(), dTok.bosId(), dTok.eosId());
                        draftEngine.reset();
                    } else {
                        drafter = std::make_unique<::mimirmind::runtime::ModelDrafter>(*draftEngine);
                        MM_LOG_INFO("main",
                                    "serve: speculative decoding ready — "
                                    "drafter=model target arch={} d_model={}, "
                                    "draft arch={} d_model={} "
                                    "(shared vocab_size={}, bos={}, eos={})",
                                    engine.config().architecture,
                                    engine.config().embeddingLength,
                                    draftEngine->config().architecture,
                                    draftEngine->config().embeddingLength,
                                    tTok.vocabSize(), tTok.bosId(), tTok.eosId());
                    }
                } catch (const std::exception& e) {
                    MM_LOG_WARN("main",
                                "serve: draft model load failed ({}) — "
                                "speculative decoding disabled", e.what());
                    draftEngine.reset();
                }
            }
        } else {
            MM_LOG_WARN("main",
                        "serve: speculative.enabled=true with drafter='model' "
                        "but speculative.draft is empty — speculative "
                        "decoding disabled");
        }
    }

    ::mimirmind::server::ServerConfig scfg{};
    scfg.host    = "0.0.0.0";
    scfg.port    = args.port.value_or(static_cast<std::uint16_t>(cfg.server.port));
    // defaultModelId in the ServerConfig picks the fallback engine when a
    // request omits `model`. Must match one of the loaded engine ids —
    // computed above as `defaultId`.
    scfg.modelId = defaultId;
    scfg.preserveThinking = effRuntime.preserveThinking.value_or(false);
    scfg.speculative.enabled  = cfg.speculative.enabled;
    scfg.speculative.draftN   = static_cast<std::size_t>(cfg.speculative.n);
    scfg.speculativeTargetId  = cfg.speculative.target;

    // Thermal profile lives inline in config.json under governor.thermal.
    // Empty `name` means "no profile" and the guard runs unprotected.
    const bool hasThermalProfile = !cfg.governor.thermal.name.empty() ||
                                   cfg.governor.thermal.hasPackageLimits();

    std::unique_ptr<::mimirmind::runtime::SystemMonitor> monitor;
    std::unique_ptr<::mimirmind::runtime::ThermalGuard>  guard;
    // In attached mode Munin drives every sysfs-WRITE regulator —
    // GpuClockGovernor + FanController. Per M-Munin ADR "Governor —
    // Sonderregel" the worker MUST NOT install those. SystemMonitor
    // and ThermalGuard are read-only (sensor read + local pacing
    // decision), so the worker installs them in both modes. That
    // keeps /v1/system/status (package temp, RAM, throttle state) alive
    // for the pegenaut dashboard and lets each worker back off decode
    // based on its own thermal reading, which is belt-and-suspenders
    // to Munin's authoritative clock cap.
    if (hasThermalProfile) {
        const ::mimirmind::runtime::ThermalProfile& profile = cfg.governor.thermal;
        try {
            monitor = std::make_unique<::mimirmind::runtime::SystemMonitor>(
                /*requirePackageTemp=*/profile.hasPackageLimits(),
                /*requireRam=*/        false);
        } catch (const std::exception& e) {
            if (attachedMode) {
                // Non-fatal in attached mode: Munin still runs its own
                // regulators. Losing the local telemetry hurts the
                // dashboard but should not refuse the worker boot.
                std::cerr << "serve: attached mode — SystemMonitor sensor "
                             "probe failed (" << e.what() << "); "
                             "continuing without local thermal telemetry\n";
            } else {
                std::cerr << "serve: profile '" << profile.name
                          << "' requires sensors the host does not expose: "
                          << e.what() << "\n";
                return 1;
            }
        }
        if (monitor) {
            guard = std::make_unique<::mimirmind::runtime::ThermalGuard>(
                profile, *monitor);
            engine.setThermalGuard(guard.get());
        }
        if (attachedMode) {
            MM_LOG_INFO("main",
                        "serve: attached mode — SystemMonitor + ThermalGuard "
                        "installed (read-only); GpuClockGovernor / "
                        "FanController skipped (Munin owns the sysfs writes)");
        }
    }
    if (!attachedMode && hasThermalProfile) {
        const ::mimirmind::runtime::ThermalProfile& profile = cfg.governor.thermal;

        // GPU clock governor lives in the same profile (field
        // gpu_target_temp_c). If present AND the iGPU sysfs is
        // writable, we install it. Otherwise we move on without one —
        // the per-token thermal pace still runs as a safety net.
        //
        // `governor.gpuClockPin` pins the software cap for the whole
        // session and suppresses the P-controller tick. Meant for
        // perf-bench runs where the M9.6.5 asymmetric gains would
        // otherwise clock down aggressively and confound the
        // measurement. Package thermal safety still runs via the
        // ThermalGuard admission check + per-token pace. Do NOT ship
        // this to sustained workloads on a passively-cooled chassis.
        //
        // Accepted values (from config.json):
        //   "rp0"            → hardware max (RP0)
        //   "rpn"            → hardware min (RPn, ~800 MHz on Xe-LPG)
        //   "<MHz integer>"  → arbitrary cap, clamped to [RPn, RP0]
        //   null / "0" / "off" / "false" / "no" → no pin (governor ticks as normal)
        enum class ClockPinIntent { None, Rp0, Rpn, Numeric };
        struct ClockPinRequest {
            ClockPinIntent intent = ClockPinIntent::None;
            std::uint32_t  mhz    = 0;
            std::string    rawEnv;
            bool           malformed = false;
        };
        const auto parseClockPin = [](std::string_view sv) {
            ClockPinRequest req;
            if (sv.empty()) {
                return req;
            }
            if (sv == "0" || sv == "off" || sv == "false" || sv == "no") {
                return req; // treat as unset
            }
            req.rawEnv = std::string{sv};
            if (sv == "rp0" || sv == "RP0") {
                req.intent = ClockPinIntent::Rp0;
                return req;
            }
            if (sv == "rpn" || sv == "RPn" || sv == "RPN") {
                req.intent = ClockPinIntent::Rpn;
                return req;
            }
            char* end = nullptr;
            const std::string zSv{sv};
            const unsigned long v = std::strtoul(zSv.c_str(), &end, 10);
            if (end != zSv.c_str() && *end == '\0' && v > 0 && v < 100000) {
                req.intent = ClockPinIntent::Numeric;
                req.mhz    = static_cast<std::uint32_t>(v);
                return req;
            }
            req.malformed = true;
            return req;
        };

        static std::unique_ptr<::mimirmind::runtime::GpuClockGovernor> governor;
        const auto pinReq = parseClockPin(
            cfg.governor.gpuClockPin.value_or(""));
        const bool clockPinRequested = pinReq.intent != ClockPinIntent::None;

        if (profile.hasGpuClockTarget()) {
            governor = std::make_unique<::mimirmind::runtime::GpuClockGovernor>();
            governor->setTargetTempC(*profile.gpu_target_temp_c);
            if (!governor->available()) {
                MM_LOG_WARN("main",
                            "thermal profile asks for GPU clock governor "
                            "(gpu_target_temp_c={:.1f}) but it is not "
                            "available: {}",
                            *profile.gpu_target_temp_c,
                            governor->unavailableReason());
                governor.reset();
            } else if (clockPinRequested) {
                std::uint32_t   requestedMhz = 0;
                std::string_view intentName  = "";
                switch (pinReq.intent) {
                    case ClockPinIntent::Rp0:
                        requestedMhz = governor->rp0Mhz();
                        intentName   = "rp0";
                        break;
                    case ClockPinIntent::Rpn:
                        requestedMhz = governor->rpnMhz();
                        intentName   = "rpn";
                        break;
                    case ClockPinIntent::Numeric:
                        requestedMhz = pinReq.mhz;
                        intentName   = "numeric";
                        break;
                    case ClockPinIntent::None:
                        break;
                }
                const auto pinned = governor->pin(
                    requestedMhz, intentName, pinReq.rawEnv);
                MM_LOG_WARN("main",
                            "governor.gpuClockPin={} — cap pinned to "
                            "{} MHz (intent={}, envelope [{},{}]). "
                            "P-controller tick suppressed. Bench mode. "
                            "Thermal safety still via ThermalGuard.",
                            pinReq.rawEnv, pinned, intentName,
                            governor->rpnMhz(), governor->rp0Mhz());
                // Install the governor anyway so ApiServer can report
                // the pin state via /system/info + /system/status. The
                // engine's tick loop consults governor->pinned() and
                // skips its adjust call, so the pin survives the run.
                engine.setGpuClockGovernor(governor.get(), monitor.get());
            } else {
                if (pinReq.malformed) {
                    MM_LOG_WARN("main",
                                "governor.gpuClockPin={} not recognised — "
                                "expected rp0 / rpn / <MHz> / 0 / off. "
                                "Installing governor as if unset.",
                                pinReq.rawEnv);
                }
                engine.setGpuClockGovernor(governor.get(), monitor.get());
            }
        } else if (clockPinRequested) {
            MM_LOG_WARN("main",
                        "governor.gpuClockPin={} ignored — thermal "
                        "profile has no gpu_target_temp_c so no governor "
                        "was going to be installed anyway.",
                        pinReq.rawEnv);
        }

        // M9.6.6.0 tick sink. `governor.tickLog:true` gates the sink;
        // `governor.tickLogFile` names the NDJSON output. For one
        // release we still honour `diagnostics.traceDecodeFile` as the
        // path when tickLogFile is unset — that reuse conflates the
        // decode-trace and governor-tick streams and is being retired.
        if (governor != nullptr && cfg.governor.tickLog) {
            std::string tickPath = cfg.governor.tickLogFile;
            bool viaDeprecated  = false;
            if (tickPath.empty() && !cfg.diagnostics.traceDecodeFile.empty()) {
                tickPath      = cfg.diagnostics.traceDecodeFile;
                viaDeprecated = true;
            }
            if (tickPath.empty()) {
                MM_LOG_WARN("main",
                            "governor.tickLog:true but governor.tickLogFile "
                            "is empty — sink stays off. Set "
                            "governor.tickLogFile to a writable path.");
            } else if (governor->setTickLogPath(tickPath)) {
                if (viaDeprecated) {
                    MM_LOG_WARN("main",
                                "GovernorTickSink using deprecated "
                                "diagnostics.traceDecodeFile='{}' as its path "
                                "— move to governor.tickLogFile in config.json "
                                "before the next release.",
                                tickPath);
                }
                MM_LOG_INFO("main",
                            "GovernorTickSink open — writing NDJSON to '{}'",
                            tickPath);
            } else {
                MM_LOG_WARN("main",
                            "governor.tickLog set with path '{}' — "
                            "could not open for append. Sink stays off.",
                            tickPath);
            }
        }
    }

    // Power telemetry — always-on attempt, never fatal. If RAPL is
    // masked (Docker / unprivileged LXC without explicit mount) the
    // monitor reports unavailable and /v1/system/status shows the
    // reason; the engine still runs.
    auto powerMonitor = std::make_unique<::mimirmind::runtime::PowerMonitor>();
    engine.setPowerMonitor(powerMonitor.get());

    // M9.11.b chassis fan controller. Probes /sys/class/hwmon/* at
    // construction; if a writable pwm/pwm_enable pair is found, the
    // engine boosts the fan at the start of each generate() and
    // releases to auto at the end. Original BIOS values captured for
    // RAII restore on process exit.
    //
    // Config knobs:
    //   governor.fan.boost:    false → do not install (kill switch)
    //   governor.fan.pwmBoost: 0-255 override boost target
    //   governor.fan.pwmMin:   0-255 override safety floor
    // Kill switch is checked first so we can disable the whole feature
    // without touching sysfs at all — useful when the BIOS refuses
    // manual mode and hwmon writes are throwing kernel warnings.
    static std::unique_ptr<::mimirmind::runtime::FanController> fanController;
    {
        // Attached-mode workers never touch the fan (see M-Munin ADR
        // Governor-Sonderregel). Munin owns the fan install; the worker
        // just runs generate() and lets Munin cool the chassis.
        const bool disabled = !cfg.governor.fan.boost || attachedMode;
        if (!disabled) {
            fanController = std::make_unique<::mimirmind::runtime::FanController>();
            if (!fanController->available()) {
                MM_LOG_WARN("main",
                            "FanController unavailable — no proactive fan "
                            "boost. Reason: {}",
                            fanController->unavailableReason());
                fanController.reset();
            } else {
                if (const auto v = cfg.governor.fan.pwmBoost;
                    v.has_value() && *v >= 0 && *v <= 255) {
                    fanController->setBoostPwm(
                        static_cast<std::uint8_t>(*v));
                }
                if (const auto v = cfg.governor.fan.pwmMin;
                    v.has_value() && *v >= 0 && *v <= 255) {
                    fanController->setMinSafePwm(
                        static_cast<std::uint8_t>(*v));
                }
                MM_LOG_INFO("main",
                            "FanController ready — chip='{}' pwm='{}' "
                            "fan_input='{}' orig_pwm={} orig_enable={} "
                            "boost={} min_safe={}",
                            fanController->chipName(),
                            fanController->pwmPath(),
                            fanController->fanInputPath(),
                            fanController->originalPwm(),
                            fanController->originalEnableMode(),
                            fanController->boostPwm(),
                            fanController->minSafePwm());
                engine.setFanController(fanController.get());
            }
        }
    }

    // Thermal-safety cross-check: governor.gpuClockPin=rp0 disables
    // the P-controller entirely, so the FanController is the only
    // active thermal regulator during sustained decode. Warn loudly
    // when the operator asks for rp0 without a functioning fan-boost
    // path — this is the exact 2026-07-01 shutdown scenario.
    {
        const std::string_view pinSv =
            cfg.governor.gpuClockPin.has_value()
                ? std::string_view{*cfg.governor.gpuClockPin}
                : std::string_view{};
        const bool  wantsRp0 = (pinSv == "rp0" || pinSv == "RP0");
        const bool fanActive =
            fanController != nullptr && fanController->available();
        if (wantsRp0 && !fanActive) {
            MM_LOG_WARN("main",
                        "governor.gpuClockPin=rp0 is set but the "
                        "FanController is not active — the P-controller "
                        "is disabled AND no proactive cooling is "
                        "installed. Sustained decode on a passively "
                        "cooled chassis can trigger a hardware thermal "
                        "shutdown (see 2026-07-01 incident). Consider "
                        "governor.gpuClockPin=<numeric MHz> as a "
                        "safer bench mode.");
        }
    }

    // In-process perf-regression detector. Feeds off the same per-token
    // wall-time the NDJSON sink already computes, so it costs a couple
    // of doubles per token and one median at end-of-run. Kill-switch:
    // `diagnostics.regressionAlert: false` in config.json skips the
    // installer entirely for the case where the detector itself
    // misbehaves and needs to be silenced without a redeploy.
    std::unique_ptr<::mimirmind::runtime::PerfRegressionDetector> perfDetector;
    {
        const bool disabled = !cfg.diagnostics.regressionAlert;
        if (!disabled) {
            std::string baselinePath;
            if (const char* h = std::getenv("HOME"); h != nullptr && h[0] != '\0') {
                baselinePath = std::string{h} +
                               "/.cache/mimirmind/perf-baseline.json";
            } else {
                baselinePath = "/tmp/mimirmind-perf-baseline.json";
            }
            std::error_code ec;
            std::filesystem::create_directories(
                std::filesystem::path{baselinePath}.parent_path(), ec);
            // create_directories failure is not fatal — the detector
            // logs a warning on the first write and keeps running.
            perfDetector =
                std::make_unique<::mimirmind::runtime::PerfRegressionDetector>(
                    baselinePath);
            engine.setPerfRegressionDetector(perfDetector.get());
        } else {
            MM_LOG_WARN("main",
                        "diagnostics.regressionAlert=false — perf-regression "
                        "detector not installed for this session");
        }
    }

    // Propagate the process-wide ancillary monitors from the default
    // engine to any extras so their generate() paths also honour
    // thermal admission, RAPL joule accounting, fan-boost pre-warm and
    // perf-regression sampling. Governor propagation is intentionally
    // skipped — its per-tick control loop is process-scoped and driven
    // by the default engine; extras would fight for the same GPU cap.
    for (auto& e : ownedEngines) {
        if (e.get() == defaultEnginePtr) continue;
        if (auto* g = engine.thermalGuard())            e->setThermalGuard(g);
        if (auto* p = engine.powerMonitor())            e->setPowerMonitor(p);
        if (auto* d = engine.perfRegressionDetector()) e->setPerfRegressionDetector(d);
        if (auto* fc = engine.fanController())          e->setFanController(fc);
    }

    // M-Cuda.Batch D2e.2 — continuous-batching worker for the default
    // (serving-class) engine. Built only for qwen35moe once the startup
    // BatchCapacityProbe has recommended serving-class (sustainableBatch
    // >= min). Requests to the default engine are then serviced through the
    // batcher's worker thread (multi-tenant continuous batching) rather than
    // the serialised single-session generate() path. Kept alive for the
    // whole server.run() below; its dtor joins the worker on shutdown.
    std::unique_ptr<::mimirmind::runtime::serving::ContinuousBatcher> batcher;
    if (engine.config().architecture == "qwen35moe" &&
        engine.servingClassEnabled()) {
        const std::size_t maxBatch =
            std::max<std::size_t>(1, engine.batchCapacity().sustainableBatch);
        const std::size_t maxContext = engine.maxContextTokens();
        try {
            batcher = std::make_unique<
                ::mimirmind::runtime::serving::ContinuousBatcher>(
                engine, maxBatch, maxContext, engine.tokenizer().eosId());
            scfg.batcher = batcher.get();
            MM_LOG_INFO("main",
                        "serve: continuous batcher ENABLED for default engine "
                        "'{}' (maxBatch={} maxContext={})",
                        defaultId, maxBatch, maxContext);
        } catch (const std::exception& e) {
            MM_LOG_WARN("main",
                        "serve: continuous batcher init failed ({}); falling "
                        "back to single-session generate()", e.what());
            batcher.reset();
            scfg.batcher = nullptr;
        }
    }

    ::mimirmind::server::ApiServer server{std::move(loadedEngines), scfg,
                                          drafter.get()};

    g_runningServer.store(&server, std::memory_order_release);
    std::signal(SIGINT,  signalStop);
    std::signal(SIGTERM, signalStop);

    std::cout << "\n[M7d/M7e] OpenAI-compatible HTTP API listening on "
              << scfg.host << ":" << scfg.port
              << "\n  GET  /health\n"
                 "  GET  /v1/models\n"
                 "  GET  /v1/system/info\n"
                 "  GET  /v1/system/status\n"
                 "  POST /v1/chat/completions  (stream=true supported)\n"
                 "  model id:           " << scfg.modelId << "\n"
                 "  preserve-thinking:  "
              << (scfg.preserveThinking ? "on (raw deltas, KV-cache friendly)"
                                        : "off (cleaned text, channel-wrapper stripped)")
              << "\n  thermal profile:    ";
    if (guard) {
        std::cout << "'" << guard->profile().name
                  << "' (package=" << monitor->packageTempSource() << ")";
    } else {
        std::cout << "\033[1;33mNOT CONFIGURED — engine is unprotected\033[0m";
    }
    std::cout << "\n  power telemetry:    ";
    if (powerMonitor->available()) {
        std::cout << "on (" << powerMonitor->domainNames().size()
                  << " RAPL domain(s))";
    } else {
        std::cout << "off (" << powerMonitor->unavailableReason() << ")";
    }
    std::cout << "\n  gpu clock governor: ";
    if (auto* gov = engine.gpuClockGovernor()) {
        std::cout << "on (target=" << gov->targetTempC()
                  << "°C, " << gov->rpnMhz() << ".."
                  << gov->rp0Mhz() << " MHz on "
                  << gov->cardPath() << ")";
    } else {
        std::cout << "off";
    }
    std::cout << "\n  perf regression:    ";
    if (auto* det = engine.perfRegressionDetector()) {
        std::cout << "on (baseline=" << det->baselineSampleCount()
                  << " samples, threshold="
                  << ::mimirmind::runtime::PerfRegressionDetector::kAlertThreshold
                  << "x)";
    } else {
        std::cout << "off (diagnostics.regressionAlert=false)";
    }
    std::cout << "\n  spec decoding:      ";
    if (drafter != nullptr) {
        std::cout << "ready (drafter=" << drafter->kind();
        if (draftEngine != nullptr) {
            std::cout << ", draft arch=" << draftEngine->config().architecture
                      << ", d_model="   << draftEngine->config().embeddingLength;
        }
        std::cout << ")";
    } else if (cfg.speculative.enabled) {
        std::cout << "disabled (draft load or vocab check failed — see log)";
    } else {
        std::cout << "off (set speculative.enabled=true in config.json to enable)";
    }
    std::cout << "\n  max context tokens: " << engine.maxContextTokens()
              << "\n  Ctrl-C to stop.\n";
    std::cout.flush();

    if (!guard) {
        MM_LOG_WARN("main",
                    "serve: no thermal profile configured. The engine will "
                    "not throttle decode on temperature/RAM limits. Fill "
                    "the governor.thermal section of config.json to "
                    "protect the host.");
    }

    try {
        server.run();
    } catch (const std::exception& e) {
        g_runningServer.store(nullptr, std::memory_order_release);
        MM_LOG_ERROR("main", "serve: {}", e.what());
        std::cerr << "serve failed: " << e.what() << "\n";
        return 1;
    }
    g_runningServer.store(nullptr, std::memory_order_release);

    MM_LOG_INFO("main", "serve: stopped cleanly");
    return 0;
}

} // namespace mimirmind::cli