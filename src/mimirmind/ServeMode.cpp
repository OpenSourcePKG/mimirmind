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
    if (!attachedMode) {
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
            MM_LOG_INFO("main", "serve: loading model '{}' (id='{}')",
                        m.path, m.id);
            e->loadModel(m.path);
        }

        const auto& arch = e->config().architecture;
        if (arch != "qwen2" && arch != "gemma4") {
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