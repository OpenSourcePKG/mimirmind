// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "mimirmind/ServerBootstrap.hpp"

#include "mimirmind/CliArgs.hpp"
#include "mimirmind/ServingBenchModes.hpp"

#ifdef MIMIRMIND_HAVE_L0
#include "compute/l0/GpuOps.hpp"
#endif
#include "core/backend/BackendPool.hpp"
#include "core/backend/BackendRegistry.hpp"
#include "core/config/Config.hpp"
#include "core/ipc/MuninClient.hpp"
#include "core/log/Log.hpp"
#include "model/Tokenizer.hpp"
#include "runtime/ComputeStack.hpp"
#include "runtime/InferenceEngine.hpp"
#include "runtime/KvCache.hpp"
#include "runtime/audio/AudioEngine.hpp"
#include "runtime/audio/SpeakEngine.hpp"
#include "runtime/encoder/DecideEngine.hpp"
#include "runtime/encoder/EmbedEngine.hpp"
#include "runtime/encoder/RerankEngine.hpp"
#include "runtime/nvfp4/ModelFormatResolver.hpp"
#include "runtime/perf/PerfRegressionDetector.hpp"
#include "runtime/serving/ContinuousBatcher.hpp"
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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace mimirmind::cli {

ServerBootstrap::SpeculativeSetup ServerBootstrap::buildSpeculative(
    const core::config::Config&    cfg,
    const runtime::InferenceEngine& targetEngine) {
    // M9.11.1 — Optional speculative decoding. Two drafter variants:
    //   * `speculative.drafter == "model"` loads a second, smaller
    //     InferenceEngine and wraps it in `ModelDrafter`. Requires a
    //     vocab-compatible model resolved via `speculative.draft`.
    //   * `speculative.drafter == "ngram"` uses in-context Prompt-Lookup
    //     Decoding — no second model, no vocab check, zero USM cost.
    // Both variants only kick in when `speculative.enabled` is true.
    SpeculativeSetup setup{};
    if (!cfg.speculative.enabled) {
        return setup;
    }

    using DrafterKind = ::mimirmind::core::config::SpeculativeSettings::Drafter;
    if (cfg.speculative.drafter == DrafterKind::NGram) {
        ::mimirmind::runtime::NGramDrafter::Config nc{};
        nc.minK = static_cast<std::size_t>(cfg.speculative.ngramMinK);
        nc.maxK = static_cast<std::size_t>(cfg.speculative.ngramMaxK);
        setup.drafter = std::make_unique<::mimirmind::runtime::NGramDrafter>(nc);
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
                setup.draftEngine =
                    std::make_unique<::mimirmind::runtime::InferenceEngine>(cfg);
                setup.draftEngine->loadModel(draftPath);

                // Vocab compatibility. Modified rejection sampling
                // only works when draft token-id N and target token-
                // id N mean the same subword. vocabSize alone
                // doesn't guarantee it, but a mismatch there is a
                // hard disqualification. bos/eos must match too
                // because we replay the same prompt-id stream
                // through both engines.
                const auto& tTok = targetEngine.tokenizer();
                const auto& dTok = setup.draftEngine->tokenizer();
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
                    setup.draftEngine.reset();
                } else {
                    setup.drafter =
                        std::make_unique<::mimirmind::runtime::ModelDrafter>(
                            *setup.draftEngine);
                    MM_LOG_INFO("main",
                                "serve: speculative decoding ready — "
                                "drafter=model target arch={} d_model={}, "
                                "draft arch={} d_model={} "
                                "(shared vocab_size={}, bos={}, eos={})",
                                targetEngine.config().architecture,
                                targetEngine.config().embeddingLength,
                                setup.draftEngine->config().architecture,
                                setup.draftEngine->config().embeddingLength,
                                tTok.vocabSize(), tTok.bosId(), tTok.eosId());
                }
            } catch (const std::exception& e) {
                MM_LOG_WARN("main",
                            "serve: draft model load failed ({}) — "
                            "speculative decoding disabled", e.what());
                setup.draftEngine.reset();
            }
        }
    } else {
        MM_LOG_WARN("main",
                    "serve: speculative.enabled=true with drafter='model' "
                    "but speculative.draft is empty — speculative "
                    "decoding disabled");
    }
    return setup;
}

ServerBootstrap::Ancillaries ServerBootstrap::wireThermalGovernorFan(
    runtime::InferenceEngine&   engine,
    const core::config::Config& cfg,
    bool                        attachedMode) {
    Ancillaries a{};

    // Thermal profile lives inline in config.json under governor.thermal.
    // Empty `name` means "no profile" and the guard runs unprotected.
    const bool hasThermalProfile = !cfg.governor.thermal.name.empty() ||
                                   cfg.governor.thermal.hasPackageLimits();

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
            a.monitor = std::make_unique<::mimirmind::runtime::SystemMonitor>(
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
                a.fatalExitCode = 1;
                return a;
            }
        }
        if (a.monitor) {
            a.guard = std::make_unique<::mimirmind::runtime::ThermalGuard>(
                profile, *a.monitor);
            engine.setThermalGuard(a.guard.get());
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
                engine.setGpuClockGovernor(governor.get(), a.monitor.get());
            } else {
                if (pinReq.malformed) {
                    MM_LOG_WARN("main",
                                "governor.gpuClockPin={} not recognised — "
                                "expected rp0 / rpn / <MHz> / 0 / off. "
                                "Installing governor as if unset.",
                                pinReq.rawEnv);
                }
                engine.setGpuClockGovernor(governor.get(), a.monitor.get());
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
    a.powerMonitor = std::make_unique<::mimirmind::runtime::PowerMonitor>();
    engine.setPowerMonitor(a.powerMonitor.get());

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
            a.perfDetector =
                std::make_unique<::mimirmind::runtime::PerfRegressionDetector>(
                    baselinePath);
            engine.setPerfRegressionDetector(a.perfDetector.get());
        } else {
            MM_LOG_WARN("main",
                        "diagnostics.regressionAlert=false — perf-regression "
                        "detector not installed for this session");
        }
    }

    return a;
}

std::unique_ptr<::mimirmind::runtime::serving::ContinuousBatcher>
ServerBootstrap::buildDefaultBatcher(runtime::InferenceEngine&    engine,
                                     const core::config::Config&  cfg,
                                     const std::string&           defaultId) {
    // 5.27.11.2: qwen4_exp re-enabled for the continuous batcher — the paged
    // stepServing (decode) and prefillSlot (single-slot prefill) forward now
    // carry per-slot PLE n-gram + the HC stream collapse, and the batcher
    // forces single-slot prefill for qwen4_exp (no ragged varlen). conc>1
    // batched forward is slot-correct + coherent (5.27.11.2).
    if (!((engine.config().architecture == "qwen35moe" ||
           engine.config().architecture == "qwen4_exp" ||
           engine.supportsBatchedDecode()) &&
          engine.servingClassEnabled())) {
        return nullptr;
    }

    std::size_t maxBatch =
        std::max<std::size_t>(1, engine.batchCapacity().sustainableBatch);
    // Operator override of the batcher's slot count. The v1 BatchCapacity probe
    // is a coarse bandwidth-tier proxy (Xe-LPG ~70 GB/s pins it to 1) — not a
    // real device-memory calc — so an operator who knows the model fits more
    // concurrent slots can set MIMIRMIND_SERVING_MAXBATCH.
    if (const char* mb = std::getenv("MIMIRMIND_SERVING_MAXBATCH")) {
        const long v = std::atol(mb);
        if (v > 0) {
            maxBatch = static_cast<std::size_t>(v);
            MM_LOG_INFO("main",
                        "serve: MIMIRMIND_SERVING_MAXBATCH override — "
                        "batcher maxBatch={} (probe said {})",
                        maxBatch, engine.batchCapacity().sustainableBatch);
        }
    }
    const std::size_t maxContext = engine.maxContextTokens();
    // Total accepted-but-unfinished cap (running + queued). Beyond it the
    // batcher sheds load with a 503 instead of an unbounded queue.
    const std::size_t maxInflight = cfg.serving.maxActiveRequests;
    // Per-tenant fairness cap (0 = off). Keeps one API-key tenant from eating
    // the whole maxInflight budget and starving co-tenants.
    const std::size_t maxInflightPerTenant =
        cfg.serving.maxActiveRequestsPerTenant;
    try {
        auto batcher = std::make_unique<
            ::mimirmind::runtime::serving::ContinuousBatcher>(
            engine, maxBatch, maxContext, engine.tokenizer().eosId(),
            maxInflight, maxInflightPerTenant);
        MM_LOG_INFO("main",
                    "serve: continuous batcher ENABLED for default engine "
                    "'{}' (maxBatch={} maxContext={} maxInflight={} "
                    "maxInflightPerTenant={})",
                    defaultId, maxBatch, maxContext,
                    batcher->maxInflight(),
                    batcher->maxInflightPerTenant());
        return batcher;
    } catch (const std::exception& e) {
        MM_LOG_WARN("main",
                    "serve: continuous batcher init failed ({}); falling "
                    "back to single-session generate()", e.what());
        return nullptr;
    }
}

void ServerBootstrap::loadModels(LoadedModels&                 out,
                                 const core::config::Config&   cfg,
                                 const cli::CliArgs&           args,
                                 core::backend::BackendPool&   backendPool,
                                 const AttachEngineFn&         attachEngine,
                                 const std::string&            defaultId,
                                 bool                          attachedMode) {
    // Probed once before the per-model attach loop so a dead Munin does not
    // manifest as N confusing per-model attach errors. The whole Munin/attach
    // chain is L0/CUDA-only; a build without either already refused --attach.
#if defined(MIMIRMIND_HAVE_L0) || defined(MIMIRMIND_HAVE_CUDA)
    if (attachedMode) {
        MM_LOG_INFO("main",
                    "serve: attached mode — probing Munin at '{}'",
                    args.attachSocket);
        auto hz = ::mimirmind::core::ipc::MuninClient::healthz(args.attachSocket);
        if (!hz) {
            std::cerr << "serve: Munin healthz failed at '"
                      << args.attachSocket << "': " << hz.error() << "\n";
            out.exitCode = 2;
            return;
        }
        if (hz->governorOwner != "munin") {
            std::cerr << "serve: refusing to attach — Munin reports "
                         "governor_owner='" << hz->governorOwner
                      << "', expected 'munin'. Standalone-worker "
                         "handoff back to Munin is not part of "
                         "Schritt 8-minimal (M-Munin ADR).\n";
            out.exitCode = 2;
            return;
        }
        MM_LOG_INFO("main",
                    "serve: Munin healthz ok — pid={} models={} owner={}",
                    hz->pid, hz->models.size(), hz->governorOwner);
        for (const auto& mm : hz->models) {
            MM_LOG_INFO("main",
                        "  munin-model id='{}' fingerprint='{}' bytes={}",
                        mm.id, mm.fingerprint, mm.totalBytes);
        }
    }
#else
    (void)args;
    (void)attachedMode;
#endif

    // Per-model runtime overrides (context / KV dtype), applied AFTER load so
    // they can inspect the loaded model state.
    auto applyRuntimeOverrides =
        [&cfg](::mimirmind::runtime::InferenceEngine& e,
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
        if (cfg.serving.kvDtype.has_value() && !cfg.serving.kvDtype->empty()) {
            const std::string_view v{*cfg.serving.kvDtype};
            if (v == "fp8")       e.setServingKvDtype(::mimirmind::runtime::KvDtype::FP8_E4M3);
            else if (v == "fp16") e.setServingKvDtype(::mimirmind::runtime::KvDtype::FP16);
            else if (v == "f32")  e.setServingKvDtype(::mimirmind::runtime::KvDtype::F32);
            MM_LOG_INFO("main", "serving.kvDtype='{}' → paged serving-pool KV tier", v);
        }
    };

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
            out.exitCode = 2;
            return;
        }
        // Rerank models take a separate path: a dense F32 cross-encoder behind
        // /v1/rerank, with its own isolated compute stack.
        if (m.task == ::mimirmind::core::config::ModelTask::Rerank) {
            try {
                out.ownedRerankStacks.push_back(
                    ::mimirmind::runtime::makeComputeStack(cfg, engineKind));
                auto& stk = out.ownedRerankStacks.back();
                auto re = std::make_unique<
                    ::mimirmind::runtime::encoder::RerankEngine>(
                    m.path, *stk.ops, *stk.matmul);
                ::mimirmind::server::LoadedReranker lr{};
                lr.id     = m.id;
                lr.title  = m.title;
                lr.engine = re.get();
                out.loadedRerankers.push_back(std::move(lr));
                out.ownedRerankers.push_back(std::move(re));
                MM_LOG_INFO("main",
                            "serve: loaded rerank model '{}' (id='{}')",
                            m.path, m.id);
            } catch (const std::exception& x) {
                std::cerr << "serve: rerank model '" << m.id
                          << "' load failed: " << x.what() << "\n";
                out.exitCode = 2;
                return;
            }
            continue;
        }

        // Embed models: a dense F32 bi-encoder behind /v1/embeddings.
        if (m.task == ::mimirmind::core::config::ModelTask::Embed) {
            try {
                out.ownedEmbedStacks.push_back(
                    ::mimirmind::runtime::makeComputeStack(cfg, engineKind));
                auto& stk = out.ownedEmbedStacks.back();
                auto ee = std::make_unique<
                    ::mimirmind::runtime::encoder::EmbedEngine>(
                    m.path, *stk.ops, *stk.matmul);
                ::mimirmind::server::LoadedEmbedder le{};
                le.id     = m.id;
                le.title  = m.title;
                le.engine = ee.get();
                out.loadedEmbedders.push_back(std::move(le));
                out.ownedEmbedders.push_back(std::move(ee));
                MM_LOG_INFO("main",
                            "serve: loaded embed model '{}' (id='{}')",
                            m.path, m.id);
            } catch (const std::exception& x) {
                std::cerr << "serve: embed model '" << m.id
                          << "' load failed: " << x.what() << "\n";
                out.exitCode = 2;
                return;
            }
            continue;
        }

        // Decide models (8.23 System-One): bge-m3 encoder + trained decision
        // heads behind /v1/decide.
        if (m.task == ::mimirmind::core::config::ModelTask::Decide) {
            try {
                out.ownedDecideStacks.push_back(
                    ::mimirmind::runtime::makeComputeStack(cfg, engineKind));
                auto& stk = out.ownedDecideStacks.back();
                auto de = std::make_unique<
                    ::mimirmind::runtime::encoder::DecideEngine>(
                    m.path, m.decideHeadsDir, *stk.ops, *stk.matmul);
                ::mimirmind::server::LoadedDecider ld{};
                ld.id     = m.id;
                ld.title  = m.title;
                ld.engine = de.get();
                MM_LOG_INFO("main",
                            "serve: loaded decide model '{}' (id='{}', {} heads)",
                            m.path, m.id, de->headCount());
                out.loadedDeciders.push_back(std::move(ld));
                out.ownedDeciders.push_back(std::move(de));
            } catch (const std::exception& x) {
                std::cerr << "serve: decide model '" << m.id
                          << "' load failed: " << x.what() << "\n";
                out.exitCode = 2;
                return;
            }
            continue;
        }

        // Transcribe models: a Whisper-class ASR checkpoint behind
        // /v1/audio/transcriptions.
        if (m.task == ::mimirmind::core::config::ModelTask::Transcribe) {
            try {
                out.ownedTranscribeStacks.push_back(
                    ::mimirmind::runtime::makeComputeStack(cfg, engineKind));
                auto& stk = out.ownedTranscribeStacks.back();
                auto ae = std::make_unique<
                    ::mimirmind::runtime::audio::AudioEngine>(
                    m.path, *stk.ops, *stk.matmul);
                ::mimirmind::server::LoadedTranscriber lt{};
                lt.id     = m.id;
                lt.title  = m.title;
                lt.engine = ae.get();
                out.loadedTranscribers.push_back(std::move(lt));
                out.ownedTranscribers.push_back(std::move(ae));
                MM_LOG_INFO("main",
                            "serve: loaded transcribe model '{}' (id='{}')",
                            m.path, m.id);
            } catch (const std::exception& x) {
                std::cerr << "serve: transcribe model '" << m.id
                          << "' load failed: " << x.what() << "\n";
                out.exitCode = 2;
                return;
            }
            continue;
        }

        // Speak (Orpheus TTS) models: an acoustic InferenceEngine backbone plus
        // the SNAC codec decoder behind /v1/audio/speech.
        if (m.task == ::mimirmind::core::config::ModelTask::Speak) {
            if (m.codecPath.empty()) {
                std::cerr << "serve: speak model '" << m.id
                          << "' requires a 'codec' path (converted SNAC "
                             "safetensors)\n";
                out.exitCode = 2;
                return;
            }
            try {
                auto e = std::make_unique<::mimirmind::runtime::InferenceEngine>(
                    cfg, engineKind);
                if (attachedMode) {
                    auto ka = attachEngine(*e, m);
                    if (!ka) {
                        out.exitCode = 2;
                        return;
                    }
                    out.attachedKeepAlive.push_back(std::move(ka->first));
                    out.attachedKeepAlive.push_back(std::move(ka->second));
                } else {
                    e->setModelIdHint(m.id);   // for the per-model profile overlay
                    if (runtime::nvfp4::resolveModelFormat(m.format, m.path)
                        == core::config::ModelFormat::Nvfp4) {
                        e->loadModelNvfp4(m.path, m.tokenizerGguf);
                    } else {
                        e->loadModel(m.path);
                    }
                }
                auto se = std::make_unique<
                    ::mimirmind::runtime::audio::SpeakEngine>(*e, m.codecPath);
                ::mimirmind::server::LoadedSpeaker ls{};
                ls.id     = m.id;
                ls.title  = m.title;
                ls.engine = se.get();
                out.loadedSpeakers.push_back(std::move(ls));
                out.ownedSpeakers.push_back(std::move(se));
                out.ownedSpeakBackbones.push_back(std::move(e));
                MM_LOG_INFO("main",
                            "serve: loaded speak model '{}' (id='{}', codec='{}')",
                            m.path, m.id, m.codecPath);
            } catch (const std::exception& x) {
                std::cerr << "serve: speak model '" << m.id
                          << "' load failed: " << x.what() << "\n";
                out.exitCode = 2;
                return;
            }
            continue;
        }

        // M-Munin.3 (full): non-default chat models defer to the pool instead of
        // eager-loading here.
        if (attachedMode && cfg.serving.modelPoolCapacity > 0 &&
            m.id != defaultId) {
            out.poolChatModels.push_back(m);
            MM_LOG_INFO("main",
                        "serve: model '{}' registered with the pool "
                        "(lazy materialize, capacity={})",
                        m.id, cfg.serving.modelPoolCapacity);
            continue;
        }

        auto e = std::make_unique<::mimirmind::runtime::InferenceEngine>(
            cfg, engineKind);

        if (attachedMode) {
            MM_LOG_INFO("main",
                        "serve: attaching to Munin for model '{}' "
                        "(local header from '{}')", m.id, m.path);
            auto ka = attachEngine(*e, m);
            if (!ka) {
                out.exitCode = 2;
                return;
            }
            out.attachedKeepAlive.push_back(std::move(ka->first));
            out.attachedKeepAlive.push_back(std::move(ka->second));
        } else {
            e->setModelIdHint(m.id);   // for the per-model profile overlay
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
        if (arch != "qwen2" && arch != "llama" && arch != "gemma4" &&
            arch != "qwen35moe" && arch != "qwen4_exp") {
            const std::string msg =
                "serve: architecture '" + arch + "' (model id '" + m.id +
                "') is not implemented yet. See "
                "Memory/mimirmind/research/m8-gemma4-staging.md.";
            MM_LOG_ERROR("main", "{}", msg);
            std::cerr << msg << "\n";
            out.exitCode = 2;
            return;
        }
        // setKvDtype + setMaxContextTokens inspect loaded model state, so apply
        // the per-model runtime overrides AFTER loadModel.
        applyRuntimeOverrides(*e, cfg.effectiveRuntime(m.id));
        if (const auto benchExit =
                ServingBenchModes::maybeRun(*e, arch, cfg)) {
            out.exitCode = *benchExit;
            return;
        }
        // M9.8b — cross-block sanity check on the effective runtime. The
        // plain-attention fallback holds scores[ATTN_MAX_TK] in 64 KiB SLM, so a
        // forced plain path (features.prefillFlash: false) above kAttentionMaxTk
        // would throw deep in GpuOps::attentionPlainAsync on the first request.
        {
            const auto effMaxCtx = e->maxContextTokens();
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
                out.exitCode = 2;
                return;
            }
#else
            (void)effMaxCtx;
#endif
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
                           : d == ::mimirmind::runtime::KvDtype::FP8_E4M3 ? "fp8_e4m3"
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
        out.loadedEngines.push_back(std::move(le));
        out.ownedEngines.push_back(std::move(e));
    }
}

} // namespace mimirmind::cli
