// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "mimirmind/ServerBootstrap.hpp"

#include "core/config/Config.hpp"
#include "core/log/Log.hpp"
#include "model/Tokenizer.hpp"
#include "runtime/InferenceEngine.hpp"
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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <system_error>

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

} // namespace mimirmind::cli
