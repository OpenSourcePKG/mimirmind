#include "server/SystemStatusBuilder.hpp"

#include "server/RequestDispatcher.hpp"
#include "server/RequestTracker.hpp"

#include "runtime/FanController.hpp"
#include "runtime/GpuClockGovernor.hpp"
#include "runtime/InferenceEngine.hpp"
#include "runtime/PerfRegressionDetector.hpp"
#include "runtime/SpeculativeDecoder.hpp"
#include "runtime/ThermalGuard.hpp"

#include <limits>
#include <vector>

namespace mimirmind::server {

using nlohmann::json;

SystemStatusBuilder::SystemStatusBuilder(runtime::InferenceEngine& engine,
                                          RequestDispatcher&        dispatcher,
                                          RequestTracker&           requestTracker,
                                          std::string_view          modelId)
    : _engine{engine},
      _dispatcher{dispatcher},
      _requestTracker{requestTracker},
      _modelId{modelId} {
    // Engine is constructed model-loaded already (loadModel is called
    // before ApiServer wraps it in main.cpp). Capture the baseline here
    // so it represents "right after model load, before any requests".
    // If the monitor is unavailable the snapshot will be empty and
    // totals stay at 0.
    if (auto* mon = _engine.powerMonitor(); mon != nullptr && mon->available()) {
        std::lock_guard lk{_powerStateMutex};
        _powerBaseline      = mon->snapshot();
        _powerLastStatus    = _powerBaseline;
        _baselineWallStart  = std::chrono::steady_clock::now();
        _baselineCaptured   = true;
    }
}

json SystemStatusBuilder::buildInfo() const {
    const auto& modelCfg = _engine.config();
    const auto& tok      = _engine.tokenizer();
    const auto& devInfo  = _engine.ctx().info();
    const auto& usmLim   = _engine.allocator().limits();

    // Model architecture + dims
    json model = {
        {"id",                   _modelId},
        {"arch",                 modelCfg.architecture},
        {"block_count",          modelCfg.blockCount},
        {"context_length",       modelCfg.contextLength},
        {"embedding_length",     modelCfg.embeddingLength},
        {"feed_forward_length",  modelCfg.feedForwardLength},
        {"head_count",           modelCfg.headCount},
        {"head_count_kv",        modelCfg.headCountKv},
        {"key_length",           modelCfg.keyLength},
        {"value_length",         modelCfg.valueLength},
        {"rms_norm_eps",         modelCfg.rmsNormEps},
        {"rope_freq_base",       modelCfg.ropeFreqBase},
    };
    if (modelCfg.slidingWindow > 0) {
        model["sliding_window"]     = modelCfg.slidingWindow;
        model["rope_freq_base_swa"] = modelCfg.ropeFreqBaseSwa;
        model["key_length_swa"]     = modelCfg.keyLengthSwa;
        model["value_length_swa"]   = modelCfg.valueLengthSwa;
        std::size_t swa = 0;
        for (bool b : modelCfg.slidingWindowPattern) {
            if (b) ++swa;
        }
        model["swa_layer_count"]  = swa;
        model["full_layer_count"] =
            modelCfg.slidingWindowPattern.size() - swa;
    }
    if (modelCfg.expertCount > 0) {
        model["expert_count"]      = modelCfg.expertCount;
        model["expert_used_count"] = modelCfg.expertUsedCount;
    }

    // Tokenizer
    json tokenizer = {
        {"model",      std::string{tok.modelType()}},
        {"vocab_size", tok.vocabSize()},
        {"bos_id",     tok.bosId()},
        {"eos_id",     tok.eosId()},
        {"unk_id",     tok.unknownId()},
        {"pad_id",     tok.padId()},
    };

    // KV cache — hard limit the engine will admit. M10.2 Phase 1a —
    // element_bytes is meaningless on Q8_0 (block-based); reports
    // block_bytes + block_elements so the same JSON shape covers all
    // three dtypes.
    const auto kvD = _engine.kvDtype();
    const char* kvDName = (kvD == runtime::KvDtype::FP16 ? "fp16"
                         : kvD == runtime::KvDtype::Q8_0 ? "q8_0"
                                                         : "f32");
    json kvCache = {
        {"max_context_tokens", _engine.maxContextTokens()},
        {"layer_count",        modelCfg.blockCount},
        {"dtype",              kvDName},
        {"block_bytes",        runtime::kvBlockBytes(kvD)},
        {"block_elements",     runtime::kvBlockElements(kvD)},
    };

    // Level-Zero device descriptor
    json hardware = {
        {"device_name",             devInfo.name},
        {"device_uuid",             devInfo.uuid},
        {"vendor_id",               devInfo.vendorId},
        {"device_id",               devInfo.deviceId},
        {"num_compute_units",       devInfo.numComputeUnits},
        {"core_clock_rate_mhz",     devInfo.coreClockRate},
        {"total_local_mem_bytes",   devInfo.totalLocalMem},
        {"usm_per_alloc_max_bytes", usmLim.perAllocMaxBytes},
    };

    // GPU clock envelope — the static parts of /system/status.gpu_clock.
    json gpuClockEnvelope;
    if (auto* gov = _engine.gpuClockGovernor();
        gov != nullptr && gov->available()) {
        gpuClockEnvelope = {
            {"card_path",     std::string{gov->cardPath()}},
            {"rp0_mhz",       gov->rp0Mhz()},
            {"rpn_mhz",       gov->rpnMhz()},
            {"target_temp_c", gov->targetTempC()},
        };
        // M9.11.a — bench-repeatability pin.
        if (gov->pinned()) {
            gpuClockEnvelope["pin"] = {
                {"intent",  std::string{gov->pinIntent()}},
                {"raw_env", std::string{gov->pinRawEnv()}},
                {"cap_mhz", gov->pinnedMhz()},
            };
        }
    } else {
        gpuClockEnvelope = nullptr;
    }

    // Thermal profile — static limits only.
    json thermalProfile;
    if (auto* guard = _engine.thermalGuard(); guard != nullptr) {
        const auto& p = guard->profile();
        thermalProfile = {
            {"name",                    p.name},
            {"description",             p.description},
            {"package_temp_hard_c",     p.package_temp_hard_c.has_value()
                                          ? json(*p.package_temp_hard_c)
                                          : json(nullptr)},
            {"package_temp_soft_c",     p.package_temp_soft_c.has_value()
                                          ? json(*p.package_temp_soft_c)
                                          : json(nullptr)},
            {"package_throttle_max_ms", p.package_throttle_max_ms},
        };
    } else {
        thermalProfile = nullptr;
    }

    json perfRegressionConfig = {
        {"threshold_ratio",
         runtime::PerfRegressionDetector::kAlertThreshold},
        {"baseline_window_days",
         runtime::PerfRegressionDetector::kBaselineDays},
        {"warmup_tokens",
         runtime::PerfRegressionDetector::kWarmupTokens},
        {"rolling_window",
         runtime::PerfRegressionDetector::kRollingWindow},
        {"min_run_samples",
         runtime::PerfRegressionDetector::kMinRunSamples},
        {"min_baseline_n",
         runtime::PerfRegressionDetector::kMinBaselineN},
    };

    // Build / process identity.
    json build = json::object();
    if (auto* det = _engine.perfRegressionDetector()) {
        build["internal_version"] = det->internalVersion();
    }

    // Fan envelope — static chip identity.
    json fanEnvelope;
    if (auto* fc = _engine.fanController();
        fc != nullptr && fc->available()) {
        fanEnvelope = {
            {"chip_name",         std::string{fc->chipName()}},
            {"chip_path",         std::string{fc->chipPath()}},
            {"pwm_path",          std::string{fc->pwmPath()}},
            {"pwm_enable_path",   std::string{fc->pwmEnablePath()}},
            {"fan_input_path",    std::string{fc->fanInputPath()}},
            {"original_pwm",      fc->originalPwm()},
            {"original_enable",   fc->originalEnableMode()},
            {"boost_pwm",         fc->boostPwm()},
            {"min_safe_pwm",      fc->minSafePwm()},
        };
    } else {
        fanEnvelope = nullptr;
    }

    // M9.11.1 + M9.11.4 — Speculative-decoding readiness.
    json speculativeDecoding;
    auto* draft   = _dispatcher.draftEngine();
    auto* specDec = _dispatcher.speculativeDecoder();
    if (draft != nullptr && specDec != nullptr) {
        const auto& draftCfg = draft->config();
        speculativeDecoding = {
            {"status",                 "ready"},
            {"mode",                   "greedy"},
            {"draft_n",                specDec->config().draftN},
            {"draft_model_arch",       draftCfg.architecture},
            {"draft_block_count",      draftCfg.blockCount},
            {"draft_embedding_length", draftCfg.embeddingLength},
        };
    } else {
        speculativeDecoding = {
            {"status", "disabled"},
        };
    }

    return json{
        {"model",                  model},
        {"tokenizer",              tokenizer},
        {"kv_cache",               kvCache},
        {"hardware",               hardware},
        {"gpu_clock_envelope",     gpuClockEnvelope},
        {"fan_envelope",           fanEnvelope},
        {"thermal_profile",        thermalProfile},
        {"perf_regression_config", perfRegressionConfig},
        {"kernels",                buildKernelsBlock()},
        {"speculative_decoding",   speculativeDecoding},
        {"build",                  build},
    };
}

json SystemStatusBuilder::buildStatus() {
    auto* guard = _engine.thermalGuard();
    json body{
        {"profile_active", guard != nullptr},
    };
    if (guard == nullptr) {
        body["warning"] =
            "no thermal profile configured — engine is unprotected. "
            "Fill the governor.thermal section in config.json.";
        return body;
    }

    const auto& p        = guard->profile();
    const auto  decision = guard->decide();
    const auto  reading  = guard->lastReading();

    json profileJson{
        {"name",        p.name},
        {"description", p.description},
    };
    if (p.hasPackageLimits()) {
        profileJson["package_temp_soft_c"]    = *p.package_temp_soft_c;
        profileJson["package_temp_hard_c"]    = *p.package_temp_hard_c;
        profileJson["package_throttle_max_ms"] = p.package_throttle_max_ms;
    }

    json readingsJson = json::object();
    if (reading.package_temp_c.has_value()) {
        readingsJson["package_temp_c"] = *reading.package_temp_c;
    }
    if (reading.ram_total_mib.has_value()) {
        readingsJson["ram_total_mib"] = *reading.ram_total_mib;
    }
    if (reading.ram_available_mib.has_value()) {
        readingsJson["ram_available_mib"] = *reading.ram_available_mib;
    }

    const char* stateStr =
        decision.state == runtime::ThermalDecision::State::Critical   ? "critical"
        : decision.state == runtime::ThermalDecision::State::Throttling ? "throttling"
                                                                        : "ok";

    body["profile"]   = std::move(profileJson);
    body["readings"]  = std::move(readingsJson);
    body["throttle"]  = json{
        {"state",                stateStr},
        {"current_pause_ms",     static_cast<int>(decision.pause.count())},
        {"next_request_allowed", decision.admit_new_request},
        {"reason",               decision.reason.empty()
                                   ? json{}
                                   : json{decision.reason}},
    };
    body["power"]           = buildPowerBlock();
    body["gpu_clock"]       = buildGpuClockBlock();
    body["fan"]             = buildFanBlock();
    body["kernels"]         = buildKernelsBlock();
    body["perf_regression"] = buildPerfRegressionBlock();
    body["current_request"] = _requestTracker.buildStatusBlock();
    return body;
}

json SystemStatusBuilder::buildPerfRegressionBlock() const {
    auto* det = _engine.perfRegressionDetector();
    if (det == nullptr) {
        return json{
            {"available", false},
            {"reason",    "no perf-regression detector installed"},
        };
    }
    json body{
        {"available",              true},
        {"internal_version",       det->internalVersion()},
        {"threshold_ratio",        runtime::PerfRegressionDetector::kAlertThreshold},
        {"baseline_window_days",   runtime::PerfRegressionDetector::kBaselineDays},
        {"warmup_tokens",          runtime::PerfRegressionDetector::kWarmupTokens},
        {"baseline_sample_count",  det->baselineSampleCount()},
    };
    const double curP50 = det->currentP50Ms();
    const double basP50 = det->baselineP50Ms();
    if (curP50 > 0.0) {
        body["current_p50_ms"] = curP50;
    } else {
        body["current_p50_ms"] = nullptr;
    }
    if (basP50 > 0.0) {
        body["baseline_p50_ms"] = basP50;
    } else {
        body["baseline_p50_ms"] = nullptr;
    }
    if (auto alert = det->lastAlert()) {
        body["last_alert"] = json{
            {"current_p50_ms",   alert->current_p50_ms},
            {"baseline_p50_ms",  alert->baseline_p50_ms},
            {"delta_ratio",      alert->delta_ratio},
            {"internal_version", alert->internal_version},
            {"detected_unix",    alert->detected_unix},
        };
    } else {
        body["last_alert"] = json{};
    }
    return body;
}

json SystemStatusBuilder::buildGpuClockBlock() const {
    auto* gov = _engine.gpuClockGovernor();
    if (gov == nullptr) {
        return json{
            {"available", false},
            {"reason",    "no GPU clock governor installed (profile "
                          "has no gpu_target_temp_c)"},
        };
    }
    if (!gov->available()) {
        return json{
            {"available", false},
            {"reason",    std::string{gov->unavailableReason()}},
        };
    }
    json body{
        {"available",       true},
        {"card_path",       std::string{gov->cardPath()}},
        {"rp0_mhz",         gov->rp0Mhz()},
        {"rpn_mhz",         gov->rpnMhz()},
        {"current_cap_mhz", gov->currentCapMhz()},
        {"target_temp_c",   gov->targetTempC()},
    };
    if (gov->pinned()) {
        body["pin"] = {
            {"intent",  std::string{gov->pinIntent()}},
            {"raw_env", std::string{gov->pinRawEnv()}},
            {"cap_mhz", gov->pinnedMhz()},
        };
    }
    return body;
}

json SystemStatusBuilder::buildFanBlock() const {
    auto* fc = _engine.fanController();
    if (fc == nullptr) {
        return json{
            {"available", false},
            {"reason",    "no FanController installed "
                          "(governor.fan.boost=false or probe failed)"},
        };
    }
    if (!fc->available()) {
        return json{
            {"available", false},
            {"reason",    std::string{fc->unavailableReason()}},
        };
    }
    return json{
        {"available",   true},
        {"chip_name",   std::string{fc->chipName()}},
        {"current_pwm", fc->currentPwm()},
        {"current_rpm", fc->currentFanRpm()},
        {"mode",        fc->currentEnableMode()},  // 1=manual, 2..=auto
        {"boost_active", fc->boostActive()},
    };
}

json SystemStatusBuilder::buildKernelsBlock() const {
    json body = json::object();

    json matmulByType = json::object();
    for (const auto& r : _engine.gpuMatmul().autotuneReport()) {
        json vecMsAtM  = json::object();
        json gemmMsAtM = json::object();
        for (std::size_t i = 0; i < r.mBuckets.size(); ++i) {
            const std::string key = std::to_string(r.mBuckets[i]);
            vecMsAtM[key]  = r.vecMsAtM[i];
            gemmMsAtM[key] = r.gemmMsAtM[i];
        }
        json gemmMinMJson = nullptr;
        if (r.gemmMinM != 0 &&
            r.gemmMinM != std::numeric_limits<std::size_t>::max())
        {
            gemmMinMJson = r.gemmMinM;
        }
        json gemmV2MsAtM = json::object();
        for (std::size_t i = 0; i < r.mBuckets.size(); ++i) {
            gemmV2MsAtM[std::to_string(r.mBuckets[i])] =
                r.gemmV2MsAtM[i];
        }
        matmulByType[r.name] = json{
            {"gemm_available",    r.gemmAvailable},
            {"gemm_picked",       r.gemmPicked},
            {"gemm_min_m",        gemmMinMJson},
            {"vec_ms",            r.vecMs},
            {"gemm_ms",           r.gemmMs},
            {"vec_ms_at_m",       std::move(vecMsAtM)},
            {"gemm_ms_at_m",      std::move(gemmMsAtM)},
            {"dp4a_available",    r.dp4aAvailable},
            {"dp4a_picked",       r.dp4aPicked},
            {"dp4a_ms",           r.dp4aMs},
            {"gemm_v2_available", r.gemmV2Available},
            {"gemm_v2_picked",    r.gemmV2Picked},
            {"gemm_v2_ms_at_m",   std::move(gemmV2MsAtM)},
            {"source",            r.source},
        };
    }
    body["matmul"] = std::move(matmulByType);

    json fusedJson;
    if (const auto* fq = _engine.fusedQkv()) {
        fusedJson = json{
            {"disabled", fq->disabled()},
            {"blocks_fused",    fq->fusedCount()},
            {"blocks_skipped",  fq->skippedCount()},
            {"usm_mib",         static_cast<unsigned long long>(
                                    (fq->totalUsmBytes()
                                     + (1ULL << 20) - 1) >> 20)},
        };
    } else {
        fusedJson = json{{"available", false}};
    }
    body["fused_qkv"] = std::move(fusedJson);

    body["selftest"] = std::string{_engine.gpuOps().selfTestStatus()};

    // Prefill-flash rollback surface — reports the two independent
    // toggles (features.flashPrefill, features.flashPrefillGqaQ8) as the
    // engine sees them post-config. Lets an operator verify a config
    // change actually took effect without diffing the startup log.
    body["prefill_flash"] = json{
        {"enabled",        _engine.gpuOps().prefillFlashEnabled()},
        {"gqa_q8_enabled", _engine.gpuOps().prefillFlashGqaQ8Enabled()},
    };

    return body;
}

json SystemStatusBuilder::buildPowerBlock() {
    auto* mon = _engine.powerMonitor();
    if (mon == nullptr) {
        return json{
            {"available", false},
            {"reason",    "no power monitor installed"},
        };
    }
    if (!mon->available()) {
        return json{
            {"available", false},
            {"reason",    std::string{mon->unavailableReason()}},
        };
    }

    const auto now = mon->snapshot();
    std::vector<double>                  totalJoules;
    std::vector<double>                  wattsNow;
    std::chrono::steady_clock::time_point baselineAt;
    bool                                  haveBaseline = false;
    {
        std::lock_guard lk{_powerStateMutex};
        if (_baselineCaptured) {
            totalJoules      = mon->energyBetween(_powerBaseline, now);
            wattsNow         = mon->averageWattsBetween(_powerLastStatus, now);
            baselineAt       = _baselineWallStart;
            _powerLastStatus = now;
            haveBaseline     = true;
        }
    }

    json domains = json::array();
    const auto names = mon->domainNames();
    for (std::size_t i = 0; i < names.size(); ++i) {
        json d{{"name", names[i]}};
        if (haveBaseline && i < wattsNow.size()) {
            d["watts_now"] = wattsNow[i];
        }
        if (haveBaseline && i < totalJoules.size()) {
            d["total_joules"] = totalJoules[i];
        }
        domains.push_back(std::move(d));
    }

    json out{
        {"available", true},
        {"domains",   std::move(domains)},
    };
    if (haveBaseline) {
        const auto uptime_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - baselineAt).count();
        out["uptime_s"] = uptime_s;
    }
    return out;
}

} // namespace mimirmind::server