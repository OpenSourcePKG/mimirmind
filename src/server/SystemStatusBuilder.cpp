// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "server/SystemStatusBuilder.hpp"

#include "server/ModelMemoryJson.hpp"
#include "server/RequestDispatcher.hpp"
#include "server/RequestTracker.hpp"

#include "core/backend/BackendPool.hpp"
#include "core/backend/BackendRegistry.hpp"

#include "runtime/thermal/FanController.hpp"
#include "runtime/thermal/GpuClockGovernor.hpp"
#include "runtime/InferenceEngine.hpp"
#include "runtime/perf/PerfRegressionDetector.hpp"
#include "runtime/spec/ModelDrafter.hpp"
#include "runtime/spec/NGramDrafter.hpp"
#include "runtime/spec/SpeculativeDecoder.hpp"
#include "runtime/thermal/ThermalGuard.hpp"
#ifdef MIMIRMIND_HAVE_CUDA
#include "runtime/thermal/NvmlTelemetry.hpp"
#endif
#ifdef MIMIRMIND_HAVE_L0
#include "core/gpu/l0/UsmAllocator.hpp"   // 8.16: L0 free-list fragmentation stats
#endif

#include <limits>
#include <vector>
#ifdef __linux__
#include <array>
#include <fstream>
#include <sstream>
#include <string>
#include <unistd.h>
#endif

namespace {
#ifdef __linux__
// 8.16 Stage B Inc-2 — passive host-VA / physical-RAM view from /proc. On GB10
// the CPU (Grace) and GPU (Blackwell) share one coherent LPDDR pool, so the
// kernel's own page accounting is a real, zero-cost window into how the shared
// RAM is distributed + fragmented — no invasive largest-free-block probe.
struct HostMemInfo {
    bool          available{false};
    std::uint64_t rssBytes{0};        // /proc/self/smaps_rollup Rss
    std::uint64_t pssBytes{0};        // proportional set size
    std::uint64_t anonBytes{0};       // Anonymous
    std::uint64_t memTotalBytes{0};   // /proc/meminfo
    std::uint64_t memFreeBytes{0};
    std::uint64_t memAvailBytes{0};
    // /proc/buddyinfo: free-block count per order (summed across zones), and
    // the derived largest contiguous free block = highest order with count>0.
    static constexpr std::size_t kMaxOrders = 16;
    std::array<std::uint64_t, kMaxOrders> freePagesByOrder{};
    std::size_t   nOrders{0};
    std::uint64_t pageSize{4096};
    std::uint64_t largestFreeContiguousBytes{0};
    std::uint64_t buddyFreeBytes{0};  // Σ count × (pageSize<<order)
};

// Read "Key:   N kB" from a /proc file → bytes. 0 when absent.
std::uint64_t readProcKb(const std::string& path, const char* key) {
    std::ifstream f{path};
    std::string   line;
    const std::string want{key};
    while (std::getline(f, line)) {
        if (line.rfind(want, 0) == 0) {
            std::istringstream ss{line.substr(want.size())};
            std::uint64_t      kb = 0;
            ss >> kb;
            return kb * 1024ULL;
        }
    }
    return 0;
}

HostMemInfo readHostMemInfo() {
    HostMemInfo h;
    h.pageSize = static_cast<std::uint64_t>(::sysconf(_SC_PAGESIZE));
    if (h.pageSize == 0) h.pageSize = 4096;

    h.rssBytes  = readProcKb("/proc/self/smaps_rollup", "Rss:");
    h.pssBytes  = readProcKb("/proc/self/smaps_rollup", "Pss:");
    h.anonBytes = readProcKb("/proc/self/smaps_rollup", "Anonymous:");
    h.memTotalBytes = readProcKb("/proc/meminfo", "MemTotal:");
    h.memFreeBytes  = readProcKb("/proc/meminfo", "MemFree:");
    h.memAvailBytes = readProcKb("/proc/meminfo", "MemAvailable:");

    // buddyinfo: "Node N, zone NAME  c0 c1 c2 ...cK" — sum counts per order.
    std::ifstream bf{"/proc/buddyinfo"};
    std::string   line;
    while (std::getline(bf, line)) {
        const auto zonePos = line.find("zone");
        if (zonePos == std::string::npos) continue;
        std::istringstream ss{line.substr(zonePos + 4)};
        std::string        zoneName;
        ss >> zoneName;                 // skip zone name token
        std::uint64_t      count = 0;
        std::size_t        order = 0;
        while (ss >> count && order < HostMemInfo::kMaxOrders) {
            h.freePagesByOrder[order] += count;
            if (order + 1 > h.nOrders) h.nOrders = order + 1;
            ++order;
        }
    }
    for (std::size_t o = 0; o < h.nOrders; ++o) {
        const std::uint64_t blockBytes = h.pageSize << o;
        h.buddyFreeBytes += h.freePagesByOrder[o] * blockBytes;
        if (h.freePagesByOrder[o] > 0) h.largestFreeContiguousBytes = blockBytes;
    }
    h.available = (h.memTotalBytes > 0);
    return h;
}
#endif // __linux__

#ifdef MIMIRMIND_HAVE_CUDA
// Process-lifetime NVML telemetry reader (observability only; never enforces).
// Lazily constructed on first status request so a CPU/L0 build without a driver
// pays nothing. Used as a fallback on the CUDA (GB10) backend where the sysfs
// PowerMonitor / GpuClockGovernor do not apply.
const ::mimirmind::runtime::thermal::NvmlTelemetry& nvmlTelemetry() {
    return ::mimirmind::runtime::thermal::nvmlTelemetry();
}
#endif
} // namespace

namespace mimirmind::server {

using nlohmann::json;

SystemStatusBuilder::SystemStatusBuilder(runtime::InferenceEngine* engine,
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
    if (_engine == nullptr) {
        return;   // pool mode: no resident engine to baseline
    }
    if (auto* mon = _engine->powerMonitor(); mon != nullptr && mon->available()) {
        std::lock_guard lk{_powerStateMutex};
        _powerBaseline      = mon->snapshot();
        _powerLastStatus    = _powerBaseline;
        _baselineWallStart  = std::chrono::steady_clock::now();
        _baselineCaptured   = true;
    }
}

json SystemStatusBuilder::buildInfo() const {
    if (_engine == nullptr) {
        // M-Munin.3 pool mode: no always-resident engine to describe. The
        // servable model ids come from /v1/models (the ModelProvider); this
        // static info block reports only the mode.
        return json{
            {"mode", "pooled_model_switch"},
            {"note", "per-request model-switch pool: no always-resident "
                     "engine; query /v1/models for servable ids"},
        };
    }
    const auto& modelCfg = _engine->config();
    const auto& tok      = _engine->tokenizer();

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
    const auto kvD = _engine->kvDtype();
    const char* kvDName = (kvD == runtime::KvDtype::FP16 ? "fp16"
                         : kvD == runtime::KvDtype::Q8_0 ? "q8_0"
                         : kvD == runtime::KvDtype::FP8_E4M3 ? "fp8_e4m3"
                                                         : "f32");
    json kvCache = {
        {"max_context_tokens", _engine->maxContextTokens()},
        {"layer_count",        modelCfg.blockCount},
        {"dtype",              kvDName},
        {"block_bytes",        runtime::kvBlockBytes(kvD)},
        {"block_elements",     runtime::kvBlockElements(kvD)},
    };

    // Hardware descriptor — populated from L0 device info + USM limits
    // when the runtime picked L0. HIP-only / CPU-only builds report a
    // minimal placeholder here; a HIP-aware version can plumb through
    // `hipGetDeviceProperties` once the backend-neutral surface lands.
    json hardware;
#ifdef MIMIRMIND_HAVE_L0
    if (_engine->computeContextKind() == core::backend::BackendKind::LevelZero) {
        const auto& devInfo  = _engine->ctx().info();
        const auto& usmLim   = _engine->allocator().limits();
        hardware = {
        {"device_name",             devInfo.name},
        {"device_uuid",             devInfo.uuid},
        {"vendor_id",               devInfo.vendorId},
        {"device_id",               devInfo.deviceId},
        {"num_compute_units",       devInfo.numComputeUnits},
        {"core_clock_rate_mhz",     devInfo.coreClockRate},
        {"total_local_mem_bytes",   devInfo.totalLocalMem},
        {"usm_per_alloc_max_bytes", usmLim.perAllocMaxBytes},
        };
    }
#endif
    if (hardware.is_null()) {
        // Fallback for HIP-only, CPU-only, or L0-off runs — expose just
        // the backend kind so operators can tell the endpoint is alive
        // and see which backend is bound to this engine.
        hardware = {
            {"backend",   core::backend::BackendRegistry::name(
                              _engine->computeContextKind())},
            {"note",      "detailed device descriptors are only wired "
                          "for the L0 backend today"},
        };
    }

    // GPU clock envelope — the static parts of /system/status.gpu_clock.
    json gpuClockEnvelope;
    if (auto* gov = _engine->gpuClockGovernor();
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
    if (auto* guard = _engine->thermalGuard(); guard != nullptr) {
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
    if (auto* det = _engine->perfRegressionDetector()) {
        build["internal_version"] = det->internalVersion();
    }

    // Fan envelope — static chip identity.
    json fanEnvelope;
    if (auto* fc = _engine->fanController();
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

    // M9.11.1 + M9.11.4 — Speculative-decoding readiness. The drafter
    // kind selects which model-specific fields we report — `model`
    // exposes the backing engine's arch/block/embedding, `ngram` its
    // minK/maxK PLD window.
    json speculativeDecoding;
    auto* drafter = _dispatcher.drafter();
    auto* specDec = _dispatcher.speculativeDecoder();
    if (drafter != nullptr && specDec != nullptr) {
        speculativeDecoding = {
            {"status",  "ready"},
            {"mode",    "greedy"},
            {"drafter", std::string{drafter->kind()}},
            {"draft_n", specDec->config().draftN},
        };
        if (auto* md = dynamic_cast<runtime::ModelDrafter*>(drafter)) {
            const auto& draftCfg = md->engine().config();
            speculativeDecoding["draft_model_arch"]       = draftCfg.architecture;
            speculativeDecoding["draft_block_count"]      = draftCfg.blockCount;
            speculativeDecoding["draft_embedding_length"] = draftCfg.embeddingLength;
        } else if (auto* nd = dynamic_cast<runtime::NGramDrafter*>(drafter)) {
            speculativeDecoding["ngram_min_k"] = nd->config().minK;
            speculativeDecoding["ngram_max_k"] = nd->config().maxK;
        }
    } else {
        speculativeDecoding = {
            {"status", "disabled"},
        };
    }

    // Backend pool — every compiled-in + available device the process
    // could bind an engine to. Also reports which entry this engine
    // actually bound to, so operators can tell "process saw the dGPU
    // AND the iGPU, this engine picked the dGPU" at a glance.
    // Discovery is cheap (a few microseconds); running it per
    // /system/info call keeps the builder stateless.
    json poolJson = json::array();
    {
        core::backend::BackendPool pool;
        pool.discoverAll();
        for (const auto& e : pool.entries()) {
            poolJson.push_back({
                {"kind",      core::backend::BackendRegistry::name(e.kind)},
                {"device_ix", e.deviceIx},
                {"token",     e.token},
                {"name",      e.name},
                {"detail",    e.detail},
            });
        }
    }
    const auto engineKind = _engine->computeContextKind();
    json engineBackend = {
        {"kind",  core::backend::BackendRegistry::name(engineKind)},
        {"token", core::backend::tokenFor(engineKind, /*deviceIx=*/0)},
    };

    // M-Startup.CapacityProbe / Bragi — startup probe snapshot + the
    // resulting serving-class gate decision. Sourced from
    // `InferenceEngine::batchCapacity()` (populated once in
    // finalizeLoad(), immutable thereafter for this engine instance).
    // Operators use this to tell "why is this instance single-session"
    // at a glance — reasoning field carries the raw probe inputs.
    const auto& est = _engine->batchCapacity();
    const auto& cfgServing = _engine->servingConfig();
    json serving = {
        {"enable_batching",         cfgServing.enableBatching == core::config::TriState::Auto    ? "auto"
                                  : cfgServing.enableBatching == core::config::TriState::Force   ? "force"
                                                                                                 : "disable"},
        {"min_batch_for_enable",    cfgServing.minBatchForEnable},
        {"sustainable_batch",       est.sustainableBatch},
        {"serving_class_enabled",   _engine->servingClassEnabled()},
        {"probe_recommended",       est.servingClassRecommended},
        {"bandwidth_gbps",          est.bandwidthGBps},
        {"weight_bytes",            est.weightBytes},
        {"kv_bytes_per_token",      est.kvBytesPerToken},
        {"max_context",             est.maxContext},
        {"reasoning",               est.reasoning},
        // M-Cuda.Batch Phase C knobs — surfaced so operators can
        // read the effective serving-loop configuration at a glance.
        // Populated by ServingSettings defaults + config.json overrides
        // (see `2715fdd`); consumed at Phase-D construction time by
        // RequestScheduler / ChunkedPrefillScheduler / PreemptionPolicy /
        // PagedKvBlockAllocator.
        {"token_budget",                    cfgServing.tokenBudget},
        {"max_active_requests",             cfgServing.maxActiveRequests},
        {"preempt_free_block_threshold",    cfgServing.preemptFreeBlockThreshold},
        {"block_size",                      cfgServing.blockSize},
    };

    return json{
        {"model",                  model},
        {"tokenizer",              tokenizer},
        {"kv_cache",               kvCache},
        {"hardware",               hardware},
        {"backend_pool",           poolJson},
        {"engine_backend",         engineBackend},
        {"serving",                serving},
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
    if (_engine == nullptr) {
        // Pool mode: report only mode + in-flight requests (engine-independent).
        return json{
            {"mode",            "pooled_model_switch"},
            {"current_request", _requestTracker.buildStatusBlock()},
        };
    }
    auto* guard = _engine->thermalGuard();
    json body{
        {"profile_active", guard != nullptr},
    };

    // Enforcement (guard-gated): thermal profile + throttle decision + the
    // guard's own readings. No guard = no throttling (engine runs unthrottled,
    // full load) — but telemetry below is still reported (observability is
    // independent of enforcement).
    if (guard != nullptr) {
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
    } else {
        body["warning"] =
            "no thermal profile configured — engine runs unthrottled (full "
            "load). Fill the governor.thermal section in config.json to enable "
            "thermal enforcement.";
#ifdef MIMIRMIND_HAVE_CUDA
        // Temperature + memory telemetry via NVML (the guard's usual source is
        // absent). Same field names as the guard readings so clients need no
        // special-casing.
        const auto s = nvmlTelemetry().sample();
        json readingsJson = json::object();
        if (s.temp_c.has_value()) {
            readingsJson["package_temp_c"] = *s.temp_c;
        }
        if (s.mem_total_mib.has_value()) {
            readingsJson["ram_total_mib"] = *s.mem_total_mib;
        }
        if (s.mem_used_mib.has_value() && s.mem_total_mib.has_value()) {
            readingsJson["ram_available_mib"] =
                *s.mem_total_mib - *s.mem_used_mib;
        }
        if (!readingsJson.empty()) {
            body["readings"] = std::move(readingsJson);
        }
#endif
    }

    // Observability (always reported, independent of the thermal guard).
    body["power"]           = buildPowerBlock();
    body["gpu_clock"]       = buildGpuClockBlock();
    body["fan"]             = buildFanBlock();
    body["kernels"]         = buildKernelsBlock();
    body["perf_regression"] = buildPerfRegressionBlock();
    body["current_request"] = _requestTracker.buildStatusBlock();
    return body;
}

json SystemStatusBuilder::buildMemory() const {
    // Per-resident-model attribution (additive `models` block). Works in both
    // modes: it enumerates the dispatcher's resident engines (eager default +
    // extras) and, in pool mode, the provider's materialized slots. Kept
    // separate from the device/category envelope below so a multi-model host
    // can attribute weight/KV bytes per model instead of lumping the extras
    // into `external`.
    const json models = buildModelsMemoryJson(_dispatcher.residentModelsMemory(),
                                               _dispatcher.isPoolMode(),
                                               _dispatcher.poolCapacity());

    // Pool (per-request model-switch) mode with no eager anchor engine: there
    // is no global device/category envelope to inspect, but the `models` block
    // (materialized pool slots) still applies.
    if (_engine == nullptr) {
        return json{
            {"mode",      "pooled_model_switch"},
            {"available", false},
            {"models",    models},
        };
    }

    const auto mt = _engine->memoryTelemetry();
    const std::size_t deviceUsed =
        (mt.deviceMemAvailable && mt.deviceTotalBytes >= mt.deviceFreeBytes)
            ? mt.deviceTotalBytes - mt.deviceFreeBytes : 0;

    // Device envelope (ground truth from the compute context's mem-info).
    json device = mt.deviceMemAvailable
        ? json{{"available",   true},
               {"total_bytes", mt.deviceTotalBytes},
               {"free_bytes",  mt.deviceFreeBytes},
               {"used_bytes",  deviceUsed}}
        : json{{"available", false},
               {"reason", "backend does not report device mem-info"}};

    // Resident owner categories (precise sums).
    json kvPaged{{"serving_active", mt.servingActive}};
    if (mt.servingActive) {
        kvPaged["resident_bytes"] = mt.kvResidentBytes;
        kvPaged["num_blocks"]     = mt.kvNumBlocks;
        kvPaged["block_size"]     = mt.kvBlockSize;
        kvPaged["num_layers"]     = mt.kvNumLayers;
    }
    json categories{
        {"weights_bytes", mt.weightBytes},
        {"kv_paged",      kvPaged},
    };

    // 8.16 Stage B — central-allocator per-category breakdown (live + peak).
    // Present only when the backend tracks categories (CUDA today; L0/HIP
    // return zeros -> owner-sum path). This decomposes what used to be one
    // `external` lump into weights/kv_cache/session/scratch/unknown.
    std::uint64_t allocTracked = 0;
    json allocatorCategories;
    if (mt.allocCatAvailable) {
        allocatorCategories = json::object();
        for (std::size_t i = 0; i < core::gpu::kAllocCategoryCount; ++i) {
            allocTracked += mt.allocCatLive[i];
            allocatorCategories[std::string(core::gpu::allocCategoryName(i))] = json{
                {"live_bytes", mt.allocCatLive[i]},
                {"peak_bytes", mt.allocCatPeak[i]},
            };
        }
    } else {
        allocatorCategories = json{
            {"available", false},
            {"reason", "central-allocator category tagging not active on this "
                       "backend"}};
    }

    // external = deviceUsed - Σ(tracked). With Stage-B category tracking the
    // subtrahend is the allocator-tracked total (precise); otherwise it falls
    // back to the owner-sum (weights + kv). Residual = allocations that bypass
    // the central allocator: GpuMatmul direct cudaMalloc, cuBLAS/cuDNN internal
    // workspaces, and (on a multi-model host) other engines' allocators.
    json external = mt.deviceMemAvailable
        ? [&] {
              const std::size_t accounted = mt.allocCatAvailable
                  ? static_cast<std::size_t>(allocTracked)
                  : mt.weightBytes + mt.kvResidentBytes;
              return json{{"available", true},
                          {"bytes", deviceUsed >= accounted
                                        ? deviceUsed - accounted : 0},
                          {"basis", mt.allocCatAvailable ? "allocator-tracked"
                                                         : "owner-sum"},
                          {"note", "unaccounted device use: GpuMatmul direct "
                                   "cudaMalloc + cuBLAS/cuDNN workspaces + "
                                   "other engines (multi-model host)"}};
          }()
        : json{{"available", false},
               {"reason", "device mem-info unavailable"}};

    // Allocator fragmentation — L0 free-list only; the CUDA allocator is a
    // pass-through (no free-list). Stage B adds central-allocator category
    // tagging for CUDA/HIP.
    json fragmentation{
        {"available", false},
        {"reason", "free-list fragmentation stats are L0-only (Stage B adds "
                   "central-allocator category tagging on CUDA/HIP)"}};
#ifdef MIMIRMIND_HAVE_L0
    if (_engine->computeContextKind() == core::backend::BackendKind::LevelZero) {
        const auto s = _engine->allocator().stats();
        fragmentation = json{
            {"available",        true},
            {"live_bytes",       s.liveBytes},
            {"peak_bytes",       s.peakBytes},
            {"bytes_requested",  s.bytesRequested},
            {"bytes_served",     s.bytesServed},
            {"free_list_hits",   s.freeListHits},
            {"free_list_misses", s.freeListMisses},
            {"oversized_allocs", s.oversizedAllocs},
        };
    }
#endif

    // 8.16 Stage B Inc-2 — passive host/physical view from /proc (GB10 unified
    // pool). process = this serve's resident footprint; system = whole-box RAM;
    // fragmentation = kernel buddy-allocator free-block histogram + the largest
    // contiguous free block (the real physical-fragmentation signal that CUDA's
    // own API cannot give). Linux-only; degrades to available:false elsewhere.
    json host;
#ifdef __linux__
    {
        const auto h = readHostMemInfo();
        if (h.available) {
            json byOrder = json::array();
            for (std::size_t o = 0; o < h.nOrders; ++o) {
                byOrder.push_back(json{
                    {"order",       o},
                    {"block_bytes", h.pageSize << o},
                    {"free_blocks", h.freePagesByOrder[o]},
                });
            }
            host = json{
                {"available", true},
                {"process", {{"rss_bytes",       h.rssBytes},
                             {"pss_bytes",       h.pssBytes},
                             {"anonymous_bytes", h.anonBytes}}},
                {"system",  {{"mem_total_bytes",     h.memTotalBytes},
                             {"mem_free_bytes",      h.memFreeBytes},
                             {"mem_available_bytes", h.memAvailBytes}}},
                {"fragmentation", {
                    {"largest_free_contiguous_bytes", h.largestFreeContiguousBytes},
                    {"buddy_free_bytes",              h.buddyFreeBytes},
                    {"free_blocks_by_order",          byOrder},
                    {"note", "kernel buddy-allocator free-page fragmentation of "
                             "the shared LPDDR pool (CPU+GPU coherent on GB10); "
                             "largest_free_contiguous_bytes = biggest single "
                             "physically-contiguous free block"}}},
            };
        } else {
            host = json{{"available", false}, {"reason", "/proc unreadable"}};
        }
    }
#else
    host = json{{"available", false}, {"reason", "host /proc view is Linux-only"}};
#endif

    return json{
        {"device",               device},
        {"categories",           categories},
        {"allocator_categories", allocatorCategories},
        {"external",             external},
        {"fragmentation",        fragmentation},
        {"host",                 host},
        {"models",               models},
    };
}

json SystemStatusBuilder::buildPerfRegressionBlock() const {
    if (_engine == nullptr) {
        return json{{"available", false},
                    {"reason", "pooled model-switch mode — no resident engine"}};
    }
    auto* det = _engine->perfRegressionDetector();
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
    if (_engine == nullptr) {
        return json{{"available", false},
                    {"reason", "pooled model-switch mode — no resident engine"}};
    }
    auto* gov = _engine->gpuClockGovernor();
    if (gov == nullptr || !gov->available()) {
#ifdef MIMIRMIND_HAVE_CUDA
        // No sysfs governor (CUDA/GB10) — report the live SM clock via NVML.
        const auto s = nvmlTelemetry().sample();
        if (s.sm_clock_mhz.has_value()) {
            json b{
                {"available",       true},
                {"source",          "nvml"},
                {"device",          nvmlTelemetry().deviceName()},
                {"current_cap_mhz", *s.sm_clock_mhz},
                {"current_mhz",     *s.sm_clock_mhz},
            };
            if (s.mem_clock_mhz.has_value()) {
                b["mem_clock_mhz"] = *s.mem_clock_mhz;
            }
            return b;
        }
#endif
        return json{
            {"available", false},
            {"reason",    gov == nullptr
                            ? std::string{"no GPU clock governor installed"}
                            : std::string{gov->unavailableReason()}},
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
    if (_engine == nullptr) {
        return json{{"available", false},
                    {"reason", "pooled model-switch mode — no resident engine"}};
    }
    auto* fc = _engine->fanController();
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
    if (_engine == nullptr) {
        return json{{"available", false},
                    {"reason", "pooled model-switch mode — no resident engine"}};
    }
    json body = json::object();

    json matmulByType = json::object();
    for (const auto& r : _engine->gpuMatmul().autotuneReport()) {
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
    if (const auto* fq = _engine->fusedQkv()) {
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

    body["selftest"] = std::string{_engine->gpuOps().selfTestStatus()};

    // Prefill-flash rollback surface — reports the two independent
    // toggles (features.flashPrefill, features.flashPrefillGqaQ8) as the
    // engine sees them post-config. Lets an operator verify a config
    // change actually took effect without diffing the startup log.
    body["prefill_flash"] = json{
        {"enabled",           _engine->gpuOps().prefillFlashEnabled()},
        {"gqa_q8_enabled",    _engine->gpuOps().prefillFlashGqaQ8Enabled()},
        {"gqa_q8_bq_enabled", _engine->gpuOps().prefillFlashGqaQ8BqEnabled()},
        {"k_tile_q8",         _engine->gpuOps().prefillFlashKTileQ8()},
        {"k_tile_q8_source",  std::string{
            _engine->gpuOps().prefillFlashKTileQ8Source()}},
    };

    // M8.K.Q8_0-Reorder — features.q8_0Reorder as-configured plus the
    // number of tensors any active backend actually routed through
    // the reorder path at load time. `active` flips true once at
    // least one tensor was reordered — Phase 5 first hit is the E4B
    // per_layer_model_proj weight. Prefill (M>1) still falls back to
    // native GEMM; the reorder kernel is matvec-only.
    body["q8_0_reorder"] = json{
        {"mode",          std::string{_engine->gpuOps().q8_0ReorderModeName()}},
        {"active",        _engine->gpuOps().q8_0ReorderTensorCount() > 0},
        {"tensor_count",  _engine->gpuOps().q8_0ReorderTensorCount()},
        {"total_bytes",   _engine->gpuOps().q8_0ReorderTotalBytes()},
    };

    return body;
}

json SystemStatusBuilder::buildPowerBlock() {
    if (_engine == nullptr) {
        return json{{"available", false},
                    {"reason", "pooled model-switch mode — no resident engine"}};
    }
    auto* mon = _engine->powerMonitor();
    if (mon == nullptr || !mon->available()) {
#ifdef MIMIRMIND_HAVE_CUDA
        // No RAPL power monitor (CUDA/GB10) — report the live GPU power draw via
        // NVML as a single "gpu" domain, matching the sysfs domains schema.
        const auto s = nvmlTelemetry().sample();
        if (s.power_w.has_value()) {
            // Name the domain "package-0" to match the RAPL-domain schema that
            // status clients (Pegenaut) read as the headline "power" figure; the
            // GB10 GPU power is effectively the whole-package draw here.
            json domain{
                {"name",      "package-0"},
                {"watts_now", *s.power_w},
            };
            // 5.17: cumulative GPU energy (W*s) when the driver exposes the
            // NVML energy counter — the GPU analogue of RAPL total_joules.
            // Pegenaut diffs this across a request for per-request energy.
            if (s.total_energy_mj.has_value()) {
                domain["total_joules"] =
                    static_cast<double>(*s.total_energy_mj) / 1000.0;
            }
            return json{
                {"available", true},
                {"source",    "nvml"},
                {"domains",   json::array({ std::move(domain) })},
            };
        }
#endif
        return json{
            {"available", false},
            {"reason",    mon == nullptr
                            ? std::string{"no power monitor installed"}
                            : std::string{mon->unavailableReason()}},
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