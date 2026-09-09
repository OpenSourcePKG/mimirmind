// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "runtime/thermal/PowerMonitor.hpp"
#include "server/ModelProvider.hpp"   // ResidentModelMemory

#include <nlohmann/json.hpp>

#include <chrono>
#include <functional>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace mimirmind::runtime {
class InferenceEngine;
}

namespace mimirmind::server {

class RequestDispatcher;
class RequestTracker;

/// Builds the JSON payloads for `/v1/system/info` (static) and
/// `/v1/system/status` (dynamic).
///
/// Owns the RAPL power baseline captured at construction — subsequent
/// `buildStatus()` calls report deltas against it, giving the operator
/// "energy since server-up" plus rolling watts against the previous
/// status poll.
class SystemStatusBuilder {
public:
    /// `engine` may be nullptr in M-Munin.3 per-request model-switch (pool)
    /// mode — there is no always-resident default engine. In that case the
    /// info/status payloads report a reduced "pooled_model_switch" object and
    /// the per-engine detail blocks report {available:false}. Eager (co-
    /// resident) mode passes the default engine and behaves as before.
    SystemStatusBuilder(runtime::InferenceEngine* engine,
                         RequestDispatcher&        dispatcher,
                         RequestTracker&           requestTracker,
                         std::string_view          modelId);

    SystemStatusBuilder(const SystemStatusBuilder&)            = delete;
    SystemStatusBuilder& operator=(const SystemStatusBuilder&) = delete;
    SystemStatusBuilder(SystemStatusBuilder&&)                 = delete;
    SystemStatusBuilder& operator=(SystemStatusBuilder&&)      = delete;

    /// Payload for GET /v1/system/info — everything fixed for the
    /// lifetime of the process.
    [[nodiscard]] nlohmann::json buildInfo() const;

    /// Payload for GET /v1/system/status — everything that changes at
    /// runtime.
    [[nodiscard]] nlohmann::json buildStatus();

    /// Payload for the admin-gated GET /v1/system/memory — categorized
    /// RAM/VRAM breakdown (weights / paged-KV / device envelope / external
    /// residual) + allocator fragmentation where available (8.16 Stage A).
    [[nodiscard]] nlohmann::json buildMemory() const;

    /// Register a provider of non-generative-pool models (embedding / rerank
    /// encoder engines) so `buildMemory()` enumerates them per-model in
    /// `models.resident[]` alongside the chat engine, instead of leaving their
    /// weights folded only into the aggregate allocator total. Set once by
    /// ApiServer after the handlers are constructed; may return an empty
    /// vector when no encoder engines are loaded.
    void setAuxModelMemoryProvider(
        std::function<std::vector<ResidentModelMemory>()> provider) {
        _auxModelMemory = std::move(provider);
    }

private:
    [[nodiscard]] nlohmann::json buildPerfRegressionBlock() const;
    [[nodiscard]] nlohmann::json buildGpuClockBlock() const;
    [[nodiscard]] nlohmann::json buildFanBlock() const;
    [[nodiscard]] nlohmann::json buildKernelsBlock() const;
    [[nodiscard]] nlohmann::json buildPowerBlock();

    runtime::InferenceEngine* _engine;    // nullptr in pool (model-switch) mode
    RequestDispatcher&        _dispatcher;
    RequestTracker&           _requestTracker;
    std::string               _modelId;

    // Non-pool encoder engines (embedding/rerank) memory snapshot provider.
    // Empty until ApiServer wires it; returns an empty vector if unset.
    std::function<std::vector<ResidentModelMemory>()> _auxModelMemory{};

    // RAPL baseline snapshot taken at construction — represents "engine
    // idle, server warmed up" since ApiServer wires this up after the
    // engine has finished loadModel().
    mutable std::mutex                    _powerStateMutex;
    runtime::PowerMonitor::Snapshot       _powerBaseline{};
    runtime::PowerMonitor::Snapshot       _powerLastStatus{};
    std::chrono::steady_clock::time_point _baselineWallStart{};
    bool                                  _baselineCaptured{false};
};

} // namespace mimirmind::server