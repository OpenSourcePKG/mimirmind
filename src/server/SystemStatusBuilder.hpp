#pragma once

#include "runtime/PowerMonitor.hpp"

#include <nlohmann/json.hpp>

#include <chrono>
#include <mutex>
#include <string>
#include <string_view>

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
    SystemStatusBuilder(runtime::InferenceEngine& engine,
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

private:
    [[nodiscard]] nlohmann::json buildPerfRegressionBlock() const;
    [[nodiscard]] nlohmann::json buildGpuClockBlock() const;
    [[nodiscard]] nlohmann::json buildFanBlock() const;
    [[nodiscard]] nlohmann::json buildKernelsBlock() const;
    [[nodiscard]] nlohmann::json buildPowerBlock();

    runtime::InferenceEngine& _engine;
    RequestDispatcher&        _dispatcher;
    RequestTracker&           _requestTracker;
    std::string               _modelId;

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