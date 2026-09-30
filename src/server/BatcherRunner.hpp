// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "runtime/InferenceEngine.hpp"   // GenerateParams, GenerateStats, PrefillDone
#include "runtime/TokenLogprobs.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace mimirmind::model {
class ToolCallConstraint;
}
namespace mimirmind::runtime::serving {
class ContinuousBatcher;
}

namespace mimirmind::server {

/// Drives one request through the ContinuousBatcher, mirroring the callback
/// contract of InferenceEngine::generate() so the response formatting is reused
/// verbatim (M-Cuda.Batch D2e.2). Submits the prompt, delivers each generated
/// token to `onToken` in order (cancels on false), and hands back prefill/cache
/// telemetry + per-token logprobs. Throws ServingTenantQuotaError /
/// ServingOverloadedError / runtime_error on the request's error. Extracted
/// verbatim from the former free `runViaBatcher` (roadmap 8.30.11.2).
class BatcherRunner {
public:
    [[nodiscard]] static std::vector<std::int32_t> run(
        runtime::serving::ContinuousBatcher&       batcher,
        std::vector<std::int32_t>                  promptIds,
        const runtime::GenerateParams&             params,
        std::vector<std::int32_t>                  stopIds,
        std::string                                tenantId,
        const std::function<bool(std::int32_t)>&   onToken,
        std::shared_ptr<model::ToolCallConstraint> constraint = nullptr,
        std::vector<runtime::TokenLogprobs>*       outLp = nullptr,
        const std::function<void(const runtime::TokenLogprobs&)>* onLp = nullptr,
        runtime::GenerateStats*                    outStats = nullptr,
        const std::function<void(const runtime::InferenceEngine::PrefillDone&)>*
                                                   onPrefillDone = nullptr);
};

} // namespace mimirmind::server
