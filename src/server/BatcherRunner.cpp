// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "server/BatcherRunner.hpp"

#include "model/ToolCallConstraint.hpp"
#include "runtime/serving/ContinuousBatcher.hpp"

#include <mutex>

namespace mimirmind::server {

std::vector<std::int32_t> BatcherRunner::run(
        runtime::serving::ContinuousBatcher&      batcher,
        std::vector<std::int32_t>                 promptIds,
        const runtime::GenerateParams&            params,
        std::vector<std::int32_t>                 stopIds,
        std::string                               tenantId,
        const std::function<bool(std::int32_t)>&  onToken,
        std::shared_ptr<model::ToolCallConstraint> constraint,
        std::vector<runtime::TokenLogprobs>*       outLp,
        const std::function<void(const runtime::TokenLogprobs&)>* onLp,
        runtime::GenerateStats*                    outStats,
        const std::function<void(const runtime::InferenceEngine::PrefillDone&)>*
                                                   onPrefillDone) {
    // 8.19.5: hand the request's sampling params to the batcher so the slot
    // decodes with them (temperature<=0 stays the greedy fast path).
    // 8.19.13.2: an optional tool-call grammar constraint rides along.
    auto req = batcher.submit(std::move(promptIds), params.maxNewTokens,
                              std::move(stopIds), std::move(tenantId),
                              params.sampling, std::move(constraint));
    std::vector<std::int32_t> out;
    std::size_t  next = 0;
    std::int32_t t    = 0;
    bool aborted = false;
    bool prefillReported = false;
    while (req->waitToken(next, t)) {
        out.push_back(t);
        // First token means prefill is done — surface the batcher's prefill
        // telemetry (cached vs freshly-prefilled prompt tokens + prefill wall
        // time) so a streaming client can emit the prefill_done event, the same
        // as the single-session path. Fired once.
        if (!prefillReported) {
            prefillReported = true;
            if (onPrefillDone != nullptr && *onPrefillDone) {
                runtime::InferenceEngine::PrefillDone pd{};
                {
                    std::lock_guard<std::mutex> lk(req->mtx);
                    pd.promptTokens    = req->promptTokens;
                    pd.prefilledTokens = req->prefilledTokens;
                    pd.prefillMs       = req->prefillMs;
                }
                (*onPrefillDone)(pd);
            }
        }
        // 8.19.14 part B — per-token logprob callback (streaming). req->logprobs
        // is appended under the same lock as the token, so index `next` is ready.
        if (onLp != nullptr && *onLp) {
            std::lock_guard<std::mutex> lk(req->mtx);
            if (next < req->logprobs.size()) { (*onLp)(req->logprobs[next]); }
        }
        ++next;
        if (onToken && !onToken(t)) { aborted = true; break; }
    }
    if (aborted) {
        batcher.cancel(req);
    }
    // Hand the prefill/cache telemetry back into GenerateStats so the serving
    // path reports the same cached_tokens / prefill_ms the single-session path
    // does (usage.prompt_tokens_details.cached_tokens + the completion log).
    if (outStats != nullptr) {
        std::lock_guard<std::mutex> lk(req->mtx);
        if (req->prefillSet) {
            outStats->promptTokens = req->promptTokens;
            outStats->cachedTokens = req->cachedTokens;
            outStats->prefillMs    = req->prefillMs;
        }
    }
    // 8.19.14 part B — hand back the per-token logprobs (aligned with `out`);
    // populated only when the request enabled logprobs.
    if (outLp != nullptr) {
        std::lock_guard<std::mutex> lk(req->mtx);
        *outLp = req->logprobs;
    }
    if (!req->error.empty()) {
        // Per-tenant quota is checked before the whole-server overload: both
        // set `error`, but a quota rejection is the caller's own fault (429),
        // not server saturation (503).
        if (req->tenantQuotaExceeded) {
            throw runtime::serving::ServingTenantQuotaError(req->error);
        }
        if (req->overloaded) {
            throw runtime::serving::ServingOverloadedError(req->error);
        }
        throw std::runtime_error(req->error);
    }
    return out;
}

} // namespace mimirmind::server
