// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "server/ApiServer.hpp"

#include "model/ChatTemplate.hpp"
#include "runtime/InferenceEngine.hpp"

#include <httplib.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace mimirmind::model {
class Tokenizer;
} // namespace mimirmind::model

namespace mimirmind::runtime {
class SpeculativeDecoder;
namespace serving { class ContinuousBatcher; }
} // namespace mimirmind::runtime

namespace mimirmind::server {

class RequestDispatcher;
class RequestTracker;
class TenantMetrics;
struct ChatRequest;
struct TrimReport;
struct ChatStreamState;

/// Handles POST /v1/chat/completions — parses the request, resolves the
/// target engine via RequestDispatcher, applies M-PT prompt trimming,
/// dispatches through either the plain generate() path or the M9.11.4
/// spec-dec orchestrator, and writes either a JSON completion or an
/// SSE stream depending on the request's `stream` flag.
class ChatCompletionHandler {
public:
    ChatCompletionHandler(RequestDispatcher&        dispatcher,
                           RequestTracker&           tracker,
                           TenantMetrics&            metrics,
                           const ServerConfig&        cfg);

    ChatCompletionHandler(const ChatCompletionHandler&)            = delete;
    ChatCompletionHandler& operator=(const ChatCompletionHandler&) = delete;
    ChatCompletionHandler(ChatCompletionHandler&&)                 = delete;
    ChatCompletionHandler& operator=(ChatCompletionHandler&&)      = delete;

    void handle(const httplib::Request& req, httplib::Response& res);

private:
    [[nodiscard]] bool prepareChatRequest(
        runtime::InferenceEngine&      targetEngine,
        const ChatRequest&             cr,
        httplib::Response&             res,
        std::vector<std::int32_t>&     promptIds,
        std::vector<std::int32_t>&     stopIds,
        runtime::GenerateParams&       params,
        TrimReport&                    report,
        std::string&                   forcedToolOpener);

    /// The sampling-policy block of prepareChatRequest, extracted verbatim
    /// (8.30.5, pure move): client sampling params + model-default overlay,
    /// tool-loop greedy clamp, thinking / answer anti-loop floors, and the M7f
    /// repetition penalties + safety floor. Mutates only `params.sampling`.
    void applySamplingPolicy(runtime::InferenceEngine& targetEngine,
                             const ChatRequest&        cr,
                             const model::Tokenizer&   tok,
                             runtime::GenerateParams&  params);

    void handleBlocking(const ChatRequest& cr, httplib::Response& res);
    void handleStream  (const ChatRequest& cr, httplib::Response& res);

    /// The SSE chunked-content-provider body of handleStream, extracted verbatim
    /// (roadmap 8.30.11.2). Runs the whole streaming state machine — role chunk,
    /// the decode loop's per-token callbacks (content / reasoning UTF-8-hold,
    /// tool-call detection, held-content), the batcher/engine dispatch, and the
    /// end-of-turn tool-call finalize + finish + [DONE] — for one request. The
    /// former inline lambda's captures are now parameters; `state` (with its
    /// `pin`) keeps the target pointers valid across the async provider.
    /// Behaviour-neutral: the body is byte-identical to the former lambda.
    bool runStreamSession(std::shared_ptr<ChatStreamState>     state,
                          runtime::InferenceEngine*            targetEngine,
                          std::mutex*                          targetMutex,
                          runtime::SpeculativeDecoder*         targetSpec,
                          runtime::serving::ContinuousBatcher* targetBatcher,
                          httplib::DataSink&                   sink);

    RequestDispatcher&         _dispatcher;
    RequestTracker&            _tracker;
    TenantMetrics&             _metrics;
    const ServerConfig&        _cfg;
};

} // namespace mimirmind::server