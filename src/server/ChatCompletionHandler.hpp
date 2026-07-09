#pragma once

#include "server/ApiServer.hpp"

#include "model/ChatTemplate.hpp"
#include "runtime/InferenceEngine.hpp"

#include <httplib.h>

#include <cstdint>
#include <vector>

namespace mimirmind::server {

class RequestDispatcher;
class RequestTracker;
struct ChatRequest;
struct TrimReport;

/// Handles POST /v1/chat/completions — parses the request, resolves the
/// target engine via RequestDispatcher, applies M-PT prompt trimming,
/// dispatches through either the plain generate() path or the M9.11.4
/// spec-dec orchestrator, and writes either a JSON completion or an
/// SSE stream depending on the request's `stream` flag.
class ChatCompletionHandler {
public:
    ChatCompletionHandler(RequestDispatcher&        dispatcher,
                           RequestTracker&           tracker,
                           model::ChatTemplate::Style chatStyle,
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
        TrimReport&                    report);

    void handleBlocking(const ChatRequest& cr, httplib::Response& res);
    void handleStream  (const ChatRequest& cr, httplib::Response& res);

    RequestDispatcher&         _dispatcher;
    RequestTracker&            _tracker;
    model::ChatTemplate::Style _chatStyle;
    const ServerConfig&        _cfg;
};

} // namespace mimirmind::server