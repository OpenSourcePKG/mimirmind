// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "server/ChatResponseBuilder.hpp"

#include "server/PromptTrimmer.hpp"

#include "runtime/InferenceEngine.hpp"

namespace mimirmind::server {

using nlohmann::json;

json ChatResponseBuilder::buildUsage(std::size_t                   promptTokens,
                                     std::size_t                   completionTokens,
                                     const runtime::GenerateStats& stats,
                                     const TrimReport&             trimReport) {
    json usage = {
        {"prompt_tokens",     promptTokens},
        {"completion_tokens", completionTokens},
        {"total_tokens",      promptTokens + completionTokens},
    };
    // OpenAI/vLLM-compatible prefix-cache visibility: how many prompt tokens were
    // served from the cache (warm-slot / GDN prefix reuse on the serving path, or
    // the single-session LCP hit) and thus skipped prefill. Lets a client
    // (Pegenaut) see cache hits + prefill cost instead of the old always-0/0.
    // `cachedTokens` is populated by both paths via GenerateStats.
    usage["prompt_tokens_details"] = { {"cached_tokens", stats.cachedTokens} };
    if (stats.prefillMs > 0.0) {
        usage["prefill_ms"] = stats.prefillMs;   // mimirmind extension
    }
    // Extension over the OpenAI shape: per-request energy delta from the RAPL
    // package counter. Quietly omitted when no power monitor was active.
    if (stats.packageJoules > 0.0) {
        usage["package_joules"] = stats.packageJoules;
    }
    // M-PT — length-discipline metadata. Only present when trim / clamp /
    // extrapolation-warn actually fired.
    PromptTrimmer::attachTrimUsage(usage, trimReport);
    return usage;
}

void ChatResponseBuilder::addCompletionTokens(json& usage, std::size_t extra) {
    if (usage.contains("completion_tokens")) {
        usage["completion_tokens"] =
            usage["completion_tokens"].get<std::int64_t>()
            + static_cast<std::int64_t>(extra);
    }
    if (usage.contains("total_tokens")) {
        usage["total_tokens"] =
            usage["total_tokens"].get<std::int64_t>()
            + static_cast<std::int64_t>(extra);
    }
}

json ChatResponseBuilder::buildAssistantMessage(
    std::string_view                    content,
    const std::vector<model::ToolCall>& toolCalls,
    std::string_view                    reasoning) {
    // `refusal` is part of the OpenAI assistant message shape; null unless the
    // model produced a safety refusal (mimirmind does not emit one today).
    json message = {{"role", "assistant"}, {"refusal", nullptr}};
    // Surface the model's thinking separately (vLLM / llama.cpp shape). Present
    // on both the content answer and a tool-call turn — a reasoning model thinks
    // before it decides to call a tool, and clients show that.
    if (!reasoning.empty()) {
        message["reasoning_content"] = std::string(reasoning);
    }
    if (toolCalls.empty()) {
        message["content"] = std::string(content);
    } else {
        // OpenAI shape: content null, calls under tool_calls[]. arguments is a
        // JSON *string* (already normalised by the parser).
        message["content"] = nullptr;
        json tcArr = json::array();
        for (const auto& call : toolCalls) {
            tcArr.push_back({
                {"id",   call.id},
                {"type", "function"},
                {"function", {
                    {"name",      call.name},
                    {"arguments", call.argumentsJson},
                }},
            });
        }
        message["tool_calls"] = std::move(tcArr);
    }
    return message;
}

json ChatResponseBuilder::buildCompletion(const std::string& id,
                                          std::int64_t       created,
                                          const std::string& model,
                                          json               message,
                                          std::string_view   finishReason,
                                          json               logprobs,
                                          json               usage) {
    return json{
        {"id",      id},
        {"object",  "chat.completion"},
        {"created", created},
        {"model",   model},
        {"choices", json::array({
            json{
                {"index", 0},
                {"message", std::move(message)},
                {"finish_reason", std::string(finishReason)},
                // 8.19.14 part B — OpenAI logprobs (null unless requested).
                {"logprobs", std::move(logprobs)},
            },
        })},
        {"usage", std::move(usage)},
    };
}

} // namespace mimirmind::server
