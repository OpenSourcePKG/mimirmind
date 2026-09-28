// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "model/ToolCall.hpp"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace mimirmind::runtime {
struct GenerateStats;
} // namespace mimirmind::runtime

namespace mimirmind::server {

struct TrimReport;

/// 8.30.5 — builds the non-streaming `chat.completion` JSON for
/// handleBlocking: the OpenAI usage object (built + patched in more than one
/// place before), the assistant message (content vs tool_calls + reasoning),
/// and the top-level completion envelope. Streaming keeps routing its chunks
/// through {@link SseEncoder}; this builder is blocking-path only.
class ChatResponseBuilder {
public:
    ChatResponseBuilder() = delete;

    /// The OpenAI usage object: prompt/completion/total token counts,
    /// `prompt_tokens_details.cached_tokens`, the optional mimirmind
    /// `prefill_ms` / `package_joules` extensions, and the M-PT trim metadata
    /// (attached only when trim / clamp / extrapolation actually fired).
    [[nodiscard]] static nlohmann::json buildUsage(
        std::size_t                    promptTokens,
        std::size_t                    completionTokens,
        const runtime::GenerateStats&  stats,
        const TrimReport&              trimReport);

    /// Fold an extra choice's completion tokens into an existing usage object
    /// (OpenAI `n`): bumps `completion_tokens` and `total_tokens` in place,
    /// each guarded by a contains-check exactly as the inline code did.
    static void addCompletionTokens(nlohmann::json& usage, std::size_t extra);

    /// The assistant message object. With no tool calls it carries `content`
    /// (the visible text); with tool calls `content` is null and the calls ride
    /// under `tool_calls[]`. `reasoning` (when non-empty) is surfaced as
    /// `reasoning_content` on either shape. `refusal` is always present (null).
    [[nodiscard]] static nlohmann::json buildAssistantMessage(
        std::string_view                        content,
        const std::vector<model::ToolCall>&     toolCalls,
        std::string_view                        reasoning);

    /// The top-level `chat.completion` envelope with a single choice (index 0).
    /// `message`, `usage` and `logprobs` are moved/copied in verbatim so the
    /// caller retains full control of their contents; extra `n` choices are
    /// appended to `response["choices"]` by the caller afterward.
    [[nodiscard]] static nlohmann::json buildCompletion(
        const std::string& id,
        std::int64_t       created,
        const std::string& model,
        nlohmann::json     message,
        std::string_view   finishReason,
        nlohmann::json     logprobs,
        nlohmann::json     usage);
};

} // namespace mimirmind::server
