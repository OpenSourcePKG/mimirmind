// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "model/ChatTemplate.hpp"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace mimirmind::server {

/// Parsed OpenAI-style chat-completions request body.
///
/// The `has*` flags for penalties distinguish "client sent 0" from
/// "client sent nothing"; the latter picks up the server-side default
/// while the former stays at exactly 0.
struct ChatRequest {
    std::vector<model::ChatMessage> messages;
    std::size_t                     maxTokens{0};   // 0 => use server default
    float                           temperature{0.0F};
    bool                            hasTemperature{false};
    float                           topP{1.0F};
    std::size_t                     topK{0};
    std::uint64_t                   seed{0};
    std::vector<std::string>        stopStrings;
    bool                            stream{false};
    std::string                     model;

    // M-FunctionCalling. `tools` are the function definitions the client
    // offered (OpenAI `tools` array); when non-empty the chat template
    // renders them into the model's tool-spec block and the handler parses
    // the model's tool-call markup back into structured tool_calls.
    // `toolChoice` mirrors OpenAI: "auto" (default), "none", "required", or
    // a named function. Empty => "auto".
    std::vector<model::ToolSpec>    tools;
    std::string                     toolChoice;

    // Debug / parity teacher-forcing: raw text appended to the prompt AFTER
    // the chat template's generation prompt (no special tokens, no BOS), so
    // the engine prefills [template prompt + this text] in one T=N pass.
    // Lets the prefill path replay an exact previously-decoded token
    // sequence for prefill-vs-decode divergence localisation. Empty (the
    // default) => no-op, standard behaviour. Not part of the OpenAI schema.
    std::string                     assistantPrefill{};

    // Debug / parity teacher-forcing by raw token IDs (bypasses the
    // tokenizer entirely). Appended to the prompt after assistantPrefill, so
    // an exact previously-generated token sequence — including special tokens
    // like </think> that text cannot round-trip — can be replayed through the
    // prefill path. Empty (default) => no-op. Not part of the OpenAI schema.
    std::vector<std::int32_t>       assistantPrefillIds{};

    // M7f — repetition-control penalties.
    float                           frequencyPenalty{0.0F};
    bool                            hasFrequencyPenalty{false};
    float                           presencePenalty{0.0F};
    bool                            hasPresencePenalty{false};
    float                           repetitionPenalty{1.0F};
    bool                            hasRepetitionPenalty{false};

    // Reasoning toggle for "thinking" models (Qwen3.6). Mirrors vLLM's
    // `chat_template_kwargs: {enable_thinking: <bool>}`; also accepted as a
    // top-level `enable_thinking`. nullopt => architecture default (Qwen: think
    // on unless a tool round). false => force a direct answer (no reasoning),
    // which is what agentic RAG wants for its final answer turn.
    std::optional<bool>             enableThinking{std::nullopt};
};

/// Parse an OpenAI chat-completions request body. Throws std::runtime_error
/// on malformed input; the caller maps that to a 400 response.
[[nodiscard]] ChatRequest parseChatRequest(const nlohmann::json& body);

} // namespace mimirmind::server