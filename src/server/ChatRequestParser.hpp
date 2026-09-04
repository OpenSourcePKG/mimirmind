// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "model/ChatTemplate.hpp"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace mimirmind::server {

/// OpenAI `response_format.type`. Parsed + carried so the handler/sampler can
/// honor it where a constraint mechanism exists; today it is best-effort
/// (no grammar-constrained decoding yet — see 8.19 Increment 2), so JsonObject/
/// JsonSchema are accepted and recorded but not hard-enforced.
enum class ResponseFormat { Text, JsonObject, JsonSchema };

/// Thrown by parseChatRequest when a request field carries a malformed value
/// (wrong JSON type / out of range). `param` names the offending field so the
/// caller can populate the OpenAI error `param`; empty when not field-specific.
/// A missing OR non-applicable standard field is NOT an error (parse skips it),
/// so this only ever fires on a value the client actually got wrong.
class ChatRequestError : public std::runtime_error {
public:
    explicit ChatRequestError(const std::string& message, std::string param = {})
        : std::runtime_error(message), _param(std::move(param)) {}

    [[nodiscard]] const std::string& param() const noexcept { return _param; }

private:
    std::string _param;
};

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
    // 8.19.7: explicit-presence flags — a request that asks for sampling
    // (temperature>0) WITHOUT its own top_p/top_k gets the model's
    // generation_config.json truncation defaults (vLLM behaviour); an
    // explicit value, even the neutral one, always wins.
    bool                            hasTopP{false};
    bool                            hasTopK{false};
    std::uint64_t                   seed{0};
    std::vector<std::string>        stopStrings;
    bool                            stream{false};
    // 8.19.12 — OpenAI `n`: number of choices to generate for this request.
    // v1 supports n>1 only on the blocking continuous-batching path without
    // tools (the handler 400s otherwise); the extra choices run as parallel
    // batcher submissions. Client-side self-consistency checks are the
    // motivating consumer (sample n=3, compare divergence).
    std::size_t                     n{1};
    // OpenAI `stream_options.include_usage`: when streaming, emit a terminal
    // chunk carrying `usage` (with an empty `choices`) just before [DONE].
    bool                            includeUsage{false};
    std::string                     model;

    // M-FunctionCalling. `tools` are the function definitions the client
    // offered (OpenAI `tools` array); when non-empty the chat template
    // renders them into the model's tool-spec block and the handler parses
    // the model's tool-call markup back into structured tool_calls.
    // `toolChoice` mirrors OpenAI: "auto" (default), "none", "required", or
    // a named function. Empty => "auto".
    std::vector<model::ToolSpec>    tools;
    std::string                     toolChoice;

    // OpenAI named-tool force: tool_choice object
    // `{type:"function",function:{name:"x"}}`. When set, the parser restricts
    // `tools` to exactly this function and sets `toolChoice = "required"`, so
    // the existing required-opener/prefill mechanic forces that one call.
    // Empty => no named force.
    std::string                     forcedToolName;

    // OpenAI `response_format`. Default Text (unconstrained). JsonObject /
    // JsonSchema are parsed + validated and carried here, but enforcement
    // (grammar-constrained decoding) is not yet implemented — best-effort:
    // the field is recorded, never faked in the response. See 8.19 Increment 2.
    ResponseFormat                  responseFormat{ResponseFormat::Text};

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
    // top-level `enable_thinking`. nullopt => server default = thinking OFF
    // (direct answer — fast, no thinking-loop surface); set true to opt INTO
    // reasoning. false explicitly forces a direct answer (agentic RAG final turn).
    std::optional<bool>             enableThinking{std::nullopt};
};

/// Parse an OpenAI chat-completions request body. Throws std::runtime_error
/// on malformed input; the caller maps that to a 400 response.
[[nodiscard]] ChatRequest parseChatRequest(const nlohmann::json& body);

} // namespace mimirmind::server