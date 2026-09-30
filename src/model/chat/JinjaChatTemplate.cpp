// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "model/chat/JinjaChatTemplate.hpp"

#include <minja/chat-template.hpp>   // third_party/minja — pulls minja.hpp + nlohmann

#include <nlohmann/json.hpp>

#include <utility>

namespace mimirmind::model::chat {

namespace {

using json = nlohmann::ordered_json;

/// Parse a JSON *string* into a value; on any parse error return the raw string
/// so a malformed field degrades to text instead of throwing. Tool-call
/// arguments and tool schemas reach us as already-validated JSON strings, but a
/// permissive fallback keeps rendering robust.
json parseOrString(const std::string& s) {
    if (s.empty()) {
        return json::object();
    }
    return json::parse(s, /*cb=*/nullptr, /*allow_exceptions=*/false);
}

/// Build the OpenAI/HF-shaped messages array minja binds as `messages`.
json buildMessages(std::span<const ChatMessage> messages) {
    json arr = json::array();
    for (const ChatMessage& m : messages) {
        json jm = json::object();
        jm["role"]    = std::string{chatRoleName(m.role)};
        jm["content"] = m.content;

        // Assistant tool calls → tool_calls[].function.{name,arguments}.
        // arguments is an OBJECT (HF/vLLM convention; templates `tojson` it).
        if (!m.toolCalls.empty()) {
            json calls = json::array();
            for (const ToolCall& tc : m.toolCalls) {
                json args = parseOrString(tc.argumentsJson);
                if (args.is_discarded()) {
                    args = tc.argumentsJson;   // keep raw on parse failure
                }
                json call = json::object();
                if (!tc.id.empty()) {
                    call["id"] = tc.id;
                }
                call["type"]     = "function";
                call["function"] = json::object({{"name", tc.name},
                                                 {"arguments", args}});
                calls.push_back(std::move(call));
            }
            jm["tool_calls"] = std::move(calls);
        }

        // Tool result turn → correlate with the assistant call.
        if (m.role == ChatRole::Tool && !m.toolCallId.empty()) {
            jm["tool_call_id"] = m.toolCallId;
        }

        arr.push_back(std::move(jm));
    }
    return arr;
}

/// Build the tools array from each ToolSpec's (already-validated) toolJson.
json buildTools(std::span<const ToolSpec> tools) {
    json arr = json::array();
    for (const ToolSpec& t : tools) {
        json parsed = parseOrString(t.toolJson);
        if (parsed.is_discarded()) {
            continue;   // skip an unparseable tool rather than corrupt the list
        }
        arr.push_back(std::move(parsed));
    }
    return arr;
}

} // namespace

JinjaChatTemplate::JinjaChatTemplate(std::string chatTemplate,
                                     std::string bosToken,
                                     std::string eosToken)
    : _tmpl(std::make_unique<minja::chat_template>(std::move(chatTemplate),
                                                   std::move(bosToken),
                                                   std::move(eosToken))) {}

JinjaChatTemplate::~JinjaChatTemplate()                                = default;
JinjaChatTemplate::JinjaChatTemplate(JinjaChatTemplate&&) noexcept     = default;
JinjaChatTemplate& JinjaChatTemplate::operator=(JinjaChatTemplate&&) noexcept = default;

std::string JinjaChatTemplate::render(std::span<const ChatMessage> messages,
                                      std::span<const ToolSpec>    tools,
                                      bool                         addGenerationPrompt,
                                      std::optional<bool>          enableThinking) const {
    minja::chat_template_inputs in;
    in.messages              = buildMessages(messages);
    in.add_generation_prompt = addGenerationPrompt;
    if (!tools.empty()) {
        in.tools = buildTools(tools);
    }
    if (enableThinking.has_value()) {
        // chat_template_kwargs-style pass-through (like vLLM): the model's
        // template reads `enable_thinking` from the context.
        in.extra_context = nlohmann::ordered_json::object(
            {{"enable_thinking", *enableThinking}});
    }
    return _tmpl->apply(in);
}

} // namespace mimirmind::model::chat
