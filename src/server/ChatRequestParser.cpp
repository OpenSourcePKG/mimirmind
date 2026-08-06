// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "server/ChatRequestParser.hpp"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace mimirmind::server {

using nlohmann::json;

ChatRequest parseChatRequest(const json& body) {
    ChatRequest req;

    if (!body.is_object()) {
        throw std::runtime_error("request body must be a JSON object");
    }

    if (body.contains("model") && body["model"].is_string()) {
        req.model = body["model"].get<std::string>();
    }

    // Debug teacher-forcing (non-OpenAI): raw prefill suffix appended after
    // the generation prompt. See ChatRequest::assistantPrefill.
    if (body.contains("assistant_prefill") &&
        body["assistant_prefill"].is_string()) {
        req.assistantPrefill = body["assistant_prefill"].get<std::string>();
    }
    if (body.contains("assistant_prefill_ids") &&
        body["assistant_prefill_ids"].is_array()) {
        for (const auto& v : body["assistant_prefill_ids"]) {
            if (v.is_number_integer()) {
                req.assistantPrefillIds.push_back(v.get<std::int32_t>());
            }
        }
    }

    if (!body.contains("messages") || !body["messages"].is_array()) {
        throw std::runtime_error("messages: missing or not an array");
    }
    for (const auto& m : body["messages"]) {
        if (!m.is_object() || !m.contains("role")) {
            throw std::runtime_error("messages[]: each entry needs a role");
        }
        const auto roleStr = m["role"].get<std::string>();
        model::ChatRole role;
        if (!model::parseChatRole(roleStr, role)) {
            throw std::runtime_error(
                "messages[].role: unsupported value '" + roleStr + "'");
        }
        model::ChatMessage msg;
        msg.role = role;

        // Content is optional for an assistant turn that only carries
        // tool_calls (OpenAI sends content:null there); tolerated as empty
        // rather than a 400 for every other role too.
        if (m.contains("content")) {
            const auto& c = m["content"];
            if (c.is_string()) {
                msg.content = c.get<std::string>();
            } else if (!c.is_null()) {
                // OpenAI also accepts content arrays (multimodal). Not supported.
                throw std::runtime_error(
                    "messages[].content: only plain strings are supported");
            }
        }

        // M-FunctionCalling: prior assistant tool calls (multi-round replay).
        if (m.contains("tool_calls") && m["tool_calls"].is_array()) {
            for (const auto& tc : m["tool_calls"]) {
                if (!tc.is_object() || !tc.contains("function") ||
                    !tc["function"].is_object()) {
                    continue;
                }
                const auto& fn = tc["function"];
                model::ToolCall call;
                if (tc.contains("id") && tc["id"].is_string()) {
                    call.id = tc["id"].get<std::string>();
                }
                if (fn.contains("name") && fn["name"].is_string()) {
                    call.name = fn["name"].get<std::string>();
                }
                if (fn.contains("arguments")) {
                    // OpenAI stringifies arguments; tolerate an object too.
                    const auto& a = fn["arguments"];
                    call.argumentsJson =
                        a.is_string() ? a.get<std::string>() : a.dump();
                }
                msg.toolCalls.push_back(std::move(call));
            }
        }

        // M-FunctionCalling: role:"tool" result correlation id.
        if (m.contains("tool_call_id") && m["tool_call_id"].is_string()) {
            msg.toolCallId = m["tool_call_id"].get<std::string>();
        }

        req.messages.push_back(std::move(msg));
    }

    // M-FunctionCalling: function tool definitions + tool_choice.
    if (body.contains("tools") && body["tools"].is_array()) {
        for (const auto& t : body["tools"]) {
            if (!t.is_object()) {
                continue;
            }
            const bool isFunction =
                !t.contains("type") ||
                (t["type"].is_string() &&
                 t["type"].get<std::string>() == "function");
            if (!isFunction || !t.contains("function") ||
                !t["function"].is_object()) {
                continue;
            }
            const auto& fn = t["function"];
            if (!fn.contains("name") || !fn["name"].is_string()) {
                continue;
            }
            model::ToolSpec spec;
            spec.name     = fn["name"].get<std::string>();
            spec.toolJson = t.dump();   // spliced verbatim into <tools>
            req.tools.push_back(std::move(spec));
        }
    }
    if (body.contains("tool_choice") && body["tool_choice"].is_string()) {
        req.toolChoice = body["tool_choice"].get<std::string>();
    }

    auto readSize = [&](const char* key, std::size_t& dst) {
        if (body.contains(key) && body[key].is_number_integer()) {
            const auto v = body[key].get<std::int64_t>();
            if (v > 0) {
                dst = static_cast<std::size_t>(v);
            }
        }
    };
    auto readFloat = [&](const char* key, float& dst, bool& has) {
        if (body.contains(key) && body[key].is_number()) {
            dst = body[key].get<float>();
            has = true;
        }
    };

    // OpenAI: max_completion_tokens (current) overrides max_tokens (legacy).
    readSize("max_tokens", req.maxTokens);
    readSize("max_completion_tokens", req.maxTokens);
    readSize("top_k", req.topK);

    bool hasTopP = false;
    readFloat("temperature", req.temperature, req.hasTemperature);
    readFloat("top_p", req.topP, hasTopP);
    (void)hasTopP;

    readFloat("frequency_penalty",  req.frequencyPenalty,  req.hasFrequencyPenalty);
    readFloat("presence_penalty",   req.presencePenalty,   req.hasPresencePenalty);
    readFloat("repetition_penalty", req.repetitionPenalty, req.hasRepetitionPenalty);

    // Reasoning toggle (vLLM-compatible). Accept a top-level `enable_thinking`
    // and the nested `chat_template_kwargs: {enable_thinking: <bool>}`.
    if (body.contains("enable_thinking") && body["enable_thinking"].is_boolean()) {
        req.enableThinking = body["enable_thinking"].get<bool>();
    } else if (body.contains("chat_template_kwargs") &&
               body["chat_template_kwargs"].is_object()) {
        const auto& kwargs = body["chat_template_kwargs"];
        if (kwargs.contains("enable_thinking") &&
            kwargs["enable_thinking"].is_boolean()) {
            req.enableThinking = kwargs["enable_thinking"].get<bool>();
        }
    }

    if (body.contains("seed") && body["seed"].is_number_integer()) {
        req.seed = body["seed"].get<std::uint64_t>();
    }

    if (body.contains("stream") && body["stream"].is_boolean()) {
        req.stream = body["stream"].get<bool>();
    }

    if (body.contains("stop")) {
        const auto& s = body["stop"];
        if (s.is_string()) {
            req.stopStrings.push_back(s.get<std::string>());
        } else if (s.is_array()) {
            for (const auto& e : s) {
                if (e.is_string()) {
                    req.stopStrings.push_back(e.get<std::string>());
                }
            }
        }
    }

    return req;
}

} // namespace mimirmind::server