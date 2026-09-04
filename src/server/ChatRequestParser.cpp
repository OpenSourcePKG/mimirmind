// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "server/ChatRequestParser.hpp"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace mimirmind::server {

using nlohmann::json;

namespace {

// --- robustness helpers ---------------------------------------------------
//
// "Be liberal in what you accept": a MISSING (or explicit-null) optional field
// returns nullopt so the caller keeps its default — never an error. A field
// the client DID send but with the wrong JSON type raises a ChatRequestError
// naming the field, which the handler maps to a 400 with `param` set. This
// replaces the bare typed nlohmann accessors (`body[k].get<T>()`) that would
// otherwise throw an opaque type_error and structurally eliminates the whole
// "unexpected JSON type -> 400/500" class.

[[noreturn]] void badType(std::string_view key, std::string_view expected) {
    throw ChatRequestError(std::string{key} + " must be " + std::string{expected},
                           std::string{key});
}

[[nodiscard]] bool present(const json& body, const char* key) {
    return body.contains(key) && !body[key].is_null();
}

[[nodiscard]] std::optional<std::string>
optString(const json& body, const char* key) {
    if (!present(body, key)) return std::nullopt;
    if (!body[key].is_string()) badType(key, "a string");
    return body[key].get<std::string>();
}

[[nodiscard]] std::optional<bool> optBool(const json& body, const char* key) {
    if (!present(body, key)) return std::nullopt;
    if (!body[key].is_boolean()) badType(key, "a boolean");
    return body[key].get<bool>();
}

// Integer-valued fields (max_tokens, top_k, seed, ...). A JSON float is a wrong
// type for an integer field -> 400 (OpenAI rejects it too).
[[nodiscard]] std::optional<std::int64_t>
optInt(const json& body, const char* key) {
    if (!present(body, key)) return std::nullopt;
    if (!body[key].is_number_integer()) badType(key, "an integer");
    return body[key].get<std::int64_t>();
}

// Real-valued fields (temperature, top_p, penalties). Accepts int or float.
[[nodiscard]] std::optional<double>
optNumber(const json& body, const char* key) {
    if (!present(body, key)) return std::nullopt;
    if (!body[key].is_number()) badType(key, "a number");
    return body[key].get<double>();
}

// A positive-integer size field: applied only when > 0 (0/negative keep the
// server default, matching the previous behaviour), but a non-integer value is
// still a hard 400.
void readSize(const json& body, const char* key, std::size_t& dst) {
    if (const auto v = optInt(body, key); v && *v > 0) {
        dst = static_cast<std::size_t>(*v);
    }
}

void readFloat(const json& body, const char* key, float& dst, bool& has) {
    if (const auto v = optNumber(body, key)) {
        dst = static_cast<float>(*v);
        has = true;
    }
}

// tool_choice accepts BOTH the string form (none|auto|required) and the OpenAI
// named-force object `{type:"function",function:{name:"x"}}`. The object sets
// forcedToolName + toolChoice="required" so the existing required-opener
// mechanic forces exactly that call (the toolset is narrowed to it later).
void parseToolChoice(const json& body, ChatRequest& req) {
    if (!present(body, "tool_choice")) return;
    const auto& tc = body["tool_choice"];

    if (tc.is_string()) {
        // none|auto|required. An unknown string is treated leniently as "auto"
        // (only "none"/"required" carry behaviour downstream) — not a 400.
        req.toolChoice = tc.get<std::string>();
        return;
    }
    if (tc.is_object()) {
        std::string name;
        if (tc.contains("function") && tc["function"].is_object()) {
            const auto& fn = tc["function"];
            if (fn.contains("name") && fn["name"].is_string()) {
                name = fn["name"].get<std::string>();
            }
        }
        if (name.empty()) {
            throw ChatRequestError(
                "tool_choice object must be "
                "{type:\"function\",function:{name:...}}",
                "tool_choice");
        }
        req.forcedToolName = name;
        req.toolChoice     = "required";
        return;
    }
    throw ChatRequestError(
        "tool_choice must be a string (none|auto|required) or a "
        "{type:\"function\",function:{name}} object",
        "tool_choice");
}

} // namespace

ChatRequest parseChatRequest(const json& body) {
    ChatRequest req;

    if (!body.is_object()) {
        throw ChatRequestError("request body must be a JSON object");
    }

    if (const auto v = optString(body, "model")) {
        req.model = *v;
    }

    // Debug teacher-forcing (non-OpenAI): raw prefill suffix appended after
    // the generation prompt. See ChatRequest::assistantPrefill.
    if (const auto v = optString(body, "assistant_prefill")) {
        req.assistantPrefill = *v;
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
        throw ChatRequestError("messages: missing or not an array", "messages");
    }
    for (const auto& m : body["messages"]) {
        if (!m.is_object() || !m.contains("role")) {
            throw ChatRequestError("messages[]: each entry needs a role",
                                   "messages");
        }
        if (!m["role"].is_string()) {
            throw ChatRequestError("messages[].role must be a string",
                                   "messages[].role");
        }
        std::string roleStr = m["role"].get<std::string>();
        // OpenAI `developer` role == a higher-priority system message; map it
        // to system (the engine has no separate developer channel).
        if (roleStr == "developer") {
            roleStr = "system";
        }
        model::ChatRole role;
        if (!model::parseChatRole(roleStr, role)) {
            throw ChatRequestError(
                "messages[].role: unsupported value '" + roleStr + "'",
                "messages[].role");
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
            } else if (c.is_array()) {
                // OpenAI multimodal content parts: concatenate the text parts;
                // ignore image/audio/other parts (vision/audio inference is out
                // of scope). A well-formed array is never a 400.
                std::string text;
                for (const auto& part : c) {
                    if (part.is_object() &&
                        part.value("type", std::string{}) == "text" &&
                        part.contains("text") && part["text"].is_string()) {
                        text += part["text"].get<std::string>();
                    }
                }
                msg.content = std::move(text);
            } else if (!c.is_null()) {
                // A scalar that is neither string nor array (number/bool) is
                // genuinely malformed.
                throw ChatRequestError(
                    "messages[].content must be a string or an array of "
                    "content parts",
                    "messages[].content");
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
    parseToolChoice(body, req);

    // Deprecated OpenAI `functions` / `function_call` -> tools / tool_choice.
    // Only used as a fallback: the modern `tools`/`tool_choice` win if present.
    if (req.tools.empty() && body.contains("functions") &&
        body["functions"].is_array()) {
        for (const auto& fn : body["functions"]) {
            if (!fn.is_object() || !fn.contains("name") ||
                !fn["name"].is_string()) {
                continue;
            }
            model::ToolSpec spec;
            spec.name = fn["name"].get<std::string>();
            // Wrap in the modern tool shape so it renders in <tools> like a
            // native tool definition.
            spec.toolJson =
                json{{"type", "function"}, {"function", fn}}.dump();
            req.tools.push_back(std::move(spec));
        }
    }
    if (req.toolChoice.empty() && req.forcedToolName.empty() &&
        present(body, "function_call")) {
        const auto& fc = body["function_call"];
        if (fc.is_string()) {
            req.toolChoice = fc.get<std::string>();      // none|auto
        } else if (fc.is_object() && fc.contains("name") &&
                   fc["name"].is_string()) {
            req.forcedToolName = fc["name"].get<std::string>();
            req.toolChoice     = "required";
        } else {
            throw ChatRequestError(
                "function_call must be \"none\"/\"auto\" or {name:...}",
                "function_call");
        }
    }

    // Named tool force: narrow the offered toolset to exactly the forced
    // function so the model can only emit that call (paired with the
    // required-opener prefill via toolChoice=="required"). Forcing a function
    // that was not offered in `tools` is a genuine malformed request.
    if (!req.forcedToolName.empty()) {
        std::vector<model::ToolSpec> only;
        for (auto& s : req.tools) {
            if (s.name == req.forcedToolName) {
                only.push_back(std::move(s));
            }
        }
        if (only.empty()) {
            throw ChatRequestError(
                "tool_choice forces function '" + req.forcedToolName +
                "' which is not present in tools",
                "tool_choice");
        }
        req.tools = std::move(only);
    }

    // OpenAI: max_completion_tokens (current) overrides max_tokens (legacy).
    readSize(body, "max_tokens", req.maxTokens);
    readSize(body, "max_completion_tokens", req.maxTokens);
    readSize(body, "n", req.n);                       // 8.19.12; 0 => 1 below
    if (req.n == 0) { req.n = 1; }
    readSize(body, "top_k", req.topK);
    req.hasTopK = body.contains("top_k") && !body["top_k"].is_null();

    readFloat(body, "temperature", req.temperature, req.hasTemperature);
    readFloat(body, "top_p", req.topP, req.hasTopP);

    readFloat(body, "frequency_penalty",  req.frequencyPenalty,  req.hasFrequencyPenalty);
    readFloat(body, "presence_penalty",   req.presencePenalty,   req.hasPresencePenalty);
    readFloat(body, "repetition_penalty", req.repetitionPenalty, req.hasRepetitionPenalty);

    // Reasoning toggle (vLLM-compatible). Accept a top-level `enable_thinking`
    // and the nested `chat_template_kwargs: {enable_thinking: <bool>}`.
    if (const auto v = optBool(body, "enable_thinking")) {
        req.enableThinking = *v;
    } else if (body.contains("chat_template_kwargs") &&
               body["chat_template_kwargs"].is_object()) {
        const auto& kwargs = body["chat_template_kwargs"];
        if (kwargs.contains("enable_thinking") &&
            kwargs["enable_thinking"].is_boolean()) {
            req.enableThinking = kwargs["enable_thinking"].get<bool>();
        }
    }

    // OpenAI reasoning_effort (o-series). Approximate map onto the thinking
    // toggle ONLY when the client didn't set enable_thinking explicitly: any
    // effort other than "none" opts into reasoning, "none" forces a direct
    // answer. A non-string value is malformed.
    if (!req.enableThinking.has_value()) {
        if (const auto eff = optString(body, "reasoning_effort")) {
            req.enableThinking = (*eff != "none");
        }
    }

    if (const auto v = optInt(body, "seed")) {
        req.seed = static_cast<std::uint64_t>(*v);
    }

    if (const auto v = optBool(body, "stream")) {
        req.stream = *v;
    }

    // OpenAI stream_options: {include_usage: bool}. Parsed regardless of
    // `stream` (a no-op when not streaming); honored by the streaming path as
    // a terminal usage chunk before [DONE].
    if (present(body, "stream_options")) {
        const auto& so = body["stream_options"];
        if (!so.is_object()) {
            throw ChatRequestError("stream_options must be an object",
                                   "stream_options");
        }
        if (const auto iu = optBool(so, "include_usage")) {
            req.includeUsage = *iu;
        }
    }

    if (body.contains("stop") && !body["stop"].is_null()) {
        const auto& s = body["stop"];
        if (s.is_string()) {
            req.stopStrings.push_back(s.get<std::string>());
        } else if (s.is_array()) {
            for (const auto& e : s) {
                if (e.is_string()) {
                    req.stopStrings.push_back(e.get<std::string>());
                }
            }
        } else {
            throw ChatRequestError("stop must be a string or an array of strings",
                                   "stop");
        }
    }

    // OpenAI response_format: {type:"text"|"json_object"|"json_schema"}. Shape
    // is validated (a malformed value is a 400); the requested format is
    // recorded on the request. Enforcement (grammar-constrained decoding) is
    // not yet implemented, so json_object/json_schema are accepted best-effort
    // and never faked in the response (8.19 Increment 2).
    if (present(body, "response_format")) {
        const auto& rf = body["response_format"];
        if (!rf.is_object() || !rf.contains("type") || !rf["type"].is_string()) {
            throw ChatRequestError(
                "response_format must be an object with a string `type`",
                "response_format");
        }
        const auto t = rf["type"].get<std::string>();
        if (t == "text") {
            req.responseFormat = ResponseFormat::Text;
        } else if (t == "json_object") {
            req.responseFormat = ResponseFormat::JsonObject;
        } else if (t == "json_schema") {
            req.responseFormat = ResponseFormat::JsonSchema;
        } else {
            throw ChatRequestError(
                "response_format.type must be text, json_object or json_schema",
                "response_format");
        }
    }

    return req;
}

} // namespace mimirmind::server
