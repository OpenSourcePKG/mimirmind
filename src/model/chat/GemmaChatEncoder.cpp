// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "model/chat/GemmaChatEncoder.hpp"

#include "model/Tokenizer.hpp"
#include "model/chat/ChatEncoderCommon.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <string>

namespace mimirmind::model::chat {

namespace {

// Gemma 2/3 use the symmetric <start_of_turn> marker; Gemma 4 the asymmetric
// <|turn> marker. The end markers are shared and live in the header.
constexpr std::string_view kGemma3StartOfTurn = "<start_of_turn>";
constexpr std::string_view kGemma4StartOfTurn = "<|turn>";

using json = nlohmann::json;

/// Gemma 3/4 chat roles. The HF Jinja templates emit "user" / "model"
/// (NOT "assistant"). System messages are prepended to the first user
/// message because Gemma's training data does not natively use a
/// separate system turn.
[[nodiscard]] std::string_view gemmaRoleName(ChatRole r) noexcept {
    switch (r) {
        case ChatRole::User:      return "user";
        case ChatRole::Assistant: return "model";
        case ChatRole::System:    return "user"; // folded into first user turn
    }
    return "user";
}

/// Shared encoder for Gemma 2/3 and Gemma 4 chat templates — they only
/// differ in the literal turn-marker token strings, so the algorithm
/// is identical. The caller passes the right pair via `startOfTurnText`
/// / `endOfTurnText`.
///
/// Format (with placeholders substituted):
///   <bos>
///   {sot}user\n{user_content}{eot}\n
///   {sot}model\n{model_content}{eot}\n
///   ...
///   {sot}model\n                                (when addGenerationPrompt)
///
/// System messages are prepended to the first user turn separated by a
/// blank line — Gemma was not trained with a separate system role.
std::vector<std::int32_t> encodeGemmaImpl(const Tokenizer&             tok,
                                          std::span<const ChatMessage> messages,
                                          bool                         addGenerationPrompt,
                                          std::string_view             startOfTurnText,
                                          std::string_view             endOfTurnText) {
    const std::int32_t startOfTurn = requireToken(tok, startOfTurnText);
    const std::int32_t endOfTurn   = requireToken(tok, endOfTurnText);
    const std::int32_t bosId       = tok.bosId();

    std::vector<std::int32_t> ids;
    ids.reserve(64);

    if (bosId >= 0) {
        ids.push_back(bosId);
    }

    // Pre-process: fold a leading system message into the first user
    // message. Gemma's HF template does this implicitly via prompt
    // pre-processing; we make it explicit so encode() stays pure.
    std::vector<ChatMessage> rendered;
    rendered.reserve(messages.size());
    std::string carriedSystem;
    for (const auto& m : messages) {
        if (m.role == ChatRole::System) {
            if (!carriedSystem.empty()) {
                carriedSystem.append("\n\n");
            }
            carriedSystem.append(m.content);
            continue;
        }
        if (!carriedSystem.empty() && m.role == ChatRole::User) {
            ChatMessage merged{ChatRole::User, carriedSystem + "\n\n" + m.content};
            rendered.push_back(std::move(merged));
            carriedSystem.clear();
        } else {
            rendered.push_back(m);
        }
    }
    // Stray system left over (no user turn followed): emit as a user turn
    // anyway so the model has the instructions.
    if (!carriedSystem.empty()) {
        rendered.push_back({ChatRole::User, std::move(carriedSystem)});
    }

    for (const auto& m : rendered) {
        ids.push_back(startOfTurn);
        std::string head{gemmaRoleName(m.role)};
        head.push_back('\n');
        head.append(m.content);
        encodeText(tok, head, ids);
        ids.push_back(endOfTurn);
        encodeText(tok, "\n", ids);
    }

    if (addGenerationPrompt) {
        ids.push_back(startOfTurn);
        encodeText(tok, "model\n", ids);
    }

    return ids;
}

// ---- M-FunctionCalling Phase 2: Gemma 4 tool rendering ----------------------
//
// Gemma 4 declares tools and emits/consumes tool calls in a custom DSL (NOT
// JSON), with its own special-token markers. The markers are single special
// tokens in the vocab, so they are emitted as token ids (not BPE-encoded
// text); free text between them goes through encodeText. This is an
// approximate renderer (name/description/property-description+type/required);
// enum/nullable/nested-schema nuances are omitted, which the model tolerates.

// Gemma tool markers, longest-first so a prefix ("<|tool>") never shadows a
// longer marker ("<|tool_call>") during the scan.
constexpr std::array<std::string_view, 7> kGemmaToolMarkers = {
    "<|tool_call>", "<tool_call|>", "<|tool_response>", "<tool_response|>",
    "<|tool>", "<tool|>", "<|\"|>",
};

// Emit a DSL string that interleaves gemma special-token markers with free
// text: each marker becomes its token id (falling back to encodeText if the
// vocab lacks it); text spans between markers go through encodeText.
void emitGemmaDsl(const Tokenizer& tok, std::string_view dsl,
                  std::vector<std::int32_t>& ids) {
    std::string pending;
    auto flush = [&] {
        if (!pending.empty()) {
            encodeText(tok, pending, ids);
            pending.clear();
        }
    };
    for (std::size_t i = 0; i < dsl.size();) {
        std::string_view marker;
        for (const auto& m : kGemmaToolMarkers) {
            if (dsl.substr(i, m.size()) == m) {
                marker = m;
                break;
            }
        }
        if (!marker.empty()) {
            flush();
            const std::int32_t id = tok.findToken(marker);
            if (id >= 0) {
                ids.push_back(id);
            } else {
                encodeText(tok, marker, ids);
            }
            i += marker.size();
        } else {
            pending.push_back(dsl[i]);
            ++i;
        }
    }
    flush();
}

// A JSON value -> gemma DSL argument value.
std::string gemmaDslValue(const json& v) {
    if (v.is_string()) {
        return "<|\"|>" + v.get<std::string>() + "<|\"|>";
    }
    if (v.is_boolean()) {
        return v.get<bool>() ? "true" : "false";
    }
    if (v.is_number_integer()) {
        return std::to_string(v.get<std::int64_t>());
    }
    if (v.is_number()) {
        return std::to_string(v.get<double>());
    }
    if (v.is_array()) {
        std::string s = "[";
        bool first = true;
        for (const auto& e : v) {
            if (!first) { s += ","; }
            first = false;
            s += gemmaDslValue(e);
        }
        return s + "]";
    }
    if (v.is_object()) {
        std::string s = "{";
        bool first = true;
        for (auto it = v.begin(); it != v.end(); ++it) {
            if (!first) { s += ","; }
            first = false;
            s += it.key();
            s += ":";
            s += gemmaDslValue(it.value());
        }
        return s + "}";
    }
    return "<|\"|><|\"|>";
}

// Render one tool's declaration DSL from its OpenAI tool JSON.
std::string gemmaRenderToolDecl(const ToolSpec& t) {
    const json tool = json::parse(t.toolJson, nullptr, /*allow_exceptions=*/false);
    if (tool.is_discarded() || !tool.contains("function") ||
        !tool["function"].is_object()) {
        return "<|tool>declaration:" + t.name + "{}<tool|>";
    }
    const json& fn = tool["function"];
    const std::string name = fn.value("name", t.name);
    std::string s = "<|tool>declaration:" + name + "{";
    bool comma = false;
    if (fn.contains("description") && fn["description"].is_string()) {
        s += "description:<|\"|>" + fn["description"].get<std::string>() + "<|\"|>";
        comma = true;
    }
    if (fn.contains("parameters") && fn["parameters"].is_object()) {
        const json& params = fn["parameters"];
        if (comma) { s += ","; }
        s += "parameters:{properties:{";
        if (params.contains("properties") && params["properties"].is_object()) {
            bool pfirst = true;
            for (auto it = params["properties"].begin();
                 it != params["properties"].end(); ++it) {
                if (!pfirst) { s += ","; }
                pfirst = false;
                s += it.key() + ":{";
                bool pc = false;
                if (it.value().contains("description") &&
                    it.value()["description"].is_string()) {
                    s += "description:<|\"|>" +
                         it.value()["description"].get<std::string>() + "<|\"|>";
                    pc = true;
                }
                if (it.value().contains("type") && it.value()["type"].is_string()) {
                    std::string ty = it.value()["type"].get<std::string>();
                    std::transform(ty.begin(), ty.end(), ty.begin(),
                                   [](unsigned char c) { return std::toupper(c); });
                    if (pc) { s += ","; }
                    s += "type:<|\"|>" + ty + "<|\"|>";
                }
                s += "}";
            }
        }
        s += "}";
        if (params.contains("required") && params["required"].is_array()) {
            s += ",required:[";
            bool rfirst = true;
            for (const auto& r : params["required"]) {
                if (!r.is_string()) { continue; }
                if (!rfirst) { s += ","; }
                rfirst = false;
                s += "<|\"|>" + r.get<std::string>() + "<|\"|>";
            }
            s += "]";
        }
        s += "}";
    }
    s += "}<tool|>";
    return s;
}

// Render an assistant tool call's arguments (JSON string) as a gemma DSL body.
std::string gemmaRenderCallArgs(const std::string& argsJson) {
    const json args = json::parse(argsJson, nullptr, /*allow_exceptions=*/false);
    if (args.is_discarded() || !args.is_object()) {
        return "{}";
    }
    return gemmaDslValue(args);
}

// Dedicated Gemma 4 encoder for the tool path: tools live in their own
// <|turn>system turn (the shared gemma encoder folds system into the user
// turn, which the tool format does not want).
std::vector<std::int32_t> encodeGemma4Tools(const Tokenizer&             tok,
                                            std::span<const ChatMessage> messages,
                                            bool                         addGenerationPrompt,
                                            std::span<const ToolSpec>    tools) {
    const std::int32_t startOfTurn = requireToken(tok, kGemma4StartOfTurn);
    const std::int32_t endOfTurn   = requireToken(tok, kGemma4EndOfTurn);
    const std::int32_t bosId       = tok.bosId();

    std::vector<std::int32_t> ids;
    ids.reserve(128);
    if (bosId >= 0) {
        ids.push_back(bosId);
    }

    std::string systemContent;
    for (const auto& m : messages) {
        if (m.role == ChatRole::System) {
            if (!systemContent.empty()) {
                systemContent.append("\n\n");
            }
            systemContent.append(m.content);
        }
    }

    // System turn carrying the tool declarations.
    ids.push_back(startOfTurn);
    encodeText(tok, "system\n", ids);
    if (!systemContent.empty()) {
        encodeText(tok, systemContent, ids);
    }
    for (const auto& t : tools) {
        emitGemmaDsl(tok, gemmaRenderToolDecl(t), ids);
    }
    ids.push_back(endOfTurn);
    encodeText(tok, "\n", ids);

    for (const auto& m : messages) {
        if (m.role == ChatRole::System) {
            continue;
        }
        if (m.role == ChatRole::Tool) {
            ids.push_back(startOfTurn);
            encodeText(tok, "user\n", ids);
            emitGemmaDsl(tok,
                         "<|tool_response>response:tool{value:<|\"|>" + m.content +
                             "<|\"|>}<tool_response|>",
                         ids);
            ids.push_back(endOfTurn);
            encodeText(tok, "\n", ids);
            continue;
        }
        ids.push_back(startOfTurn);
        std::string head{gemmaRoleName(m.role)};
        head.push_back('\n');
        encodeText(tok, head, ids);
        if (!m.content.empty()) {
            encodeText(tok, m.content, ids);
        }
        if (m.role == ChatRole::Assistant && !m.toolCalls.empty()) {
            for (const auto& call : m.toolCalls) {
                emitGemmaDsl(tok,
                             "<|tool_call>call:" + call.name +
                                 gemmaRenderCallArgs(call.argumentsJson) +
                                 "<tool_call|>",
                             ids);
            }
        }
        ids.push_back(endOfTurn);
        encodeText(tok, "\n", ids);
    }

    if (addGenerationPrompt) {
        ids.push_back(startOfTurn);
        encodeText(tok, "model\n", ids);
    }
    return ids;
}

} // namespace

std::vector<std::int32_t> GemmaChatEncoder::encodeGemma3(
        const Tokenizer&              tok,
        std::span<const ChatMessage>  messages,
        bool                          addGenerationPrompt) {
    return encodeGemmaImpl(tok, messages, addGenerationPrompt,
                           kGemma3StartOfTurn, kGemma3EndOfTurn);
}

std::vector<std::int32_t> GemmaChatEncoder::encodeGemma4(
        const Tokenizer&              tok,
        std::span<const ChatMessage>  messages,
        bool                          addGenerationPrompt,
        std::span<const ToolSpec>     tools) {
    if (!tools.empty()) {
        return encodeGemma4Tools(tok, messages, addGenerationPrompt, tools);
    }
    return encodeGemmaImpl(tok, messages, addGenerationPrompt,
                           kGemma4StartOfTurn, kGemma4EndOfTurn);
}

} // namespace mimirmind::model::chat
