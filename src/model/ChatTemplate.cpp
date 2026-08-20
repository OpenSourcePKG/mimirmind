// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "model/ChatTemplate.hpp"

#include "model/Tokenizer.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <stdexcept>
#include <string>

namespace mimirmind::model {

namespace {

constexpr std::string_view kQwenImStart = "<|im_start|>";
constexpr std::string_view kQwenImEnd   = "<|im_end|>";

/// Mirrors the official Qwen2.5 default when the conversation has no
/// explicit system message. Kept identical to llama.cpp / HF Jinja
/// template so encoded bytes match.
constexpr std::string_view kQwenDefaultSystem =
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";

// Gemma 2/3 use the symmetric <start_of_turn>/<end_of_turn> tokens;
// Gemma 4 dropped these in favour of the asymmetric <|turn>/<turn|>
// pair, plus a thinking-channel wrapper. Keep both — the chat-template
// dispatch picks the right pair per Style.
constexpr std::string_view kGemma3StartOfTurn = "<start_of_turn>";
constexpr std::string_view kGemma3EndOfTurn   = "<end_of_turn>";

constexpr std::string_view kGemma4StartOfTurn = "<|turn>";
constexpr std::string_view kGemma4EndOfTurn   = "<turn|>";

/// Markup that Gemma 4 emits at the start of every model response,
/// wrapping an (often empty) thinking-channel block.
constexpr std::string_view kGemma4ChannelStart = "<|channel>";
constexpr std::string_view kGemma4ChannelEnd   = "<channel|>";

// Llama-3.x special tokens (Llama-3.0/3.1/3.2). The GGUF architecture is
// "llama" for all of them; this style additionally assumes the Llama-3 header
// tokens are present in the vocab (requireToken throws otherwise), which also
// safely rejects a Llama-2 "llama" GGUF that uses the [INST] template instead.
constexpr std::string_view kLlama3BeginOfText = "<|begin_of_text|>";
constexpr std::string_view kLlama3StartHeader = "<|start_header_id|>";
constexpr std::string_view kLlama3EndHeader   = "<|end_header_id|>";
constexpr std::string_view kLlama3Eot         = "<|eot_id|>";

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

std::string toLower(std::string_view in) {
    std::string out;
    out.reserve(in.size());
    for (char c : in) {
        out.push_back(static_cast<char>(
            std::tolower(static_cast<unsigned char>(c))));
    }
    return out;
}

std::int32_t requireToken(const Tokenizer& tok, std::string_view text) {
    const std::int32_t id = tok.findToken(text);
    if (id < 0) {
        throw std::runtime_error(
            "ChatTemplate: tokenizer is missing required special token '" +
            std::string{text} + "'");
    }
    return id;
}

void encodeText(const Tokenizer&         tok,
                std::string_view         text,
                std::vector<std::int32_t>& out) {
    if (text.empty()) {
        return;
    }
    const auto ids = tok.encode(text, /*addBos=*/false);
    out.insert(out.end(), ids.begin(), ids.end());
}

std::vector<std::int32_t> encodeQwen(const Tokenizer&             tok,
                                     std::span<const ChatMessage> messages,
                                     bool                         addGenerationPrompt,
                                     std::span<const ToolSpec>    tools,
                                     std::optional<bool>          enableThinking) {
    const std::int32_t imStart = requireToken(tok, kQwenImStart);
    const std::int32_t imEnd   = requireToken(tok, kQwenImEnd);

    std::vector<std::int32_t> ids;
    ids.reserve(64);  // chat headers + a short message fit comfortably

    const bool hasExplicitSystem =
        !messages.empty() && messages.front().role == ChatRole::System;

    // Qwen2 / Qwen2.5 inject a default system turn ("You are Qwen, created
    // by Alibaba Cloud. …") when the caller provides none — their HF Jinja
    // template does exactly this. Qwen3 / Qwen3.5 / Qwen3.6 (the "thinking"
    // models, `qwen35moe`) do NOT: their template renders no system turn at
    // all unless one is supplied. Prepending the 2.5 default to a 3.x model
    // is an out-of-distribution context it was never trained on and yields
    // off-topic / early-EOS garbage. Discriminate on the <think> special
    // token, exactly as the generation-prompt branch below already does.
    const bool isThinkingFamily = tok.findToken("<think>") >= 0;

    // M-FunctionCalling: Hermes-style tool-spec block Qwen2.5 is trained on.
    // Rendered into the system turn when the request carries tools; the model
    // then answers with <tool_call>{...}</tool_call>, which the handler parses
    // back into structured tool_calls.
    std::string toolsBlock;
    if (!tools.empty()) {
        toolsBlock =
            "\n\n# Tools\n\nYou may call one or more functions to assist with "
            "the user query.\n\nYou are provided with function signatures "
            "within <tools></tools> XML tags:\n<tools>";
        for (const auto& t : tools) {
            toolsBlock += "\n";
            toolsBlock += t.toolJson;
        }
        toolsBlock +=
            "\n</tools>\n\nFor each function call, return a json object with "
            "function name and arguments within <tool_call></tool_call> XML "
            "tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": "
            "<args-json-object>}\n</tool_call>";
    }

    auto emitTurn = [&](std::string_view role, std::string_view content) {
        ids.push_back(imStart);
        std::string head{role};
        head.push_back('\n');
        head.append(content);
        encodeText(tok, head, ids);
        ids.push_back(imEnd);
        encodeText(tok, "\n", ids);
    };

    // System turn. Without an explicit system: Qwen2.5 gets its default, a
    // thinking model gets none — but if tools are present they still need a
    // home, so emit a (possibly bare) system turn carrying the tools block.
    if (!hasExplicitSystem) {
        std::string sys =
            isThinkingFamily ? std::string{} : std::string{kQwenDefaultSystem};
        sys += toolsBlock;
        if (!sys.empty()) {
            emitTurn("system", sys);
        }
    }

    // When the caller DID supply a system turn, the tools block is appended
    // to it on its first occurrence.
    bool explicitSystemToolsPending = hasExplicitSystem && !toolsBlock.empty();

    for (const auto& m : messages) {
        if (explicitSystemToolsPending && m.role == ChatRole::System) {
            emitTurn("system", m.content + toolsBlock);
            explicitSystemToolsPending = false;
            continue;
        }
        // Assistant turn that invoked tools: render each call as a
        // <tool_call>{...}</tool_call> block after any visible content.
        if (m.role == ChatRole::Assistant && !m.toolCalls.empty()) {
            std::string body = m.content;
            for (const auto& call : m.toolCalls) {
                if (!body.empty()) {
                    body += "\n";
                }
                body += "<tool_call>\n{\"name\": \"";
                body += call.name;
                body += "\", \"arguments\": ";
                body += call.argumentsJson;
                body += "}\n</tool_call>";
            }
            emitTurn("assistant", body);
            continue;
        }
        // Tool result: Qwen2.5 carries it in a user turn wrapped in
        // <tool_response></tool_response>.
        if (m.role == ChatRole::Tool) {
            emitTurn("user",
                     "<tool_response>\n" + m.content + "\n</tool_response>");
            continue;
        }
        emitTurn(chatRoleName(m.role), m.content);
    }

    if (addGenerationPrompt) {
        ids.push_back(imStart);
        encodeText(tok, "assistant\n", ids);
        // Qwen3 "thinking" models (Qwen3 / Qwen3.5 / Qwen3.6 `qwen35moe`)
        // pre-open a <think> block in the default generation prompt — the
        // model is trained to continue *inside* it and close with </think>.
        // Their HF/GGUF chat_template appends '<think>\n' after
        // 'assistant\n' (default) or '<think>\n\n</think>\n\n' when thinking
        // is disabled. Qwen2 / Qwen2.5 have no <think> token and use a plain
        // ChatML generation prompt. Auto-detect by the presence of the
        // <think> special token so both families work without an arch
        // switch. Without this, a thinking model emits a spurious </think>
        // and stops immediately.
        const std::int32_t think = tok.findToken("<think>");
        if (think >= 0) {
            ids.push_back(think);
            // Explicit enable_thinking (OpenAI chat_template_kwargs, like vLLM)
            // overrides the tools heuristic: enable_thinking=false forces the
            // empty pre-closed block even WITHOUT tools (direct answer, no
            // reasoning). This is the fix for agentic RAG whose final answer
            // call carries no tools but must not leak an unclosed <think>.
            // Unset (nullopt) keeps the tools-based default.
            const bool thinkOn = enableThinking.value_or(tools.empty());
            if (thinkOn) {
                // pre-OPEN <think>: the model reasons, then closes
                // with </think>. This is the answer path; the reasoning is
                // surfaced as reasoning_content.
                encodeText(tok, "\n", ids);
            } else {
                // Tool round → pre-CLOSE an empty think block (Qwen3's
                // "thinking disabled" prompt shape: <think>\n\n</think>\n\n)
                // so the model emits the tool call directly instead of a
                // multi-thousand-token chain-of-thought. Tool selection does
                // not need deep reasoning, and those think blocks made agentic
                // RAG unusably slow (~4096 tokens / round). The final answer
                // (sent without tools) still reasons.
                encodeText(tok, "\n\n", ids);
                const std::int32_t thinkEnd = tok.findToken("</think>");
                if (thinkEnd >= 0) {
                    ids.push_back(thinkEnd);
                } else {
                    encodeText(tok, "</think>", ids);
                }
                encodeText(tok, "\n\n", ids);
            }
        }
    }

    return ids;
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

std::vector<std::int32_t> encodeGemma3(const Tokenizer&             tok,
                                       std::span<const ChatMessage> messages,
                                       bool                         addGenerationPrompt) {
    return encodeGemmaImpl(tok, messages, addGenerationPrompt,
                           kGemma3StartOfTurn, kGemma3EndOfTurn);
}

// ---- M-FunctionCalling Phase 2: Gemma 4 tool rendering ----------------------
//
// Gemma 4 declares tools and emits/consumes tool calls in a custom DSL (NOT
// JSON), with its own special-token markers. The markers are single special
// tokens in the vocab, so they are emitted as token ids (not BPE-encoded
// text); free text between them goes through encodeText. This is an
// approximate renderer (name/description/property-description+type/required);
// enum/nullable/nested-schema nuances are omitted, which the model tolerates.

using json = nlohmann::json;

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

std::vector<std::int32_t> encodeGemma4(const Tokenizer&             tok,
                                       std::span<const ChatMessage> messages,
                                       bool                         addGenerationPrompt,
                                       std::span<const ToolSpec>    tools) {
    if (!tools.empty()) {
        return encodeGemma4Tools(tok, messages, addGenerationPrompt, tools);
    }
    return encodeGemmaImpl(tok, messages, addGenerationPrompt,
                           kGemma4StartOfTurn, kGemma4EndOfTurn);
}

/// Llama-3.x chat template. Matches the reference (llama.cpp built-in "llama3"
/// / HF Jinja) byte-for-byte for the system/user/assistant path:
///
///   <|begin_of_text|>
///   <|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>   (per turn)
///   <|start_header_id|>assistant<|end_header_id|>\n\n                    (gen prompt)
///
/// No default system turn is injected (unlike Qwen2.5) — Llama-3 renders one
/// only when supplied. Special tokens go in as their ids; role names and
/// content are plain BPE. Tool calling is not wired for this style yet.
std::vector<std::int32_t> encodeLlama3(const Tokenizer&             tok,
                                       std::span<const ChatMessage> messages,
                                       bool                         addGenerationPrompt) {
    const std::int32_t bos         = requireToken(tok, kLlama3BeginOfText);
    const std::int32_t startHeader = requireToken(tok, kLlama3StartHeader);
    const std::int32_t endHeader   = requireToken(tok, kLlama3EndHeader);
    const std::int32_t eot         = requireToken(tok, kLlama3Eot);

    std::vector<std::int32_t> ids;
    ids.reserve(64);
    ids.push_back(bos);

    auto emitTurn = [&](std::string_view role, std::string_view content) {
        ids.push_back(startHeader);
        encodeText(tok, role, ids);
        ids.push_back(endHeader);
        std::string body{"\n\n"};
        body.append(content);
        encodeText(tok, body, ids);
        ids.push_back(eot);
    };

    for (const auto& m : messages) {
        // Llama-3 uses system/user/assistant (tool turns = "ipython", not used
        // here since tool calling is not wired for this style).
        emitTurn(chatRoleName(m.role), m.content);
    }

    if (addGenerationPrompt) {
        ids.push_back(startHeader);
        encodeText(tok, "assistant", ids);
        ids.push_back(endHeader);
        encodeText(tok, "\n\n", ids);
    }
    return ids;
}

} // namespace

std::string_view chatRoleName(ChatRole r) noexcept {
    switch (r) {
        case ChatRole::System:    return "system";
        case ChatRole::User:      return "user";
        case ChatRole::Assistant: return "assistant";
        case ChatRole::Tool:      return "tool";
    }
    return "user";
}

bool parseChatRole(std::string_view s, ChatRole& out) noexcept {
    const std::string low = toLower(s);
    if (low == "system")    { out = ChatRole::System;    return true; }
    if (low == "user")      { out = ChatRole::User;      return true; }
    if (low == "assistant") { out = ChatRole::Assistant; return true; }
    if (low == "tool")      { out = ChatRole::Tool;      return true; }
    return false;
}

ChatTemplate::Style
ChatTemplate::detectFromArch(std::string_view architecture) {
    const std::string arch = toLower(architecture);
    // Qwen2, Qwen2.5, Qwen3 all use ChatML.
    if (arch.rfind("qwen", 0) == 0) {
        return Style::QwenChatML;
    }
    // Gemma 2 and Gemma 3 share <start_of_turn>/<end_of_turn> — they
    // also share the role/role naming. Gemma 2 GGUFs report arch
    // "gemma2"; Gemma 3 reports "gemma3".
    if (arch == "gemma2" || arch == "gemma3") {
        return Style::Gemma3;
    }
    // Gemma 4 dropped the symmetric tokens for <|turn>/<turn|>.
    if (arch == "gemma4") {
        return Style::Gemma4;
    }
    // Llama-3.x GGUFs report architecture "llama" (Orpheus TTS backbone, plain
    // Llama-3.2 chat). The header-token check in encodeLlama3 rejects a Llama-2
    // "llama" GGUF, whose [INST] template we do not implement.
    if (arch == "llama") {
        return Style::Llama3;
    }
    throw std::runtime_error(
        "ChatTemplate: no hardcoded chat template for architecture '" +
        std::string{architecture} +
        "' yet — supported: qwen*, gemma2, gemma3, gemma4, llama");
}

std::vector<std::int32_t>
ChatTemplate::encode(Style                        style,
                     const Tokenizer&             tok,
                     std::span<const ChatMessage> messages,
                     bool                         addGenerationPrompt,
                     std::span<const ToolSpec>    tools,
                     std::optional<bool>          enableThinking) {
    switch (style) {
        case Style::QwenChatML:
            return encodeQwen(tok, messages, addGenerationPrompt, tools,
                              enableThinking);
        case Style::Gemma3:
            // Gemma 3 tool rendering not implemented (Gemma 4 is the target).
            return encodeGemma3(tok, messages, addGenerationPrompt);
        case Style::Gemma4:
            return encodeGemma4(tok, messages, addGenerationPrompt, tools);
        case Style::Llama3:
            // Tool rendering not implemented for Llama-3 yet (tools ignored).
            return encodeLlama3(tok, messages, addGenerationPrompt);
    }
    throw std::runtime_error("ChatTemplate::encode: unhandled style");
}

std::vector<std::int32_t>
ChatTemplate::stopIds(Style style, const Tokenizer& tok) {
    std::vector<std::int32_t> ids;
    switch (style) {
        case Style::QwenChatML: {
            const std::int32_t imEnd = tok.findToken(kQwenImEnd);
            if (imEnd >= 0) {
                ids.push_back(imEnd);
            }
            break;
        }
        case Style::Gemma3: {
            const std::int32_t endOfTurn = tok.findToken(kGemma3EndOfTurn);
            if (endOfTurn >= 0) {
                ids.push_back(endOfTurn);
            }
            break;
        }
        case Style::Gemma4: {
            const std::int32_t endOfTurn = tok.findToken(kGemma4EndOfTurn);
            if (endOfTurn >= 0) {
                ids.push_back(endOfTurn);
            }
            break;
        }
        case Style::Llama3: {
            // <|eot_id|> ends an assistant turn; the true EOS <|end_of_text|>
            // is handled by the tokenizer's EOS.
            const std::int32_t eot = tok.findToken(kLlama3Eot);
            if (eot >= 0) {
                ids.push_back(eot);
            }
            break;
        }
    }
    return ids;
}

std::vector<std::int32_t>
ChatTemplate::toolCallOpenerIds(Style style, const Tokenizer& tok) {
    std::vector<std::int32_t> ids;
    switch (style) {
        case Style::QwenChatML:
            // Hermes opener is plain text — same as the template renders it.
            // Prefilling it forces weak Qwen models (e.g. 1.5B) that would
            // otherwise narrate a refusal into an actual <tool_call> block.
            encodeText(tok, "<tool_call>\n", ids);
            break;
        case Style::Gemma4:
            // No opener: Gemma 4 opens every response with an auto-emitted
            // thinking-channel wrapper (`<|channel>thought\n<channel|>`, present
            // even when thinking is off), so a prefilled `<|tool_call>` opener
            // would be orphaned ahead of the wrapper and the model then emits
            // its own complete call — corrupting the parse. Gemma 4 (26B) calls
            // tools reliably unforced, so "required" relies on natural emission
            // + the existing parse. Left empty deliberately.
            break;
        case Style::Gemma3:
            break;  // no tool-calling support
        case Style::Llama3:
            break;  // tool-calling not wired for Llama-3 yet
    }
    return ids;
}

std::string_view
ChatTemplate::toolCallOpenerText(Style style) noexcept {
    switch (style) {
        case Style::QwenChatML: return "<tool_call>\n";
        case Style::Gemma4:     return {};  // see toolCallOpenerIds — no prefill
        case Style::Gemma3:     return {};
        case Style::Llama3:     return {};  // no tool support yet
    }
    return {};
}

std::vector<std::int32_t>
ChatTemplate::toolCallStopIds(Style style, const Tokenizer& tok) {
    std::vector<std::int32_t> ids;
    if (style == Style::Gemma4) {
        // `<tool_call|>` closes a Gemma 4 tool call and is a single special
        // token — halt right after the first complete call to stop the loop.
        const std::int32_t close = tok.findToken("<tool_call|>");
        if (close >= 0) {
            ids.push_back(close);
        }
    }
    // QwenChatML / Gemma3: none — see the header.
    return ids;
}

namespace {

/// Drop one occurrence of `needle` from the start of `s`, then drop any
/// leading whitespace that follows. Returns true if removed.
bool stripLeading(std::string& s, std::string_view needle) {
    if (s.size() < needle.size() ||
        std::string_view(s).substr(0, needle.size()) != needle) {
        return false;
    }
    s.erase(0, needle.size());
    while (!s.empty() && (s.front() == '\n' || s.front() == ' ' || s.front() == '\t')) {
        s.erase(0, 1);
    }
    return true;
}

/// Drop one trailing occurrence of `needle`, plus any trailing whitespace.
bool stripTrailing(std::string& s, std::string_view needle) {
    while (!s.empty() && (s.back() == '\n' || s.back() == ' ' || s.back() == '\t')) {
        s.pop_back();
    }
    if (s.size() < needle.size()) {
        return false;
    }
    const auto offset = s.size() - needle.size();
    if (std::string_view(s).substr(offset, needle.size()) != needle) {
        return false;
    }
    s.erase(offset);
    while (!s.empty() && (s.back() == '\n' || s.back() == ' ' || s.back() == '\t')) {
        s.pop_back();
    }
    return true;
}

} // namespace

std::string
ChatTemplate::cleanResponse(Style style, std::string_view text,
                            std::string* reasoningOut) {
    std::string out{text};
    switch (style) {
        case Style::QwenChatML: {
            // Qwen3 "thinking" models (qwen35moe) have <think> pre-opened in
            // the generation prompt, so the response starts INSIDE the thinking
            // block and closes it with a lone </think>. The text up to that
            // closer is reasoning; drop it (or hand it back via reasoningOut),
            // then any leading whitespace. Qwen2/2.5 never emit </think>, so
            // this is a no-op there. <|im_end|> is already removed upstream via
            // stopIds.
            constexpr std::string_view kThinkEnd{"</think>"};
            const auto end = out.find(kThinkEnd);
            if (end != std::string::npos) {
                if (reasoningOut != nullptr) {
                    *reasoningOut = out.substr(0, end);
                }
                out.erase(0, end + kThinkEnd.size());
                while (!out.empty() && (out.front() == '\n' || out.front() == ' ' ||
                                        out.front() == '\t' || out.front() == '\r')) {
                    out.erase(0, 1);
                }
            }
            return out;
        }
        case Style::Gemma3:
            stripTrailing(out, kGemma3EndOfTurn);
            return out;
        case Style::Gemma4: {
            // Gemma 4 emits <|channel>thought\n<channel|> at the very
            // start of every response, even when thinking mode is off
            // (then the channel is empty). Drop the whole wrapper.
            if (stripLeading(out, kGemma4ChannelStart)) {
                // Now skip the channel-tag name + newline up to <channel|>.
                // The body between the tag name and the closer is the thinking.
                const auto end = out.find(kGemma4ChannelEnd);
                if (end != std::string::npos) {
                    if (reasoningOut != nullptr) {
                        *reasoningOut = out.substr(0, end);
                    }
                    out.erase(0, end + kGemma4ChannelEnd.size());
                }
                while (!out.empty() && (out.front() == '\n' ||
                                        out.front() == ' '  ||
                                        out.front() == '\t')) {
                    out.erase(0, 1);
                }
            }
            stripTrailing(out, kGemma4EndOfTurn);
            return out;
        }
        case Style::Llama3:
            // <|eot_id|> is normally consumed by stopIds, but strip a trailing
            // one defensively (mirrors Gemma). No thinking-channel to unwrap.
            stripTrailing(out, kLlama3Eot);
            return out;
    }
    return out;
}

} // namespace mimirmind::model