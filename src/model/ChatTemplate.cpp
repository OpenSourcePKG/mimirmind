#include "model/ChatTemplate.hpp"

#include "model/Tokenizer.hpp"

#include <algorithm>
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

constexpr std::string_view kGemmaStartOfTurn = "<start_of_turn>";
constexpr std::string_view kGemmaEndOfTurn   = "<end_of_turn>";

/// Gemma 3/4 chat roles. The HF Jinja template emits "user" / "model"
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
                                     bool                         addGenerationPrompt) {
    const std::int32_t imStart = requireToken(tok, kQwenImStart);
    const std::int32_t imEnd   = requireToken(tok, kQwenImEnd);

    std::vector<std::int32_t> ids;
    ids.reserve(64);  // chat headers + a short message fit comfortably

    // If the caller did not provide an explicit system message as the
    // first turn, insert Qwen's default — exactly what the HF Jinja
    // template does.
    const bool hasExplicitSystem =
        !messages.empty() && messages.front().role == ChatRole::System;

    auto emitTurn = [&](std::string_view role, std::string_view content) {
        ids.push_back(imStart);
        std::string head{role};
        head.push_back('\n');
        head.append(content);
        encodeText(tok, head, ids);
        ids.push_back(imEnd);
        encodeText(tok, "\n", ids);
    };

    if (!hasExplicitSystem) {
        emitTurn("system", kQwenDefaultSystem);
    }

    for (const auto& m : messages) {
        emitTurn(chatRoleName(m.role), m.content);
    }

    if (addGenerationPrompt) {
        ids.push_back(imStart);
        encodeText(tok, "assistant\n", ids);
    }

    return ids;
}

/// Gemma 3 / Gemma 4 chat template (mirrors the HF Jinja template).
///
/// Format:
///   <bos>
///   <start_of_turn>user\n{user_content}<end_of_turn>\n
///   <start_of_turn>model\n{model_content}<end_of_turn>\n
///   ...
///   <start_of_turn>model\n              (when addGenerationPrompt)
///
/// System messages are prepended to the first user turn separated by a
/// blank line — Gemma was not trained with a separate system role.
std::vector<std::int32_t> encodeGemma4(const Tokenizer&             tok,
                                       std::span<const ChatMessage> messages,
                                       bool                         addGenerationPrompt) {
    const std::int32_t startOfTurn = requireToken(tok, kGemmaStartOfTurn);
    const std::int32_t endOfTurn   = requireToken(tok, kGemmaEndOfTurn);
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

} // namespace

std::string_view chatRoleName(ChatRole r) noexcept {
    switch (r) {
        case ChatRole::System:    return "system";
        case ChatRole::User:      return "user";
        case ChatRole::Assistant: return "assistant";
    }
    return "user";
}

bool parseChatRole(std::string_view s, ChatRole& out) noexcept {
    const std::string low = toLower(s);
    if (low == "system")    { out = ChatRole::System;    return true; }
    if (low == "user")      { out = ChatRole::User;      return true; }
    if (low == "assistant") { out = ChatRole::Assistant; return true; }
    return false;
}

ChatTemplate::Style
ChatTemplate::detectFromArch(std::string_view architecture) {
    const std::string arch = toLower(architecture);
    // Qwen2, Qwen2.5, Qwen3 all use ChatML.
    if (arch.rfind("qwen", 0) == 0) {
        return Style::QwenChatML;
    }
    // Gemma 3 and Gemma 4 share the <start_of_turn>/<end_of_turn> template.
    if (arch == "gemma4" || arch == "gemma3") {
        return Style::Gemma4;
    }
    throw std::runtime_error(
        "ChatTemplate: no hardcoded chat template for architecture '" +
        std::string{architecture} +
        "' yet — supported: qwen*, gemma3, gemma4");
}

std::vector<std::int32_t>
ChatTemplate::encode(Style                        style,
                     const Tokenizer&             tok,
                     std::span<const ChatMessage> messages,
                     bool                         addGenerationPrompt) {
    switch (style) {
        case Style::QwenChatML:
            return encodeQwen(tok, messages, addGenerationPrompt);
        case Style::Gemma4:
            return encodeGemma4(tok, messages, addGenerationPrompt);
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
        case Style::Gemma4: {
            const std::int32_t endOfTurn = tok.findToken(kGemmaEndOfTurn);
            if (endOfTurn >= 0) {
                ids.push_back(endOfTurn);
            }
            break;
        }
    }
    return ids;
}

} // namespace mimirmind::model