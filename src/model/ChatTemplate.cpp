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
    throw std::runtime_error(
        "ChatTemplate: no hardcoded chat template for architecture '" +
        std::string{architecture} +
        "' yet — M7c only supports Qwen ChatML");
}

std::vector<std::int32_t>
ChatTemplate::encode(Style                        style,
                     const Tokenizer&             tok,
                     std::span<const ChatMessage> messages,
                     bool                         addGenerationPrompt) {
    switch (style) {
        case Style::QwenChatML:
            return encodeQwen(tok, messages, addGenerationPrompt);
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
    }
    return ids;
}

} // namespace mimirmind::model