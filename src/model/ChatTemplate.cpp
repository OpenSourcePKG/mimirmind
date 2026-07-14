// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

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

std::vector<std::int32_t> encodeGemma4(const Tokenizer&             tok,
                                       std::span<const ChatMessage> messages,
                                       bool                         addGenerationPrompt) {
    return encodeGemmaImpl(tok, messages, addGenerationPrompt,
                           kGemma4StartOfTurn, kGemma4EndOfTurn);
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
    throw std::runtime_error(
        "ChatTemplate: no hardcoded chat template for architecture '" +
        std::string{architecture} +
        "' yet — supported: qwen*, gemma2, gemma3, gemma4");
}

std::vector<std::int32_t>
ChatTemplate::encode(Style                        style,
                     const Tokenizer&             tok,
                     std::span<const ChatMessage> messages,
                     bool                         addGenerationPrompt) {
    switch (style) {
        case Style::QwenChatML:
            return encodeQwen(tok, messages, addGenerationPrompt);
        case Style::Gemma3:
            return encodeGemma3(tok, messages, addGenerationPrompt);
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
    }
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
ChatTemplate::cleanResponse(Style style, std::string_view text) {
    std::string out{text};
    switch (style) {
        case Style::QwenChatML:
            // Decoder already drops <|im_end|> via stopIds; nothing else
            // structural sits in the visible content.
            return out;
        case Style::Gemma3:
            stripTrailing(out, kGemma3EndOfTurn);
            return out;
        case Style::Gemma4: {
            // Gemma 4 emits <|channel>thought\n<channel|> at the very
            // start of every response, even when thinking mode is off
            // (then the channel is empty). Drop the whole wrapper.
            if (stripLeading(out, kGemma4ChannelStart)) {
                // Now skip the channel-tag name + newline up to <channel|>.
                const auto end = out.find(kGemma4ChannelEnd);
                if (end != std::string::npos) {
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
    }
    return out;
}

} // namespace mimirmind::model