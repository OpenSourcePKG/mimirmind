// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "model/ChatTemplate.hpp"

#include "model/Tokenizer.hpp"
#include "model/chat/ChatEncoderCommon.hpp"
#include "model/chat/GemmaChatEncoder.hpp"
#include "model/chat/Llama3ChatEncoder.hpp"
#include "model/chat/QwenChatEncoder.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <stdexcept>
#include <string>

namespace mimirmind::model {

namespace {



/// Markup that Gemma 4 emits at the start of every model response,
/// wrapping an (often empty) thinking-channel block.
constexpr std::string_view kGemma4ChannelStart = "<|channel>";
constexpr std::string_view kGemma4ChannelEnd   = "<channel|>";


std::string toLower(std::string_view in) {
    std::string out;
    out.reserve(in.size());
    for (char c : in) {
        out.push_back(static_cast<char>(
            std::tolower(static_cast<unsigned char>(c))));
    }
    return out;
}

// encodeText moved to model::chat::ChatEncoderCommon (roadmap 8.30.11.3); the
// per-family encoders now live under src/model/chat/. Pull it into this TU's
// unqualified scope for the tool-call-opener id helper below.
using chat::encodeText;



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

ChatTemplate::ToolFormat
ChatTemplate::toolFormatFromArch(std::string_view architecture) noexcept {
    const std::string arch = toLower(architecture);
    // Qwen3.5 / Qwen3.6 / Qwen3.8 (NVFP4 wire ids "qwen35moe" incl. the dense
    // qwen3_5 tower, "qwen4_exp"; GGUF "qwen3next") ship the Qwen3-Coder XML
    // tool format in their chat template. Qwen2/2.5/Qwen3 are Hermes-JSON.
    if (arch.rfind("qwen35", 0) == 0 || arch.rfind("qwen4", 0) == 0 ||
        arch.rfind("qwen3next", 0) == 0) {
        return ToolFormat::QwenXml;
    }
    return ToolFormat::HermesJson;
}

std::vector<std::int32_t>
ChatTemplate::encode(Style                        style,
                     const Tokenizer&             tok,
                     std::span<const ChatMessage> messages,
                     bool                         addGenerationPrompt,
                     std::span<const ToolSpec>    tools,
                     std::optional<bool>          enableThinking,
                     ToolFormat                   toolFormat,
                     std::optional<bool>          templateUsesThink,
                     std::optional<bool>          toolDefsStructuredXml) {
    switch (style) {
        case Style::QwenChatML:
            return chat::QwenChatEncoder::encode(
                tok, messages, addGenerationPrompt, tools,
                enableThinking, toolFormat, templateUsesThink,
                toolDefsStructuredXml);
        case Style::Gemma3:
            // Gemma 3 tool rendering not implemented (Gemma 4 is the target).
            return chat::GemmaChatEncoder::encodeGemma3(
                tok, messages, addGenerationPrompt);
        case Style::Gemma4:
            return chat::GemmaChatEncoder::encodeGemma4(
                tok, messages, addGenerationPrompt, tools);
        case Style::Llama3:
            // Tool rendering not implemented for Llama-3 yet (tools ignored).
            return chat::Llama3ChatEncoder::encode(tok, messages,
                                                   addGenerationPrompt);
    }
    throw std::runtime_error("ChatTemplate::encode: unhandled style");
}

std::vector<std::int32_t>
ChatTemplate::stopIds(Style style, const Tokenizer& tok) {
    std::vector<std::int32_t> ids;
    switch (style) {
        case Style::QwenChatML: {
            const std::int32_t imEnd = tok.findToken(chat::kQwenImEnd);
            if (imEnd >= 0) {
                ids.push_back(imEnd);
            }
            break;
        }
        case Style::Gemma3: {
            const std::int32_t endOfTurn = tok.findToken(chat::kGemma3EndOfTurn);
            if (endOfTurn >= 0) {
                ids.push_back(endOfTurn);
            }
            break;
        }
        case Style::Gemma4: {
            const std::int32_t endOfTurn = tok.findToken(chat::kGemma4EndOfTurn);
            if (endOfTurn >= 0) {
                ids.push_back(endOfTurn);
            }
            break;
        }
        case Style::Llama3: {
            // <|eot_id|> ends an assistant turn; the true EOS <|end_of_text|>
            // is handled by the tokenizer's EOS.
            const std::int32_t eot = tok.findToken(chat::kLlama3Eot);
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

std::string_view
ChatTemplate::toolCallSalvageOpenerText(Style style) noexcept {
    // The FULLER opener the tool-salvage re-decode prefills to force a drifted
    // model back onto the canonical call syntax — for Qwen ChatML this includes
    // the `<function=` TagDispatch trigger (which toolCallOpenerText deliberately
    // omits, since the model normally emits it itself). Per-style here, not
    // hardcoded in the server. Empty => no salvage opener for this dialect.
    switch (style) {
        case Style::QwenChatML: return "<tool_call>\n<function=";
        case Style::Gemma4:     return {};
        case Style::Gemma3:     return {};
        case Style::Llama3:     return {};
    }
    return {};
}

std::span<const std::string_view>
ChatTemplate::toolIntentMarkers(Style style) noexcept {
    // Literal substrings that signal the model TRIED to emit a tool call in this
    // dialect (used by the salvage re-decode trigger). Kept per-style here rather
    // than as a literal list in the server. QwenChatML keeps Gemma's ```tool_code
    // fence too as a belt-and-suspenders cross-dialect leak marker.
    static constexpr std::string_view kQwen[] = {
        "```tool_code", "<function", "<tool_call"};
    switch (style) {
        case Style::QwenChatML: return {kQwen, std::size(kQwen)};
        case Style::Gemma4:     return {kQwen, std::size(kQwen)};
        case Style::Gemma3:     return {};
        case Style::Llama3:     return {};
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
                            std::string* reasoningOut, bool thinkPreOpened) {
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
                return out;
            }
            // No closer. When <think> was PRE-OPENED (enable_thinking:true) the
            // whole span is an unclosed reasoning block — the model ran out
            // (max_tokens) or reasoned markerless without ever emitting </think>.
            // Route it ALL to reasoning with empty content, matching the streaming
            // ResponseCleaner (which starts InThink and stays there to EOS) and
            // vLLM's corrected parser (missing end token under thinking-on =
            // everything is reasoning). Without a pre-open this is a plain
            // non-thinking answer (Qwen2/2.5 or enable_thinking:false) — leave it
            // as content.
            if (thinkPreOpened) {
                if (reasoningOut != nullptr) {
                    *reasoningOut = std::move(out);
                }
                return std::string{};
            }
            return out;
        }
        case Style::Gemma3:
            stripTrailing(out, chat::kGemma3EndOfTurn);
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
            stripTrailing(out, chat::kGemma4EndOfTurn);
            return out;
        }
        case Style::Llama3:
            // <|eot_id|> is normally consumed by stopIds, but strip a trailing
            // one defensively (mirrors Gemma). No thinking-channel to unwrap.
            stripTrailing(out, chat::kLlama3Eot);
            return out;
    }
    return out;
}

} // namespace mimirmind::model