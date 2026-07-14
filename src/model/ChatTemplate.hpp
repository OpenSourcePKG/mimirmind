// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace mimirmind::model {

class Tokenizer;

/// Author of a chat message. Mirrors the OpenAI Chat Completions roles
/// — anything else (tool, function) is not yet supported by M7c.
enum class ChatRole {
    System,
    User,
    Assistant,
};

[[nodiscard]] std::string_view chatRoleName(ChatRole r) noexcept;

/// Parsed role name. Returns false if the string is not a recognised
/// role. Case-sensitive (OpenAI uses lowercase).
[[nodiscard]] bool parseChatRole(std::string_view s, ChatRole& out) noexcept;

struct ChatMessage {
    ChatRole    role{ChatRole::User};
    std::string content;
};

/**
 * Chat-template renderer.
 *
 * M7c supports a single hardcoded style: Qwen2.5 ChatML. The Jinja
 * template stored under `tokenizer.chat_template` in the GGUF is not
 * interpreted — it's logged at INFO so divergence shows up loud.
 *
 * Special tokens `<|im_start|>` and `<|im_end|>` are emitted directly
 * as their token IDs (looked up via Tokenizer::findToken) — they are
 * NOT routed through BPE. Free-text spans between specials are
 * encoded via the tokenizer's regular encode() call, matching the
 * llama.cpp `parse_special=true` behaviour.
 */
class ChatTemplate {
public:
    enum class Style {
        QwenChatML,   ///< Qwen2 / Qwen2.5 / Qwen3 — <|im_start|>/<|im_end|>.
        Gemma3,       ///< Gemma 2/3 — symmetric <start_of_turn>/<end_of_turn>.
        Gemma4,       ///< Gemma 4 — asymmetric <|turn>/<turn|>. Also has the
                      ///<              thinking-channel markup <|channel>/<channel|>
                      ///<              wrapping the model's response.
    };

    /// Pick a template style by GGUF `general.architecture`. Throws
    /// std::runtime_error for architectures we have not hardcoded yet.
    [[nodiscard]] static Style detectFromArch(std::string_view architecture);

    /**
     * Build the token sequence for `messages` using `style`.
     *
     * If `addGenerationPrompt` is true (the default for serve-mode
     * /v1/chat/completions calls), the output ends with the model's
     * own assistant header (`<|im_start|>assistant\n` for Qwen) so the
     * decoder continues from there.
     *
     * Throws if the tokenizer is missing a required special token id
     * — that's a mismatch between the chat-template style and the
     * loaded GGUF vocab.
     */
    [[nodiscard]] static std::vector<std::int32_t>
    encode(Style                          style,
           const Tokenizer&               tok,
           std::span<const ChatMessage>   messages,
           bool                           addGenerationPrompt = true);

    /**
     * Token ids that should terminate decoding for `style`, in addition
     * to the tokenizer's EOS. For Qwen this is `<|im_end|>` — model
     * emits it to signal end-of-turn (the actual EOS marker
     * `<|endoftext|>` is rare in chat completions).
     */
    [[nodiscard]] static std::vector<std::int32_t>
    stopIds(Style style, const Tokenizer& tok);

    /**
     * Strip style-specific control markup from a freshly-decoded model
     * response so that what reaches an OpenAI-compatible client is just
     * the visible content.
     *
     *   QwenChatML: no-op, the model's output is already plain text.
     *   Gemma3:     trim trailing <end_of_turn> if it slipped through.
     *   Gemma4:     drop the leading <|channel>thought\n<channel|> wrapper
     *               (Gemma 4 emits an empty thinking channel even when
     *               thinking mode is off), drop a trailing <turn|>.
     *
     * Safe to call on partial / streaming chunks: leading wrapper is only
     * removed when fully present in the input.
     */
    [[nodiscard]] static std::string
    cleanResponse(Style style, std::string_view text);
};

} // namespace mimirmind::model