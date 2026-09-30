// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "model/ChatTemplate.hpp"   // ChatMessage
#include "model/ToolCall.hpp"       // ToolSpec

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace mimirmind::model {
class Tokenizer;
}

namespace mimirmind::model::chat {

/// Gemma end-of-turn tokens. Live here because they are shared between the
/// encoder and ChatTemplate's stopIds / cleanResponse metadata (single source
/// of truth). The start-of-turn + thinking-channel markers stay private to the
/// respective TUs.
inline constexpr std::string_view kGemma3EndOfTurn = "<end_of_turn>";
inline constexpr std::string_view kGemma4EndOfTurn = "<turn|>";

/// Per-family chat encoder for the Gemma 2/3/4 templates. Gemma folds system
/// turns into the first user turn (no separate system role); Gemma 4 adds a
/// custom special-token tool DSL. Extracted verbatim from the former free
/// encodeGemma3/encodeGemma4/encodeGemmaImpl (roadmap 8.30.11.3);
/// PARITY-CRITICAL.
class GemmaChatEncoder {
public:
    /// Gemma 2/3 symmetric <start_of_turn>/<end_of_turn> template. Tool
    /// rendering is not implemented for Gemma 3 (Gemma 4 is the target).
    [[nodiscard]] static std::vector<std::int32_t> encodeGemma3(
        const Tokenizer&              tok,
        std::span<const ChatMessage>  messages,
        bool                          addGenerationPrompt);

    /// Gemma 4 asymmetric <|turn>/<turn|> template. With tools present, routes
    /// to the dedicated tool-DSL path (tools in their own system turn).
    [[nodiscard]] static std::vector<std::int32_t> encodeGemma4(
        const Tokenizer&              tok,
        std::span<const ChatMessage>  messages,
        bool                          addGenerationPrompt,
        std::span<const ToolSpec>     tools);
};

} // namespace mimirmind::model::chat
