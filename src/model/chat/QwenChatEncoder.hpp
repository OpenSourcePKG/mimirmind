// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "model/ChatTemplate.hpp"   // ChatMessage, ChatRole, ChatTemplate::ToolFormat
#include "model/ToolCall.hpp"       // ToolSpec

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace mimirmind::model {
class Tokenizer;
}

namespace mimirmind::model::chat {

/// Qwen ChatML end-of-turn token. Lives here because it is shared between the
/// encoder and ChatTemplate's stop-id metadata (single source of truth).
inline constexpr std::string_view kQwenImEnd = "<|im_end|>";

/// Per-family chat encoder for the Qwen ChatML family (Qwen2/2.5/3/3.5/3.6/3.8,
/// incl. Qwen3-Coder-Next). Renders the exact prompt-token sequence the model's
/// chat_template produces — HermesJson vs QwenXml tool dialects, the default
/// system turn, and the pre-open/pre-close <think> block. Extracted verbatim
/// from the former free `encodeQwen` (roadmap 8.30.11.3). PARITY-CRITICAL:
/// changes must be bit-verified via the prompt-token dump vs the reference.
class QwenChatEncoder {
public:
    [[nodiscard]] static std::vector<std::int32_t> encode(
        const Tokenizer&              tok,
        std::span<const ChatMessage>  messages,
        bool                          addGenerationPrompt,
        std::span<const ToolSpec>     tools,
        std::optional<bool>           enableThinking,
        ChatTemplate::ToolFormat      toolFormat,
        std::optional<bool>           templateUsesThink,
        std::optional<bool>           toolDefsStructuredXml);
};

} // namespace mimirmind::model::chat
