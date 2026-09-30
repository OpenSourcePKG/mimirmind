// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "model/ChatTemplate.hpp"   // ChatMessage

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace mimirmind::model {
class Tokenizer;
}

namespace mimirmind::model::chat {

/// Llama-3.x end-of-turn token. Shared between the encoder and ChatTemplate's
/// stop-id / cleanResponse metadata (single source of truth).
inline constexpr std::string_view kLlama3Eot = "<|eot_id|>";

/// Per-family chat encoder for the Llama-3.x header-token template
/// (Llama-3.0/3.1/3.2). Renders `<|begin_of_text|>` +
/// `<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>` per turn,
/// plus the assistant generation-prompt header. No default system turn is
/// injected (unlike Qwen2.5). Tool calling is not wired for this style yet.
/// Extracted verbatim from the former free `encodeLlama3` (roadmap 8.30.11.3);
/// PARITY-CRITICAL.
class Llama3ChatEncoder {
public:
    [[nodiscard]] static std::vector<std::int32_t> encode(
        const Tokenizer&              tok,
        std::span<const ChatMessage>  messages,
        bool                          addGenerationPrompt);
};

} // namespace mimirmind::model::chat
