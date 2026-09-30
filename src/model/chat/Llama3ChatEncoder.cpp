// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "model/chat/Llama3ChatEncoder.hpp"

#include "model/Tokenizer.hpp"
#include "model/chat/ChatEncoderCommon.hpp"

#include <string>

namespace mimirmind::model::chat {

namespace {

// Llama-3.x special tokens (Llama-3.0/3.1/3.2). The GGUF architecture is
// "llama" for all of them; this style additionally assumes the Llama-3 header
// tokens are present in the vocab (requireToken throws otherwise), which also
// safely rejects a Llama-2 "llama" GGUF that uses the [INST] template instead.
constexpr std::string_view kLlama3BeginOfText = "<|begin_of_text|>";
constexpr std::string_view kLlama3StartHeader = "<|start_header_id|>";
constexpr std::string_view kLlama3EndHeader   = "<|end_header_id|>";
// kLlama3Eot is shared and lives in the header.

} // namespace

std::vector<std::int32_t> Llama3ChatEncoder::encode(
        const Tokenizer&              tok,
        std::span<const ChatMessage>  messages,
        bool                          addGenerationPrompt) {
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

} // namespace mimirmind::model::chat
