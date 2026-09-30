// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "model/chat/ChatEncoderCommon.hpp"

#include "model/Tokenizer.hpp"

#include <stdexcept>
#include <string>

namespace mimirmind::model::chat {

std::int32_t requireToken(const Tokenizer& tok, std::string_view text) {
    const std::int32_t id = tok.findToken(text);
    if (id < 0) {
        throw std::runtime_error(
            "ChatTemplate: tokenizer is missing required special token '" +
            std::string{text} + "'");
    }
    return id;
}

void encodeText(const Tokenizer&           tok,
                std::string_view           text,
                std::vector<std::int32_t>& out) {
    if (text.empty()) {
        return;
    }
    const auto ids = tok.encode(text, /*addBos=*/false);
    out.insert(out.end(), ids.begin(), ids.end());
}

} // namespace mimirmind::model::chat
