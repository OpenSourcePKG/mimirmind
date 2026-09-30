// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

namespace mimirmind::model {
class Tokenizer;
}

namespace mimirmind::model::chat {

/// Resolve a required special token id or throw a labelled runtime_error.
/// Shared by every per-family chat encoder (roadmap 8.30.11.3 / 8.24).
[[nodiscard]] std::int32_t requireToken(const Tokenizer& tok,
                                        std::string_view  text);

/// Append the tokenisation of `text` (addBos=false) to `out`. No-op when
/// `text` is empty. Shared by every per-family chat encoder.
void encodeText(const Tokenizer&           tok,
                std::string_view           text,
                std::vector<std::int32_t>& out);

} // namespace mimirmind::model::chat
