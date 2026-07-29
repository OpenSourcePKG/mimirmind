// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "model/ToolCall.hpp"

#include <string>
#include <string_view>
#include <vector>

namespace mimirmind::model {

/// Extracts a model's *native* tool-call markup from decoded output and
/// normalises it to the OpenAI {@link ToolCall} shape. Per-architecture,
/// because each model family wraps calls differently. Returns an empty
/// vector when the text carries no well-formed tool call — the caller then
/// treats the output as a plain assistant answer.
class ToolCallParser {
public:
    /// Qwen2 / Qwen2.5 (Hermes format): one or more
    ///   <tool_call>
    ///   {"name": "fn", "arguments": { ... }}
    ///   </tool_call>
    /// blocks. Whitespace around the JSON is tolerated; `arguments` may be a
    /// JSON object (normalised to a compact string) or already a string.
    /// A block whose JSON is malformed or lacks a string `name` is skipped
    /// rather than aborting the whole parse. Ids are synthesised "call_0",
    /// "call_1", … in emission order.
    [[nodiscard]] static std::vector<ToolCall> parseQwen(std::string_view text);

    /// True when `text` contains at least one `<tool_call>` opener — a cheap
    /// pre-check so the hot path (no tools in the request) never parses JSON.
    [[nodiscard]] static bool looksLikeQwenToolCall(std::string_view text) noexcept;

    /// Gemma 4: one or more blocks in Gemma's custom tool-call DSL
    ///   <|tool_call>call:NAME{key:VALUE,key2:VALUE2,...}<tool_call|>
    /// where VALUE is a Gemma-quoted string `<|"|>text<|"|>` (a plain
    /// "text" is also tolerated), `true`/`false`, a number, a nested
    /// `{...}` object, or a `[...]` array. Arguments are normalised to a JSON
    /// string on each ToolCall. A malformed block is skipped. Ids are
    /// synthesised "call_0", "call_1", … in emission order.
    [[nodiscard]] static std::vector<ToolCall> parseGemma(std::string_view text);

    /// True when `text` contains at least one `<|tool_call>` opener (the
    /// Gemma marker — note the leading pipe, distinct from Qwen's).
    [[nodiscard]] static bool looksLikeGemmaToolCall(std::string_view text) noexcept;
};

} // namespace mimirmind::model
