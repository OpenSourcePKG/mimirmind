// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "model/ToolCall.hpp"

#include <span>
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
    /// Shared by parseQwen and parseQwenXml (same outer marker).
    [[nodiscard]] static bool looksLikeQwenToolCall(std::string_view text) noexcept;

    /// Qwen3.5 / Qwen3.6 / Qwen3.8 (Qwen3-Coder XML format): one or more
    ///   <tool_call>
    ///   <function=fn>
    ///   <parameter=key>
    ///   value
    ///   </parameter>
    ///   </function>
    ///   </tool_call>
    /// blocks. Parameter values are plain text spanning multiple lines; they
    /// are coerced to typed JSON via the offered tool's parameter schema in
    /// `specs` (integer/number/boolean/object/array), defaulting to string —
    /// mirroring vLLM's qwen3coder tool parser. A truncated final block
    /// (missing closers, e.g. cut at max_tokens) still yields the complete
    /// parameters seen so far; a block without a `<function=` header is
    /// skipped. Ids are synthesised "call_0", "call_1", … in emission order.
    [[nodiscard]] static std::vector<ToolCall>
    parseQwenXml(std::string_view text, std::span<const ToolSpec> specs);

    /// 8.19.9 — same dialect WITHOUT the `<tool_call>` envelope: under
    /// pressure (repeated error tool_responses) Qwen3.6 emits the bare
    /// `<function=NAME>…</function>` body, often with a dangling closing
    /// `</tool_call>` at the end. Same failure class as the bare-JSON leak
    /// (ffb6a4f): envelope dropped, payload kept. Guard rails: a call is
    /// accepted only when NAME matches an offered tool in `specs`, and
    /// `<function=` occurrences inside ```fenced``` code blocks are ignored
    /// (documentation, not a call). Dangling closers are simply never part
    /// of a parsed span.
    [[nodiscard]] static std::vector<ToolCall>
    parseQwenXmlBare(std::string_view text, std::span<const ToolSpec> specs);

    /// Cheap pre-check for `parseQwenXmlBare`: `text` contains
    /// `<function=NAME` for at least one offered tool.
    [[nodiscard]] static bool
    looksLikeBareQwenXmlCall(std::string_view          text,
                             std::span<const ToolSpec> specs) noexcept;

    /// 8.19.10 — degenerate tool-markup "salad" that no parser can rescue
    /// (e.g. `<tooltool_0>\n\n</invoke>`): tag-like spans whose content
    /// smells of tool markup (tool/invoke/function/call inside a `<…>` tag),
    /// or a short response dominated by tag characters. Used to trigger the
    /// one-shot salvage re-decode with a force-prefilled opener — NOT a
    /// parser; there is nothing to parse. Deliberately conservative: long
    /// tag-heavy answers without the keywords (real HTML/XML output) do NOT
    /// match.
    [[nodiscard]] static bool
    looksLikeToolMarkupSalad(std::string_view text) noexcept;

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

    /// Fallback for reasoning models (e.g. Qwen3.6) that sometimes emit the
    /// call as a bare JSON object with NO `<tool_call>` wrapper —
    ///   {"name": "fn", "arguments": { ... }}
    /// possibly several concatenated, embedded in surrounding text. Every
    /// balanced `{…}` is tried; an object is accepted only when its string
    /// `name` is one of `knownNames` (the tools the request offered), so a
    /// legitimate JSON *answer* is never mistaken for a call. `arguments` is
    /// normalised to a compact JSON string; ids are "call_0", "call_1", ….
    [[nodiscard]] static std::vector<ToolCall>
    parseBareJson(std::string_view text, const std::vector<std::string>& knownNames);

    /// Cheap pre-check for `parseBareJson`: `text` mentions a `"name"` key and
    /// at least one of the offered tool names. Gate the (JSON-scanning) parse
    /// behind this so the common no-call answer stays fast.
    [[nodiscard]] static bool
    looksLikeBareJsonToolCall(std::string_view text,
                              const std::vector<std::string>& knownNames) noexcept;
};

} // namespace mimirmind::model
