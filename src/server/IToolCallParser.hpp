// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "model/ToolCall.hpp"

#include <span>
#include <string_view>
#include <vector>

namespace mimirmind::server {

/// 8.25.2 — the decode-side counterpart to the 8.24 Jinja renderer: one
/// interface for parsing a model's output back into OpenAI `tool_calls`, so the
/// per-checkpoint format branching (Hermes-JSON vs Qwen3-Coder-XML vs bare vs
/// Gemma tool_code) lives behind named, server-selected parser classes instead
/// of a God-function spread across the handler. Mirrors vLLM's ToolParser
/// (extract_tool_calls + extract_tool_calls_streaming); see the ADR
/// `2026-10-01-tool-call-parser-registry`.
///
/// Parsers are PURE: they return the parsed calls (plus signals the handler uses
/// to reproduce its logging / content-suppression), and do NO content mutation,
/// logging, name-normalisation or salvage re-decode — those side-effects stay in
/// the handler exactly as today (ADR scope).

/// Result of a blocking (whole-response) parse. The signal flags mirror the
/// triggers the current handler branch consults so the migration (8.25.6) can
/// reproduce the existing side-effects byte-for-byte.
struct ToolCallExtraction {
    std::vector<model::ToolCall> calls;
    /// A control-token-noised span was salvaged (parseQwenXmlNoisy produced the
    /// result) — the handler logs it and drops the `<tool_call>` span.
    bool usedNoisy{false};
    /// The native ```tool_code fence produced the result — the handler logs it
    /// and clears the content (the fence was the whole reply).
    bool fromToolCode{false};
    /// The text looked like this parser's tool-call format (even if the parse
    /// came back empty) — the handler suppresses the raw marker span from the
    /// visible content instead of leaking markup to the client.
    bool sawToolMarkup{false};
};

/// A named, stateless tool-call parser for one model family/dialect. Selected
/// per-model by the server (config.serve.json + auto-detect, 8.25.4); never a
/// user toggle.
class IToolCallParser {
public:
    virtual ~IToolCallParser() = default;

    /// Registry name, e.g. "qwen3-coder-xml" / "hermes" / "llama3-json".
    [[nodiscard]] virtual std::string_view name() const noexcept = 0;

    /// Cheap pre-check: does `text` look like this parser's tool-call format?
    /// Used for auto-detect and to skip parsing on a plain prose reply, mirroring
    /// the handler's looksLike* dispatch. Must not allocate on the hot no-call
    /// path.
    [[nodiscard]] virtual bool matches(std::string_view text) const noexcept = 0;

    /// Blocking parse of a whole decoded response span.
    [[nodiscard]] virtual ToolCallExtraction extract(
        std::string_view                 text,
        std::span<const model::ToolSpec> tools) const = 0;

    /// Streaming granularity: parse ONE marker-delimited block the stream
    /// detector handed over. `bare` = the outer `<tool_call>` envelope was
    /// already dropped (the block starts at `<function=`). Returns the calls in
    /// that block; the stream state machine owns detection and delta emission.
    [[nodiscard]] virtual std::vector<model::ToolCall> extractBlock(
        std::string_view                 block,
        std::span<const model::ToolSpec> tools,
        bool                             bare) const = 0;
};

} // namespace mimirmind::server
