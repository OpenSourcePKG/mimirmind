// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "model/ChatTemplate.hpp"
#include "model/ToolCall.hpp"

#include <span>
#include <string_view>
#include <vector>

namespace mimirmind::server {

/// 8.30.5 — shared home for the tool-call PARSE ladders that the blocking and
/// streaming chat paths otherwise duplicate. It wraps the primitives in
/// `model::ToolCallParser` (parseQwen / parseQwenXml / parseQwenXmlBare /
/// parseQwenXmlNoisy / parseBareJson / parseToolCodeCall + parseGemma) into the
/// exact ladders both handlers ran inline.
///
/// IMPORTANT: the two call sites differ in input granularity — the blocking
/// path parses the WHOLE response text, the stream path parses ONE
/// marker-delimited block — and their ladder ORDER is preserved here via
/// parameters/flags rather than forced into a single order. This class does NO
/// content mutation, logging, name-normalisation or salvage re-decode: those
/// side-effects stay in the handler, which calls these pure parse helpers.
class ToolCallExtractor {
public:
    ToolCallExtractor() = delete;

    /// The Qwen-dialect ladder shared by the blocking Qwen branch and (via
    /// {@link extractBlock}) the streaming block dispatcher: try the model's
    /// NATIVE dialect first (parseQwenXml when the arch is QwenXml, else
    /// parseQwen), then the OTHER dialect, and — only when `tryNoisy` is set and
    /// both came back empty — the control-token-noised salvage
    /// (parseQwenXmlNoisy). When `usedNoisy` is non-null it is set to true iff
    /// the noisy stage is what produced the (non-empty) result, so the caller
    /// can reproduce its noisy-specific logging / content suppression.
    [[nodiscard]] static std::vector<model::ToolCall> parseQwenDialects(
        std::string_view                       block,
        model::ChatTemplate::ToolFormat        toolFormat,
        std::span<const model::ToolSpec>       toolSpecs,
        bool                                   tryNoisy,
        bool*                                  usedNoisy = nullptr);

    /// Full block dispatcher used by the streaming path's emitToolCallBlock:
    /// a Gemma4 style parses with parseGemma; a `bare` block (envelope dropped,
    /// starts with `<function=`) parses with the name-gated parseQwenXmlBare;
    /// otherwise it runs the Qwen-dialect ladder (with noisy salvage). Mirrors
    /// the blocking path's native-first / other / noisy order.
    [[nodiscard]] static std::vector<model::ToolCall> extractBlock(
        std::string_view                       block,
        model::ChatTemplate::Style             style,
        model::ChatTemplate::ToolFormat        toolFormat,
        std::span<const model::ToolSpec>       toolSpecs,
        bool                                   bare);

    /// Result of the envelope-less whole-response fallback ladder.
    struct FallbackResult {
        std::vector<model::ToolCall> calls;
        /// True iff `calls` came from the native tool_code fence parser — the
        /// caller uses this to reproduce the tool_code-specific log line (and,
        /// on the blocking path, the "the fence was the whole reply" content
        /// clear).
        bool                         fromToolCode = false;
    };

    /// The bare / envelope-less fallback ladder shared by the blocking else
    /// branch and the streaming end-of-turn held-content resolution: name-gated
    /// bare-XML (parseQwenXmlBare) → bare-JSON (parseBareJson) → native
    /// tool_code fence (parseToolCodeCall), in that order, each gated by its
    /// looksLike* predicate exactly as before.
    [[nodiscard]] static FallbackResult extractBareFallbacks(
        std::string_view                 text,
        std::span<const model::ToolSpec> toolSpecs);

    /// Parse a canonical `<tool_call>\n<function=…` block — the single parse
    /// attempt the salvage re-decode makes on both paths (parseQwenXml over the
    /// re-decoded, opener-prefixed text). Kept here so the salvage
    /// orchestration in the handler routes its parse through this class.
    [[nodiscard]] static std::vector<model::ToolCall> parseCanonicalXml(
        std::string_view                 text,
        std::span<const model::ToolSpec> toolSpecs);
};

} // namespace mimirmind::server
