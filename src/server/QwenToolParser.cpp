// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "server/QwenToolParser.hpp"

#include "model/ToolCallParser.hpp"
#include "server/ToolCallExtractor.hpp"

namespace mimirmind::server {

bool QwenToolParser::matches(std::string_view text) const noexcept {
    // The outer <tool_call> marker covers both Qwen dialects. The bare /
    // tool_code fallbacks are name-gated and only consulted on extract(), so
    // matches() stays a cheap envelope pre-check (same as the handler's primary
    // looksLikeQwenToolCall dispatch).
    return model::ToolCallParser::looksLikeQwenToolCall(text);
}

ToolCallExtraction QwenToolParser::extract(
    std::string_view                 text,
    std::span<const model::ToolSpec> tools) const {
    ToolCallExtraction out;
    if (model::ToolCallParser::looksLikeQwenToolCall(text)) {
        // Native-first / other-dialect / noisy-salvage ladder (8.30.5).
        out.sawToolMarkup = true;
        bool usedNoisy = false;
        out.calls = ToolCallExtractor::parseQwenDialects(
            text, _format, tools, /*tryNoisy=*/true, &usedNoisy);
        out.usedNoisy = usedNoisy;
    } else {
        // Envelope-less fallback ladder: bare-XML -> bare-JSON -> tool_code.
        auto fb = ToolCallExtractor::extractBareFallbacks(text, tools);
        out.calls        = std::move(fb.calls);
        out.fromToolCode = fb.fromToolCode;
    }
    return out;
}

std::vector<model::ToolCall> QwenToolParser::extractBlock(
    std::string_view                 block,
    std::span<const model::ToolSpec> tools,
    bool                             bare) const {
    // The streaming path hands over one detected block; delegate to the shared
    // block dispatcher with the Qwen chat style and this parser's native dialect.
    return ToolCallExtractor::extractBlock(
        block, model::ChatTemplate::Style::QwenChatML, _format, tools, bare);
}

} // namespace mimirmind::server
