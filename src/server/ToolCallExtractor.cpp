// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "server/ToolCallExtractor.hpp"

#include "model/ToolCallParser.hpp"

#include <string>

namespace mimirmind::server {

std::vector<model::ToolCall> ToolCallExtractor::parseQwenDialects(
    std::string_view                 block,
    model::ChatTemplate::ToolFormat  toolFormat,
    std::span<const model::ToolSpec> toolSpecs,
    bool                             tryNoisy,
    bool*                            usedNoisy) {
    // 8.19.6: both Qwen dialects share the outer <tool_call> marker; try the
    // model's native one first, then the other — Qwen3.6 has been seen imitating
    // the Hermes shape when driven by agent prompts, and vice-versa a Hermes
    // model can drift XML-wards.
    const bool xmlNative =
        toolFormat == model::ChatTemplate::ToolFormat::QwenXml;
    std::vector<model::ToolCall> calls =
        xmlNative ? model::ToolCallParser::parseQwenXml(block, toolSpecs)
                  : model::ToolCallParser::parseQwen(block);
    if (calls.empty()) {
        calls = xmlNative ? model::ToolCallParser::parseQwen(block)
                          : model::ToolCallParser::parseQwenXml(block, toolSpecs);
    }
    if (calls.empty() && tryNoisy) {
        // Salvage a control-token-noised call (Qwen3-Coder-Next injects
        // <|im_start|> into the tool-call span under heavy tool prompts,
        // dropping the '<' of <function=). Only offered names survive.
        calls = model::ToolCallParser::parseQwenXmlNoisy(block, toolSpecs);
        if (!calls.empty() && usedNoisy != nullptr) {
            *usedNoisy = true;
        }
    }
    return calls;
}

std::vector<model::ToolCall> ToolCallExtractor::extractBlock(
    std::string_view                 block,
    model::ChatTemplate::Style       style,
    model::ChatTemplate::ToolFormat  toolFormat,
    std::span<const model::ToolSpec> toolSpecs,
    bool                             bare) {
    if (style == model::ChatTemplate::Style::Gemma4) {
        return model::ToolCallParser::parseGemma(block);
    }
    // A block captured via the BARE `<function=` opener (8.19.9 — envelope
    // dropped) goes through the name-gated bare parser.
    if (bare) {
        return model::ToolCallParser::parseQwenXmlBare(block, toolSpecs);
    }
    // Same native-dialect-first, other-dialect-fallback order as the blocking
    // path (8.19.6); noisy salvage only when not a bare capture.
    return parseQwenDialects(block, toolFormat, toolSpecs, /*tryNoisy=*/!bare);
}

ToolCallExtractor::FallbackResult ToolCallExtractor::extractBareFallbacks(
    std::string_view                 text,
    std::span<const model::ToolSpec> toolSpecs) {
    FallbackResult out;
    // 8.19.9 — bare-XML fallback: after repeated error tool_responses Qwen3.6
    // drops the <tool_call> envelope but keeps the <function=…>…</function>
    // payload (often with a dangling </tool_call> at the end). Name-gated on the
    // offered tools and fenced code blocks are ignored, so prose showing an
    // example never parses as a call.
    if (model::ToolCallParser::looksLikeBareQwenXmlCall(text, toolSpecs)) {
        out.calls = model::ToolCallParser::parseQwenXmlBare(text, toolSpecs);
    }
    // Bare-JSON fallback: a reasoning model (Qwen3.6) sometimes emits the call
    // as plain {"name":…,"arguments":…} with no <tool_call> wrapper, which would
    // otherwise leak into content. Gate on the offered tool names so a legit
    // JSON answer is never parsed as a call.
    if (out.calls.empty()) {
        std::vector<std::string> toolNames;
        toolNames.reserve(toolSpecs.size());
        for (const auto& spec : toolSpecs) {
            toolNames.push_back(spec.name);
        }
        if (model::ToolCallParser::looksLikeBareJsonToolCall(text, toolNames)) {
            out.calls = model::ToolCallParser::parseBareJson(text, toolNames);
        }
    }
    // tool_code fallback: Coder-Next's native auto reply is a ```tool_code /
    // ```python fence with a Python NAME(args) call (no XML envelope).
    // Offered-name + call-syntax gated.
    if (out.calls.empty()) {
        out.calls = model::ToolCallParser::parseToolCodeCall(text, toolSpecs);
        if (!out.calls.empty()) {
            out.fromToolCode = true;
        }
    }
    return out;
}

std::vector<model::ToolCall> ToolCallExtractor::parseCanonicalXml(
    std::string_view                 text,
    std::span<const model::ToolSpec> toolSpecs) {
    return model::ToolCallParser::parseQwenXml(text, toolSpecs);
}

} // namespace mimirmind::server
