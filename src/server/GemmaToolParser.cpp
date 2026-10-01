// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "server/GemmaToolParser.hpp"

#include "model/ChatTemplate.hpp"
#include "model/ToolCallParser.hpp"
#include "server/ToolCallExtractor.hpp"

namespace mimirmind::server {

bool GemmaToolParser::matches(std::string_view text) const noexcept {
    return model::ToolCallParser::looksLikeGemmaToolCall(text);
}

ToolCallExtraction GemmaToolParser::extract(
    std::string_view                 text,
    std::span<const model::ToolSpec> /*tools*/) const {
    ToolCallExtraction out;
    if (model::ToolCallParser::looksLikeGemmaToolCall(text)) {
        out.sawToolMarkup = true;
        out.calls         = model::ToolCallParser::parseGemma(text);
    }
    return out;
}

std::vector<model::ToolCall> GemmaToolParser::extractBlock(
    std::string_view                 block,
    std::span<const model::ToolSpec> tools,
    bool                             bare) const {
    // Style::Gemma4 routes the shared block dispatcher through parseGemma; the
    // tool format is irrelevant for Gemma (ignored by that path).
    return ToolCallExtractor::extractBlock(
        block, model::ChatTemplate::Style::Gemma4,
        model::ChatTemplate::ToolFormat::HermesJson, tools, bare);
}

} // namespace mimirmind::server
