// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "server/ToolCallParserRegistry.hpp"

#include "model/ChatTemplate.hpp"
#include "server/QwenToolParser.hpp"

namespace mimirmind::server {

ToolCallParserRegistry::ToolCallParserRegistry() {
    // Built-in parsers. One QwenToolParser class serves both Qwen dialects; the
    // name only sets the native-dialect order (native-first, the other is still
    // attempted). Gemma / Llama3 / standalone-bare parsers land in 8.25.3.
    _parsers.push_back(std::make_unique<QwenToolParser>(
        "qwen3-coder-xml", model::ChatTemplate::ToolFormat::QwenXml));
    _parsers.push_back(std::make_unique<QwenToolParser>(
        "hermes", model::ChatTemplate::ToolFormat::HermesJson));
}

const ToolCallParserRegistry& ToolCallParserRegistry::instance() {
    static const ToolCallParserRegistry kRegistry;
    return kRegistry;
}

const IToolCallParser* ToolCallParserRegistry::get(std::string_view name) const noexcept {
    for (const auto& p : _parsers) {
        if (p->name() == name) {
            return p.get();
        }
    }
    return nullptr;
}

std::vector<std::string_view> ToolCallParserRegistry::names() const {
    std::vector<std::string_view> out;
    out.reserve(_parsers.size());
    for (const auto& p : _parsers) {
        out.push_back(p->name());
    }
    return out;
}

} // namespace mimirmind::server
