// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "server/ToolCallParserRegistry.hpp"

#include "model/ChatTemplate.hpp"
#include "server/GemmaToolParser.hpp"
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
    _parsers.push_back(std::make_unique<GemmaToolParser>());
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

const IToolCallParser* ToolCallParserRegistry::resolve(
    std::string_view configName, std::string_view architecture) const noexcept {
    // 1. Explicit per-model config wins (nullptr if it names an unknown parser).
    if (!configName.empty()) {
        return get(configName);
    }
    // 2. Auto-detect from the model's chat style + tool format. detectFromArch
    //    throws on an architecture we have not hardcoded — treat that as "no
    //    auto-detect" (nullptr) rather than propagating.
    using CT = model::ChatTemplate;
    CT::Style style{};
    try {
        style = CT::detectFromArch(architecture);
    } catch (...) {
        return nullptr;
    }
    switch (style) {
    case CT::Style::Gemma4:
        return get("gemma");
    case CT::Style::QwenChatML:
        return CT::toolFormatFromArch(architecture) == CT::ToolFormat::QwenXml
                   ? get("qwen3-coder-xml")
                   : get("hermes");
    case CT::Style::Gemma3:
    case CT::Style::Llama3:
    default:
        // No dedicated parser yet (Gemma3 DSL differs; Llama3 deferred) — the
        // caller keeps its current behaviour.
        return nullptr;
    }
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
