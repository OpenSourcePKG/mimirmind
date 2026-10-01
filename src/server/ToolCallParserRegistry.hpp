// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "server/IToolCallParser.hpp"

#include <memory>
#include <string_view>
#include <vector>

namespace mimirmind::server {

/// 8.25.2 — the named tool-call parser registry (vLLM's ToolParserManager
/// equivalent). Owns one stateless const instance per registered name; the
/// server resolves a model's parser by name (config.serve.json + auto-detect,
/// 8.25.4). Parsers are const + stateless, so single shared instances are safe
/// across concurrent streams.
class ToolCallParserRegistry {
public:
    /// Process-wide registry with the built-in parsers registered once.
    [[nodiscard]] static const ToolCallParserRegistry& instance();

    /// Resolve a parser by registry name; nullptr if unknown (caller falls back).
    [[nodiscard]] const IToolCallParser* get(std::string_view name) const noexcept;

    /// 8.25.4 — server-side parser selection (no user toggle). Priority:
    ///   1. explicit `configName` (per-model config.serve.json tool_call_parser) —
    ///      wins even over auto-detect; nullptr if it names an unknown parser
    ///      (a misconfiguration the caller surfaces / falls back on).
    ///   2. auto-detect from the model `architecture` (chat-style + tool-format):
    ///      Gemma4 -> "gemma"; QwenChatML -> "qwen3-coder-xml" (QwenXml) or
    ///      "hermes" (HermesJson).
    ///   3. nullptr for anything else (Gemma3 / Llama3 / unknown arch) — the
    ///      caller keeps its current behaviour.
    /// Never throws (an unknown architecture is treated as "no auto-detect").
    [[nodiscard]] const IToolCallParser* resolve(std::string_view configName,
                                                 std::string_view architecture) const noexcept;

    /// All registered parser names (for logging / diagnostics / config validation).
    [[nodiscard]] std::vector<std::string_view> names() const;

    ToolCallParserRegistry(const ToolCallParserRegistry&)            = delete;
    ToolCallParserRegistry& operator=(const ToolCallParserRegistry&) = delete;

private:
    ToolCallParserRegistry();   // registers the built-ins

    std::vector<std::unique_ptr<IToolCallParser>> _parsers;
};

} // namespace mimirmind::server
