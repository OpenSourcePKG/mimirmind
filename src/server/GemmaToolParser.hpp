// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "server/IToolCallParser.hpp"

#include <string_view>

namespace mimirmind::server {

/// 8.25.3 — the Gemma-4 tool-call parser: Gemma emits its own DSL (`<|tool_call>`
/// blocks with `<|"|>…<|"|>`-quoted values, the leading pipe distinguishing it
/// from Qwen's `<tool_call>`). Wraps the proven `model::ToolCallParser::parseGemma`
/// primitive verbatim, so it is behaviour-identical to the handler's Gemma branch
/// (not yet routed through it; migration = 8.25.6).
///
/// Note: the ```tool_code fence (`parseToolCodeCall`) is NOT Gemma-exclusive —
/// Qwen3-Coder-Next emits it too — so it stays in the shared bare/fallback ladder
/// (QwenToolParser's envelope-less branch), not here.
class GemmaToolParser final : public IToolCallParser {
public:
    [[nodiscard]] std::string_view name() const noexcept override { return "gemma"; }

    [[nodiscard]] bool matches(std::string_view text) const noexcept override;

    [[nodiscard]] ToolCallExtraction extract(
        std::string_view                 text,
        std::span<const model::ToolSpec> tools) const override;

    [[nodiscard]] std::vector<model::ToolCall> extractBlock(
        std::string_view                 block,
        std::span<const model::ToolSpec> tools,
        bool                             bare) const override;
};

} // namespace mimirmind::server
