// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "model/ChatTemplate.hpp"   // ToolFormat
#include "server/IToolCallParser.hpp"

#include <string>
#include <string_view>

namespace mimirmind::server {

/// 8.25.2 — the Qwen-family tool-call parser (Qwen2/2.5/3 = Hermes-JSON,
/// Qwen3.5/3.6/3.8 = Qwen3-Coder-XML). Both dialects share the outer
/// `<tool_call>` marker, so one parser handles the family: it tries the model's
/// NATIVE dialect first, then the other, then control-token-noised salvage —
/// and falls back to the bare / tool_code ladder when the envelope is absent.
/// This WRAPS the proven `ToolCallExtractor` ladders (8.30.5) verbatim, so it is
/// behaviour-identical to the current handler branch; the handler is not yet
/// routed through it (migration = 8.25.6).
///
/// `nativeFormat` only sets the dialect ORDER (native-first); both dialects are
/// always attempted. The registry exposes two names over this one class:
/// "qwen3-coder-xml" (QwenXml native) and "hermes" (HermesJson native).
class QwenToolParser final : public IToolCallParser {
public:
    QwenToolParser(std::string name, model::ChatTemplate::ToolFormat nativeFormat)
        : _name(std::move(name)), _format(nativeFormat) {}

    [[nodiscard]] std::string_view name() const noexcept override { return _name; }

    [[nodiscard]] bool matches(std::string_view text) const noexcept override;

    [[nodiscard]] ToolCallExtraction extract(
        std::string_view                 text,
        std::span<const model::ToolSpec> tools) const override;

    [[nodiscard]] std::vector<model::ToolCall> extractBlock(
        std::string_view                 block,
        std::span<const model::ToolSpec> tools,
        bool                             bare) const override;

private:
    std::string                     _name;
    model::ChatTemplate::ToolFormat _format;
};

} // namespace mimirmind::server
