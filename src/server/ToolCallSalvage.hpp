// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "model/ChatTemplate.hpp"          // ChatTemplate::Style
#include "model/ToolCall.hpp"              // ToolSpec
#include "runtime/InferenceEngine.hpp"     // GenerateParams

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace mimirmind::model {
class Tokenizer;
class ToolCallConstraint;
}

namespace mimirmind::server {

/**
 * Shared one-shot tool-call salvage re-decode (roadmap 8.30.11.2). The blocking
 * and streaming chat paths each detected a tool-intent leak that parsed into no
 * (or an incomplete) call and force-prefilled the canonical `<tool_call>\n
 * <function=` opener to re-decode ONCE — the two copies were a near byte-twin
 * (the top maintenance hazard). This collapses the identical core (build salvage
 * prompt + bounded re-decode params + optional body-rooted grammar mask +
 * execute) into one place; each caller keeps its own trigger detection and
 * post-processing (which legitimately differ) and supplies the executor.
 */
class ToolCallSalvage {
public:
    /// Runs the actual bounded re-decode of `salvagePrompt` with sampling `sp`
    /// and the optional grammar `constraint`, returning the generated token ids
    /// (empty on failure). Supplied by the caller so the batcher-vs-engine path,
    /// callbacks, mutex and tenant stay owned by the caller.
    using Redecoder = std::function<std::vector<std::int32_t>(
        const std::vector<std::int32_t>&           salvagePrompt,
        const runtime::GenerateParams&             sp,
        std::shared_ptr<model::ToolCallConstraint> constraint)>;

    /// Build the salvage prompt (promptIds + the style's canonical salvage
    /// opener), the bounded re-decode params (maxNewTokens capped at 1024; a
    /// body-rooted `ToolCallConstraint` when `grammarOn`), pick the temperature
    /// policy, and run `redecoder`. Returns the re-decoded token ids (empty when
    /// the style has no salvage opener, the executor throws, or nothing was
    /// produced). Parsing + keep-original/suppress decisions stay with the
    /// caller.
    ///
    /// `alwaysGreedy`: force temperature 0 even when a grammar mask is active
    /// (the streaming path); when false, greedy is used only WITHOUT a mask so
    /// a masked re-decode keeps the request's own (anti-loop-lifted) sampling
    /// (the blocking path — see 8.19.13.5).
    /// `logCtx`: prefix for the "re-decode failed" warning (e.g. "stream 42: ").
    [[nodiscard]] static std::vector<std::int32_t> redecode(
        const std::vector<std::int32_t>&  promptIds,
        const runtime::GenerateParams&    params,
        std::span<const model::ToolSpec>  tools,
        const model::Tokenizer&           tok,
        model::ChatTemplate::Style        style,
        bool                              grammarOn,
        bool                              alwaysGreedy,
        const Redecoder&                  redecoder,
        std::string_view                  logCtx);
};

} // namespace mimirmind::server
