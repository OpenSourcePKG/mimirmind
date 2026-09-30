// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "server/ToolCallSalvage.hpp"

#include "core/log/Log.hpp"
#include "model/ToolCallConstraint.hpp"
#include "model/Tokenizer.hpp"

#include <algorithm>
#include <exception>
#include <string>

namespace mimirmind::server {

std::vector<std::int32_t> ToolCallSalvage::redecode(
        const std::vector<std::int32_t>&  promptIds,
        const runtime::GenerateParams&    params,
        std::span<const model::ToolSpec>  tools,
        const model::Tokenizer&           tok,
        model::ChatTemplate::Style        style,
        bool                              grammarOn,
        bool                              alwaysGreedy,
        const Redecoder&                  redecoder,
        std::string_view                  logCtx) {
    const std::string_view opener =
        model::ChatTemplate::toolCallSalvageOpenerText(style);
    if (opener.empty()) {
        return {};   // no salvage opener for this dialect
    }

    // The opener `<function=` is force-prefilled here, so it never reaches the
    // matcher (which only sees generated tokens) — the re-decode is driven by a
    // BODY-ROOTED grammar (assumeOpenerConsumed=true): NAME forced from the
    // first generated token, every required param before the call can close,
    // regardless of the drifted dialect.
    const auto openerIds = tok.encode(std::string{opener}, /*addBos=*/false);
    std::vector<std::int32_t> salvagePrompt = promptIds;
    salvagePrompt.insert(salvagePrompt.end(), openerIds.begin(), openerIds.end());

    runtime::GenerateParams sp = params;
    sp.maxNewTokens = std::min<std::size_t>(sp.maxNewTokens, 1024);

    std::shared_ptr<model::ToolCallConstraint> forceC;
    if (grammarOn) {
        auto c = std::make_shared<model::ToolCallConstraint>(
            tools, tok, /*assumeOpenerConsumed=*/true);
        if (c->active()) { forceC = c; }
    }

    // A GREEDY re-decode just repeats the degeneration that produced the dead
    // call (e.g. query="\n\n"). With a grammar mask active the FORMAT is
    // guaranteed at any temperature, so keep the request's own (anti-loop-
    // lifted) sampling — unless the caller forces greedy (streaming). WITHOUT a
    // mask, greedy is the historical salvage (sampled calls break format when
    // nothing constrains them).
    if (alwaysGreedy || !forceC) {
        sp.sampling.temperature = 0.0F;
    }

    try {
        return redecoder(salvagePrompt, sp, forceC);
    } catch (const std::exception& e) {
        MM_LOG_WARN("server", "{}tool-salvage re-decode failed: {}",
                    logCtx, e.what());
        return {};
    }
}

} // namespace mimirmind::server
