// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "server/PromptTrimmer.hpp"

#include "model/Tokenizer.hpp"
#include "core/log/Log.hpp"

#include <algorithm>
#include <cstddef>

namespace mimirmind::server {

using nlohmann::json;

bool PromptTrimmer::applyPromptTrim(
    std::vector<model::ChatMessage>& msgs,
    std::vector<std::int32_t>&       promptIds,
    std::size_t&                     maxNewTokens,
    std::size_t                      maxContextTokens,
    std::size_t                      modelContextLength,
    const model::Tokenizer&          tok,
    model::ChatTemplate::Style       chatStyle,
    std::span<const model::ToolSpec> tools,
    std::optional<bool>              enableThinking,
    model::ChatTemplate::ToolFormat  toolFormat,
    TrimReport&                      report,
    std::string&                     errorMessage) {
    report.originalPromptTokens = promptIds.size();

    // 8.19.8: `max_tokens` is a CEILING on output, never a reservation that
    // may eat the prompt (OpenAI / vLLM semantics). Before this fix the drop
    // loop ran against prompt+maxNew, so a routine agent request with
    // max_tokens 64000 (Claude Code's default) silently trimmed a perfectly
    // fitting prompt — tools block, task and tool history — down to a
    // ~37-token remnant and the model hallucinated from it. Message-drop
    // trimming now fires ONLY when the PROMPT ALONE (+ slack + 1 output
    // token) overflows the budget; otherwise maxNew is clamped below.

    // 1. Message-drop-trim loop — prompt-alone overflow only. Each iteration
    //    drops a BATCH of oldest droppable messages sized by a chars/4 token
    //    estimate of the overflow, then re-encodes once — a per-message
    //    re-encode capped the old loop at kTrimIterLimit drops, which turned
    //    any history longer than ~20 messages over budget into a hard 400.
    for (std::size_t iter = 0; iter < kTrimIterLimit; ++iter) {
        if (promptIds.size() + kCapSlack + 1 <= maxContextTokens) {
            break;
        }
        const std::size_t overflow =
            promptIds.size() + kCapSlack + 1 - maxContextTokens;
        std::size_t freedEstimate = 0;
        bool        droppedAny    = false;
        while (freedEstimate < overflow) {
            // Find the LAST user message — must be preserved. Walk backwards.
            std::size_t lastUserIdx = static_cast<std::size_t>(-1);
            for (std::size_t i = msgs.size(); i-- > 0; ) {
                if (msgs[i].role == model::ChatRole::User) {
                    lastUserIdx = i;
                    break;
                }
            }
            // Pick the earliest droppable index: not system, not last-user.
            std::size_t dropIdx = static_cast<std::size_t>(-1);
            for (std::size_t i = 0; i < msgs.size(); ++i) {
                if (msgs[i].role == model::ChatRole::System) continue;
                if (i == lastUserIdx) continue;
                dropIdx = i;
                break;
            }
            if (dropIdx == static_cast<std::size_t>(-1)) {
                // Only system + last-user left — cannot drop more.
                break;
            }
            std::size_t bytes = msgs[dropIdx].content.size();
            for (const auto& call : msgs[dropIdx].toolCalls) {
                bytes += call.name.size() + call.argumentsJson.size();
            }
            freedEstimate += bytes / 4 + 16;  // ~4 chars/token + turn markup
            msgs.erase(msgs.begin() + static_cast<std::ptrdiff_t>(dropIdx));
            ++report.droppedMessages;
            droppedAny = true;
        }
        if (!droppedAny) {
            break;
        }
        // Re-encode with the SAME tools / thinking / tool-format shape as
        // the original prompt — the pre-8.19.8 re-encode dropped the tools
        // block entirely, so a trimmed agent request lost its tools.
        promptIds = model::ChatTemplate::encode(chatStyle, tok, msgs,
                                                /*addGenerationPrompt=*/true,
                                                tools, enableThinking,
                                                toolFormat);
    }

    // 2. Prompt alone still does not fit → 400, never a silent partial
    //    prompt.
    const std::size_t Tp = promptIds.size();
    if (Tp + kCapSlack + 1 > maxContextTokens) {
        errorMessage =
            "prompt too long: " + std::to_string(Tp) +
            " tokens after trimming " + std::to_string(report.droppedMessages) +
            " message(s) + slack " + std::to_string(kCapSlack) +
            " does not fit context budget " + std::to_string(maxContextTokens) +
            " — raise runtime.maxContextTokens or shorten the last "
            "user message";
        return false;
    }

    // 3. Clamp maxNew to the budget the prompt leaves. Output that hits the
    //    clamped ceiling finishes with finish_reason "length" as usual.
    if (Tp + maxNewTokens + kCapSlack > maxContextTokens) {
        const std::size_t newMax = maxContextTokens - Tp - kCapSlack;
        report.maxNewClampedFrom = maxNewTokens;
        report.maxNewClampedTo   = newMax;
        maxNewTokens             = newMax;
    }

    report.effectivePromptTokens = Tp;

    // 3. Model-native context-length warning (RoPE extrapolation zone).
    if (modelContextLength > 0 && Tp > modelContextLength) {
        report.contextExtrapolated   = true;
        report.contextExtrapolatedBy = Tp - modelContextLength;
    }
    return true;
}

void PromptTrimmer::attachTrimHeaders(httplib::Response& res, const TrimReport& r) {
    if (r.droppedMessages > 0) {
        res.set_header("x-mimirmind-dropped-messages",
                       std::to_string(r.droppedMessages));
    }
    if (r.maxNewClampedFrom > 0) {
        res.set_header("x-mimirmind-max-new-clamped",
                       std::to_string(r.maxNewClampedFrom) + "->" +
                       std::to_string(r.maxNewClampedTo));
    }
    if (r.contextExtrapolated) {
        res.set_header("x-mimirmind-context-extrapolated-by",
                       std::to_string(r.contextExtrapolatedBy));
    }
}

void PromptTrimmer::attachTrimUsage(json& usage, const TrimReport& r) {
    if (r.droppedMessages > 0) {
        usage["mimirmind_dropped_messages"]        = r.droppedMessages;
        usage["mimirmind_original_prompt_tokens"]  = r.originalPromptTokens;
    }
    if (r.maxNewClampedFrom > 0) {
        usage["mimirmind_max_new_clamped_from"] = r.maxNewClampedFrom;
        usage["mimirmind_max_new_clamped_to"]   = r.maxNewClampedTo;
    }
    if (r.contextExtrapolated) {
        usage["mimirmind_context_extrapolated_by"] = r.contextExtrapolatedBy;
    }
}

void PromptTrimmer::extendStopIds(const model::Tokenizer&         tok,
                                    const std::vector<std::string>& strings,
                                    std::vector<std::int32_t>&      stopIds) {
    for (const auto& s : strings) {
        if (s.empty()) {
            continue;
        }
        const auto ids = tok.encode(s, /*addBos=*/false);
        if (ids.empty()) {
            continue;
        }
        if (ids.size() > 1) {
            MM_LOG_WARN("server",
                        "stop string '{}' encodes to {} tokens; only the "
                        "first ({}) is used — substring matching is TODO",
                        s, ids.size(), ids[0]);
        }
        if (std::find(stopIds.begin(), stopIds.end(), ids[0]) == stopIds.end()) {
            stopIds.push_back(ids[0]);
        }
    }
}

} // namespace mimirmind::server