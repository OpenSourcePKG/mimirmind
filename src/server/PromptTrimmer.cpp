#include "server/PromptTrimmer.hpp"

#include "model/Tokenizer.hpp"
#include "runtime/Log.hpp"

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
    TrimReport&                      report,
    std::string&                     errorMessage) {
    report.originalPromptTokens = promptIds.size();

    // 1. Message-drop-trim loop.
    for (std::size_t iter = 0; iter < kTrimIterLimit; ++iter) {
        if (promptIds.size() + maxNewTokens + kCapSlack <= maxContextTokens) {
            break;
        }
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
        msgs.erase(msgs.begin() + static_cast<std::ptrdiff_t>(dropIdx));
        ++report.droppedMessages;
        promptIds = model::ChatTemplate::encode(chatStyle, tok, msgs,
                                                /*addGenerationPrompt=*/true);
    }

    // 2. Post-trim: clamp maxNew against remaining budget if still too big.
    const std::size_t Tp = promptIds.size();
    if (Tp + maxNewTokens + kCapSlack > maxContextTokens) {
        if (Tp + kCapSlack >= maxContextTokens) {
            // Prompt alone exhausts the budget even without any completion.
            errorMessage =
                "prompt too long: " + std::to_string(Tp) +
                " tokens after trimming " + std::to_string(report.droppedMessages) +
                " message(s) + slack " + std::to_string(kCapSlack) +
                " does not fit context budget " + std::to_string(maxContextTokens) +
                " — raise runtime.maxContextTokens or shorten the last "
                "user message";
            return false;
        }
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