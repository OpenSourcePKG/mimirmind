#pragma once

#include "model/ChatTemplate.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mimirmind::model { class Tokenizer; }

namespace mimirmind::server {

/// M-PT — server-side length discipline.
///
/// Iteratively drops the oldest non-system, non-last-user message and
/// clamps `max_new_tokens` until the request fits the KV cache budget,
/// then reports what was done so callers can attach x-mimirmind-* HTTP
/// headers or extend the `usage` JSON block. See the M-PT Synaipse
/// note for the design rationale.
struct TrimReport {
    std::size_t droppedMessages{0};
    std::size_t originalPromptTokens{0};
    std::size_t effectivePromptTokens{0};
    std::size_t maxNewClampedFrom{0};   // 0 = no clamp
    std::size_t maxNewClampedTo{0};
    bool        contextExtrapolated{false};
    std::size_t contextExtrapolatedBy{0};

    [[nodiscard]] bool fired() const noexcept {
        return droppedMessages > 0
            || maxNewClampedFrom > 0
            || contextExtrapolated;
    }
};

/// Static-only helper class grouping the M-PT primitives.
struct PromptTrimmer {
    /// Slack — matches InferenceEngine::ensureCapacity's `+ 4`. If those
    /// two ever drift, ensureCapacity will start throwing again with a
    /// request the trim helper thought was safe.
    static constexpr std::size_t kCapSlack   = 4;
    static constexpr std::size_t kTrimIterLimit = 20;

    /// Drop oldest droppable messages + clamp `maxNewTokens` until the
    /// request fits `maxContextTokens`. Returns false only if the prompt
    /// alone (after all drops) already exceeds the budget — caller maps
    /// that to a 400 with `errorMessage`.
    [[nodiscard]] static bool applyPromptTrim(
        std::vector<model::ChatMessage>& msgs,
        std::vector<std::int32_t>&       promptIds,
        std::size_t&                     maxNewTokens,
        std::size_t                      maxContextTokens,
        std::size_t                      modelContextLength,
        const model::Tokenizer&          tok,
        model::ChatTemplate::Style       chatStyle,
        TrimReport&                      report,
        std::string&                     errorMessage);

    /// Attach the M-PT report as `x-mimirmind-*` HTTP headers. Only
    /// headers for fields that actually fired are set — clients that
    /// never trigger the trim path see zero overhead.
    static void attachTrimHeaders(httplib::Response& res, const TrimReport& r);

    /// Extend a chat-completion `usage` JSON block with `mimirmind_*`
    /// fields. Vanilla OpenAI clients ignore unknown keys.
    static void attachTrimUsage(nlohmann::json& usage, const TrimReport& r);

    /// Extend the engine's stopIds with the first token of each
    /// user-supplied stop string. Multi-token stops emit a warning —
    /// robust substring matching belongs in a later iteration.
    static void extendStopIds(const model::Tokenizer&         tok,
                              const std::vector<std::string>& strings,
                              std::vector<std::int32_t>&      stopIds);
};

} // namespace mimirmind::server