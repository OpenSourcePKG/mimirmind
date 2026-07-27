// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "model/ChatTemplate.hpp"
#include "model/Tokenizer.hpp"

#include <cstdint>
#include <string>
#include <string_view>

namespace mimirmind::model {

/**
 * Streaming counterpart to ChatTemplate::cleanResponse().
 *
 * cleanResponse() runs once over the full assembled string in the
 * non-streaming /v1/chat/completions path. The SSE path emits text
 * token by token and never assembles a full string before sending,
 * so it needs a stateful filter — that is this class.
 *
 * The only structural markup that reaches the cleaner is Gemma 4's
 * leading channel wrapper:
 *
 *   <|channel>thought\n<channel|>
 *
 * The model emits it at the very start of every response, even when
 * thinking mode is off (then the channel body is empty). OpenAI-style
 * clients should never see it.
 *
 * Stop tokens like <turn|> are handled upstream by the engine's
 * stopIds and never reach `feed`.
 *
 * For Qwen / Gemma 3 the cleaner is a pure pass-through.
 *
 * Construct once per chat turn — state is reset by construction.
 */
class ResponseCleaner {
public:
    /// Build a cleaner for `style`. The IDs are only consulted for
    /// Style::Gemma4 and may safely be -1 if absent in the vocab — the
    /// wrapper strip becomes a no-op in that case.
    ResponseCleaner(ChatTemplate::Style style,
                    std::int32_t        channelStartId,
                    std::int32_t        channelEndId) noexcept;

    /// Convenience: look up the Gemma 4 channel token ids on `tok` and
    /// build the cleaner. For non-Gemma4 styles `tok` is not consulted.
    ///
    /// Inline so call sites only pay the Tokenizer dependency if they
    /// actually use this factory — keeps the test binary linkable
    /// without dragging in Tokenizer.cpp.
    [[nodiscard]] static inline ResponseCleaner
    forStyle(ChatTemplate::Style style, const Tokenizer& tok) {
        if (style == ChatTemplate::Style::Gemma4) {
            constexpr std::string_view kChannelStart{"<|channel>"};
            constexpr std::string_view kChannelEnd  {"<channel|>"};
            return ResponseCleaner{
                style,
                tok.findToken(kChannelStart),
                tok.findToken(kChannelEnd),
            };
        }
        // QwenChatML: qwen35moe pre-opens <think> in the generation prompt
        // (ChatTemplate::encodeQwen), so the response begins INSIDE the thinking
        // block and its content up to the first </think> is reasoning to drop.
        // The strip is string-based (see feed) — the tags arrive as literal
        // text, sometimes split across tokens, so token-id matching is
        // unreliable. Gate it on the model actually having a <think> token:
        // findToken>=0 means thinking is pre-opened; Qwen2/2.5 (no such token,
        // -1) get pure pass-through so a normal answer is never swallowed.
        return ResponseCleaner{style, tok.findToken("<think>"), -1};
    }

    /// Decide whether the decoded text for `tokenId` reaches the client.
    /// May mutate `text` in place — specifically, after the closing
    /// `<channel|>` we strip leading whitespace so the visible content
    /// starts cleanly. Returns true if the (possibly modified) `text`
    /// should be emitted, false if the token is structural and should
    /// be dropped.
    [[nodiscard]] bool feed(std::int32_t tokenId, std::string& text);

private:
    // Streaming strip of the pre-opened Qwen thinking block. InThink: swallow
    // until the first </think> closer; Done: pass-through (also the initial
    // state for non-thinking Qwen2/2.5).
    enum class ThinkPhase { InThink, Done };

    bool feedGemma4(std::int32_t tokenId, std::string& text);
    bool feedQwenThink(std::string& text);

    ChatTemplate::Style _style;
    std::int32_t        _channelStartId;
    std::int32_t        _channelEndId;
    bool                _inChannel    {false};
    bool                _stripLeading {false};
    ThinkPhase          _thinkPhase   {ThinkPhase::Done};
    std::string         _pending;
};

} // namespace mimirmind::model