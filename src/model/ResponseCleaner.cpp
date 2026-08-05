// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "model/ResponseCleaner.hpp"

#include <algorithm>
#include <cstddef>
#include <string_view>
#include <utility>

namespace mimirmind::model {

namespace {

bool isAsciiWs(char c) noexcept {
    return c == '\n' || c == ' ' || c == '\t' || c == '\r';
}

} // namespace

ResponseCleaner::ResponseCleaner(ChatTemplate::Style style,
                                  std::int32_t        channelStartId,
                                  std::int32_t        channelEndId) noexcept
    : _style{style},
      _channelStartId{channelStartId},
      _channelEndId{channelEndId} {
    // Qwen thinking models pre-open <think> in the prompt (signalled by a
    // non-negative <think> token id in channelStartId), so the response begins
    // inside the thinking block — start swallowing. Everything else (Qwen2/2.5,
    // Gemma) starts in pass-through.
    if (_style == ChatTemplate::Style::QwenChatML && _channelStartId >= 0) {
        _thinkPhase = ThinkPhase::InThink;
    }
}

bool ResponseCleaner::feed(std::int32_t tokenId, std::string& text,
                           std::string& reasoning) {
    reasoning.clear();
    switch (_style) {
        case ChatTemplate::Style::Gemma4:
            return feedGemma4(tokenId, text, reasoning);
        case ChatTemplate::Style::QwenChatML:
            return feedQwenThink(text, reasoning);
        default:
            return !text.empty();
    }
}

// Gemma 4: split the <|channel>thought<channel|> wrapper at the token level —
// the channel body is the model's thinking and is surfaced as reasoning.
bool ResponseCleaner::feedGemma4(std::int32_t tokenId, std::string& text,
                                 std::string& reasoning) {
    // Channel-open marker — enter thinking mode. The token id is the
    // single special token <|channel>; the channel body that follows
    // is the reasoning text plus eventually the <channel|> closer.
    if (_channelStartId >= 0 && tokenId == _channelStartId) {
        _inChannel = true;
        return false;
    }

    // Channel-close marker — leave thinking mode and arm a
    // leading-whitespace strip so the first visible content does not
    // start with the "\n" that immediately follows the close.
    if (_inChannel && _channelEndId >= 0 && tokenId == _channelEndId) {
        _inChannel    = false;
        _stripLeading = true;
        return false;
    }

    if (_inChannel) {
        // Channel body = thinking. Surface it as reasoning rather than dropping.
        reasoning = std::move(text);
        text.clear();
        return false;
    }

    if (_stripLeading) {
        std::size_t i = 0;
        while (i < text.size() && isAsciiWs(text[i])) {
            ++i;
        }
        if (i == text.size()) {
            return false;       // whole token was whitespace; keep stripping
        }
        text.erase(0, i);
        _stripLeading = false;
    }

    return !text.empty();
}

// Qwen3 thinking models (qwen35moe) pre-open <think> in the prompt, so the
// response begins inside the thinking block and closes it with the first
// </think> — emitted as literal text, sometimes split across tokens, so a
// string match is used rather than the (unreliable) </think> token id. Swallow
// everything up to and including that closer; then pass through. For a
// non-thinking model the phase starts at Done, so this is a plain pass-through.
bool ResponseCleaner::feedQwenThink(std::string& text, std::string& reasoning) {
    constexpr std::string_view kOpen{"<think>"};
    constexpr std::string_view kClose{"</think>"};

    _pending.append(text);
    std::string emit;    // answer content
    std::string think;   // reasoning content

    // Alternate between a thinking block (InThink, until </think>, streamed out
    // as reasoning) and answer text (Done, passed through, until a spurious
    // re-opened <think>). qwen35moe usually emits one pre-opened block, but some
    // prompts make it re-open a second <think>…</think> mid-answer — the loop
    // handles every such block. <think>/</think> are single tokens, so they
    // arrive whole; the </think> tail-hold below only guards a defensive split.
    for (bool progress = true; progress;) {
        progress = false;

        if (_thinkPhase == ThinkPhase::InThink) {
            const std::size_t close = _pending.find(kClose);
            if (close == std::string::npos) {
                // Stream out the reasoning so far, holding back only a tail that
                // could be a split </think> prefix (so we never emit a partial
                // closer as reasoning).
                if (_pending.size() > kClose.size() - 1) {
                    const std::size_t take =
                        _pending.size() - (kClose.size() - 1);
                    think.append(_pending, 0, take);
                    _pending.erase(0, take);
                }
                break;
            }
            think.append(_pending, 0, close);   // reasoning up to the closer
            _pending.erase(0, close + kClose.size());
            _thinkPhase   = ThinkPhase::Done;
            _stripLeading = true;   // drop the newline(s) right after </think>
            progress      = true;
            continue;
        }

        // Done: pass answer text through, but drop any leading whitespace left
        // over from the closer and swallow a spuriously re-opened <think>.
        if (_stripLeading) {
            std::size_t i = 0;
            while (i < _pending.size() && isAsciiWs(_pending[i])) {
                ++i;
            }
            _pending.erase(0, i);
            if (_pending.empty()) {
                break;              // only whitespace so far; keep stripping
            }
            _stripLeading = false;
        }

        const std::size_t open = _pending.find(kOpen);
        if (open == std::string::npos) {
            emit.append(_pending);
            _pending.clear();
            break;
        }
        emit.append(_pending, 0, open);
        _pending.erase(0, open + kOpen.size());
        _thinkPhase = ThinkPhase::InThink;
        progress    = true;
    }

    text      = std::move(emit);
    reasoning = std::move(think);
    return !text.empty();
}

} // namespace mimirmind::model