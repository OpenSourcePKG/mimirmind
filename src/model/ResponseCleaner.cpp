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

// A reasoning-block opener found in `s`: the real "<think>" token-as-text, OR a
// HALLUCINATED pseudo special-token "<|WORD_start|>". The checkpoint's real
// reasoning markers are token IDS (<think>/</think>), so ANY literal "<|...|>" in
// the decoded text is a model artifact — when the model reasons without a primed
// think channel (e.g. a non-thinking request nudged into sampling) it invents
// markers like "<|mask_start|>...<|mask_end|>". Treat that leading block as
// reasoning too instead of leaking it into the answer. Generic (any WORD), so
// it is not tied to one checkpoint's hallucinated spelling.
struct ThinkOpen {
    std::size_t pos{std::string::npos};
    std::size_t len{0};
    std::string close;   // the matching closer to look for
    bool        valid{false};
};

ThinkOpen findThinkOpen(const std::string& s) {
    ThinkOpen best;
    if (const std::size_t t = s.find("<think>"); t != std::string::npos) {
        best = {t, 7, "</think>", true};
    }
    std::size_t scan = 0;
    while (true) {
        const std::size_t p = s.find("<|", scan);
        if (p == std::string::npos) { break; }
        const std::size_t gt = s.find("|>", p + 2);
        if (gt == std::string::npos) { break; }   // partial token (held by caller)
        const std::string content = s.substr(p + 2, gt - (p + 2));   // between <| |>
        constexpr std::string_view kSuf{"_start"};
        if (content.size() > kSuf.size() &&
            content.compare(content.size() - kSuf.size(), kSuf.size(), kSuf) == 0) {
            const std::string word = content.substr(0, content.size() - kSuf.size());
            if (!best.valid || p < best.pos) {
                best = {p, (gt + 2) - p, "<|" + word + "_end|>", true};
            }
            break;
        }
        scan = p + 2;
    }
    return best;
}

// Largest emit length that cannot cut a partial opener at the tail: hold back
// from the last '<' that has no '>' after it (a partial "<think"/"<|WORD_start"
// still arriving token by token). Legit prose with a stray '<' is held one step.
std::size_t safeEmitLen(const std::string& s) {
    const std::size_t lt = s.rfind('<');
    if (lt == std::string::npos) { return s.size(); }
    if (s.find('>', lt) != std::string::npos) { return s.size(); }
    return lt;
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
    _pending.append(text);
    std::string emit;    // answer content
    std::string think;   // reasoning content

    // Alternate between a thinking block (InThink, until its closer, streamed out
    // as reasoning) and answer text (Done, passed through, until the next opener).
    // The opener/closer is the real <think>/</think> OR a hallucinated pseudo
    // special-token <|WORD_start|>/<|WORD_end|> (findThinkOpen). A partial opener
    // still arriving at the tail is held back (safeEmitLen) so it is never emitted
    // as answer text.
    for (bool progress = true; progress;) {
        progress = false;

        if (_thinkPhase == ThinkPhase::InThink) {
            const std::size_t close = _pending.find(_thinkClose);
            if (close == std::string::npos) {
                // Stream out the reasoning so far, holding back a tail that could
                // be a split closer prefix.
                if (_pending.size() > _thinkClose.size() - 1) {
                    const std::size_t take =
                        _pending.size() - (_thinkClose.size() - 1);
                    think.append(_pending, 0, take);
                    _pending.erase(0, take);
                }
                break;
            }
            think.append(_pending, 0, close);   // reasoning up to the closer
            _pending.erase(0, close + _thinkClose.size());
            _thinkPhase   = ThinkPhase::Done;
            _thinkClose   = "</think>";   // reset for a later real re-open
            _stripLeading = true;         // drop the newline(s) after the closer
            progress      = true;
            continue;
        }

        // Done: pass answer text through, but drop leftover leading whitespace and
        // route a (real or hallucinated) re-opened reasoning block to reasoning.
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

        const ThinkOpen open = findThinkOpen(_pending);
        if (!open.valid) {
            // No complete opener: emit all but a partial opener still at the tail.
            const std::size_t safe = safeEmitLen(_pending);
            emit.append(_pending, 0, safe);
            _pending.erase(0, safe);
            break;
        }
        emit.append(_pending, 0, open.pos);
        _pending.erase(0, open.pos + open.len);
        _thinkClose = open.close;
        _thinkPhase = ThinkPhase::InThink;
        progress    = true;
    }

    // Defensive scrub: any COMPLETE literal "<|...|>" left in the answer is a
    // stray hallucinated marker (e.g. a closer with no opener) — real special
    // tokens are token ids, never text. Remove them wholesale (partial ones were
    // already held back by safeEmitLen, so anything here is complete).
    for (std::size_t p = emit.find("<|"); p != std::string::npos;
         p = emit.find("<|", p)) {
        const std::size_t gt = emit.find("|>", p + 2);
        if (gt == std::string::npos) { break; }
        emit.erase(p, (gt + 2) - p);
    }

    text      = std::move(emit);
    reasoning = std::move(think);
    return !text.empty();
}

} // namespace mimirmind::model