// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "model/ToolCallStreamDetector.hpp"

#include <algorithm>

namespace mimirmind::model {

namespace {

constexpr std::string_view kQwenOpen   = "<tool_call>";
constexpr std::string_view kQwenClose  = "</tool_call>";
constexpr std::string_view kGemmaOpen  = "<|tool_call>";
constexpr std::string_view kGemmaClose = "<tool_call|>";
constexpr std::string_view kBareOpen   = "<function=";
constexpr std::string_view kBareClose  = "</function>";
constexpr std::string_view kFence      = "```";

std::string_view openMarkerFor(ChatTemplate::Style style) noexcept {
    switch (style) {
        case ChatTemplate::Style::QwenChatML: return kQwenOpen;
        case ChatTemplate::Style::Gemma4:     return kGemmaOpen;
        default:                              return {};
    }
}

std::string_view closeMarkerFor(ChatTemplate::Style style) noexcept {
    switch (style) {
        case ChatTemplate::Style::QwenChatML: return kQwenClose;
        case ChatTemplate::Style::Gemma4:     return kGemmaClose;
        default:                              return {};
    }
}

} // namespace

ToolCallStreamDetector::ToolCallStreamDetector(ChatTemplate::Style style,
                                                bool                enabled,
                                                std::string         forcedOpener)
    : _open{openMarkerFor(style)},
      _close{closeMarkerFor(style)},
      _openBare{style == ChatTemplate::Style::QwenChatML ? kBareOpen
                                                         : std::string_view{}},
      _closeBare{style == ChatTemplate::Style::QwenChatML ? kBareClose
                                                          : std::string_view{}},
      _active{enabled && !_open.empty()} {
    if (_active && !forcedOpener.empty()) {
        // The opener lives in the prompt (tool_choice:"required"), never in
        // the generated span — start already inside the call so the first
        // generated tokens (the call's body) are buffered from token one
        // instead of leaking as plain content while we wait for an opener
        // marker that will never arrive.
        _buffer = std::move(forcedOpener);
        _state  = State::Buffering;
    }
}

bool ToolCallStreamDetector::feed(std::string& text) {
    if (!_active || _state == State::Completed) {
        return false;
    }

    // Best-effort ``` fence parity over content that leaves the detector as
    // plain text — gates the bare opener so fenced documentation showing a
    // `<function=…>` example is not captured as a call.
    const auto updateFence = [this](std::string_view flushed) {
        std::size_t p = 0;
        while ((p = flushed.find(kFence, p)) != std::string_view::npos) {
            _inFence = !_inFence;
            p += kFence.size();
        }
    };

    if (_state == State::PassThrough) {
        _pending.append(text);
        text.clear();

        // 8.19.9: right after a completed BARE block the model usually emits
        // the dangling `</tool_call>` of the envelope it forgot to open —
        // swallow exactly one, whitespace-tolerant, then return to normal.
        if (_swallowDangling) {
            std::size_t ws = 0;
            while (ws < _pending.size() &&
                   (_pending[ws] == '\n' || _pending[ws] == ' ' ||
                    _pending[ws] == '\t' || _pending[ws] == '\r')) {
                ++ws;
            }
            const std::string_view rest =
                std::string_view{_pending}.substr(ws);
            if (rest.size() < kQwenClose.size()) {
                if (kQwenClose.substr(0, rest.size()) == rest) {
                    // Could still become the dangling closer — hold.
                    return false;
                }
                _swallowDangling = false;
            } else if (rest.substr(0, kQwenClose.size()) == kQwenClose) {
                _pending.erase(0, ws + kQwenClose.size());
                _swallowDangling = false;
            } else {
                _swallowDangling = false;
            }
        }

        const std::size_t open = _pending.find(_open);
        // Bare opener: only occurrences OUTSIDE a fence count. Parity at a
        // position = carried _inFence + toggles for fences earlier in the
        // held-back text (a fence may open inside _pending itself).
        std::size_t openBare = std::string::npos;
        if (!_openBare.empty()) {
            const auto fencedAt = [this](std::size_t pos) {
                bool        f = _inFence;
                std::size_t p = 0;
                while (p < pos) {
                    p = _pending.find(kFence, p);
                    if (p == std::string::npos || p >= pos) {
                        break;
                    }
                    f = !f;
                    p += kFence.size();
                }
                return f;
            };
            std::size_t q = _pending.find(_openBare);
            while (q != std::string::npos) {
                if (!fencedAt(q)) {
                    openBare = q;
                    break;
                }
                q = _pending.find(_openBare, q + 1);
            }
        }
        const std::size_t hit = std::min(open, openBare);
        if (hit == std::string::npos) {
            // No opener yet. Hold back a tail that could be a split opener
            // prefix; emit the rest as normal content.
            const std::size_t holdBack =
                std::max(_open.size(), _openBare.size()) - 1;
            if (_pending.size() > holdBack) {
                const std::size_t take = _pending.size() - holdBack;
                text.append(_pending, 0, take);
                updateFence(text);
                _pending.erase(0, take);
            }
            return false;
        }

        // Found an opener: content before it is normal text, flush it back;
        // the opener onward starts the tool-call buffer.
        _bare = (openBare == hit && open != hit);
        text.append(_pending, 0, hit);
        updateFence(text);
        _buffer.assign(_pending, hit, _pending.size() - hit);
        _pending.clear();
        _state = State::Buffering;
    } else {
        _buffer.append(text);
        text.clear();
    }

    const std::string_view closeMarker = _bare ? _closeBare : _close;
    const std::size_t close = _buffer.find(closeMarker);
    if (close == std::string::npos) {
        return false;
    }

    // Content after the closer within the same chunk (rare — closers are
    // effectively always their own token) is not part of this call.
    const std::size_t afterClose = close + closeMarker.size();
    if (afterClose < _buffer.size()) {
        _trailingAfterClose = _buffer.substr(afterClose);
        _buffer.erase(afterClose);
    }
    if (_bare) {
        _swallowDangling = true;
    }
    _state = State::Completed;
    return true;
}

void ToolCallStreamDetector::closeOnStop() noexcept {
    if (!_active || _state != State::Buffering) {
        return;
    }
    _state = State::Completed;
}

void ToolCallStreamDetector::reset() noexcept {
    _state = State::PassThrough;
    _bare  = false;
    _buffer.clear();
    if (!_trailingAfterClose.empty()) {
        _pending = std::move(_trailingAfterClose);
        _trailingAfterClose.clear();
    }
}

std::string ToolCallStreamDetector::takePendingFlush() {
    // End-of-stream with the dangling `</tool_call>` (or a split prefix of
    // it) still held: drop it here — it belongs to the bare block's missing
    // envelope, not to the visible answer.
    if (_swallowDangling && !_pending.empty()) {
        std::size_t ws = 0;
        while (ws < _pending.size() &&
               (_pending[ws] == '\n' || _pending[ws] == ' ' ||
                _pending[ws] == '\t' || _pending[ws] == '\r')) {
            ++ws;
        }
        const std::string_view rest = std::string_view{_pending}.substr(ws);
        const std::size_t n = std::min(rest.size(), kQwenClose.size());
        if (n > 0 && rest.substr(0, n) == kQwenClose.substr(0, n)) {
            _pending.erase(0, ws + n);
        }
        _swallowDangling = false;
    }
    std::string out = std::move(_pending);
    _pending.clear();
    return out;
}

} // namespace mimirmind::model
