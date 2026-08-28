// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "model/ToolCallStreamDetector.hpp"

namespace mimirmind::model {

namespace {

constexpr std::string_view kQwenOpen   = "<tool_call>";
constexpr std::string_view kQwenClose  = "</tool_call>";
constexpr std::string_view kGemmaOpen  = "<|tool_call>";
constexpr std::string_view kGemmaClose = "<tool_call|>";

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

    if (_state == State::PassThrough) {
        _pending.append(text);
        text.clear();

        const std::size_t open = _pending.find(_open);
        if (open == std::string::npos) {
            // No opener yet. Hold back a tail that could be a split opener
            // prefix; emit the rest as normal content.
            const std::size_t holdBack = _open.size() - 1;
            if (_pending.size() > holdBack) {
                const std::size_t take = _pending.size() - holdBack;
                text.append(_pending, 0, take);
                _pending.erase(0, take);
            }
            return false;
        }

        // Found an opener: content before it is normal text, flush it back;
        // the opener onward starts the tool-call buffer.
        text.append(_pending, 0, open);
        _buffer.assign(_pending, open, _pending.size() - open);
        _pending.clear();
        _state = State::Buffering;
    } else {
        _buffer.append(text);
        text.clear();
    }

    const std::size_t close = _buffer.find(_close);
    if (close == std::string::npos) {
        return false;
    }

    // Content after the closer within the same chunk (rare — closers are
    // effectively always their own token) is not part of this call.
    const std::size_t afterClose = close + _close.size();
    if (afterClose < _buffer.size()) {
        _trailingAfterClose = _buffer.substr(afterClose);
        _buffer.erase(afterClose);
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
    _buffer.clear();
    if (!_trailingAfterClose.empty()) {
        _pending = std::move(_trailingAfterClose);
        _trailingAfterClose.clear();
    }
}

std::string ToolCallStreamDetector::takePendingFlush() {
    std::string out = std::move(_pending);
    _pending.clear();
    return out;
}

} // namespace mimirmind::model
