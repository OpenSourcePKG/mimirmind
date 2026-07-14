// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "model/ResponseCleaner.hpp"

#include <cstddef>

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
      _channelEndId{channelEndId} {}

bool ResponseCleaner::feed(std::int32_t tokenId, std::string& text) {
    if (_style != ChatTemplate::Style::Gemma4) {
        return !text.empty();
    }

    // Channel-open marker — enter swallow mode. The token id is the
    // single special token <|channel>; the channel body that follows
    // is regular text plus eventually the <channel|> closer.
    if (_channelStartId >= 0 && tokenId == _channelStartId) {
        _inChannel = true;
        return false;
    }

    // Channel-close marker — leave swallow mode and arm a
    // leading-whitespace strip so the first visible content does not
    // start with the "\n" that immediately follows the close.
    if (_inChannel && _channelEndId >= 0 && tokenId == _channelEndId) {
        _inChannel    = false;
        _stripLeading = true;
        return false;
    }

    if (_inChannel) {
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

} // namespace mimirmind::model