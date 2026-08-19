// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/audio/WhisperDecode.hpp"

#include <stdexcept>

namespace mimirmind::runtime::audio {

std::vector<std::int32_t>
whisperInitialPromptTokens(const WhisperDecodeOptions& opt) {
    std::vector<std::int32_t> t;
    t.reserve(4);
    t.push_back(opt.special.sot);
    t.push_back(opt.langToken);
    t.push_back(opt.translate ? opt.special.translate : opt.special.transcribe);
    if (!opt.timestamps) {
        t.push_back(opt.special.noTimestamps);
    }
    return t;
}

std::int32_t argmaxRow(std::span<const float> logits) {
    if (logits.empty()) {
        throw std::runtime_error("argmaxRow: empty logits");
    }
    std::int32_t best = 0;
    float bestVal = logits[0];
    for (std::size_t i = 1; i < logits.size(); ++i) {
        if (logits[i] > bestVal) {
            bestVal = logits[i];
            best = static_cast<std::int32_t>(i);
        }
    }
    return best;
}

} // namespace mimirmind::runtime::audio
