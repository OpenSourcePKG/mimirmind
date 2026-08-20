// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/audio/OrpheusCodes.hpp"

namespace mimirmind::runtime::audio {

const std::vector<std::string>& orpheusVoices() {
    static const std::vector<std::string> kVoices = {
        "tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"};
    return kVoices;
}

std::int32_t orpheusTokenToCode(std::int32_t tid, std::size_t index,
                                const OrpheusTokens& t) {
    const std::int32_t slot = static_cast<std::int32_t>(index % 7);
    const std::int32_t code = tid - t.codeOffset - slot * t.codebookSize;
    if (code < 0 || code >= t.codebookSize) {
        return -1;
    }
    return code;
}

std::vector<std::vector<std::int32_t>>
regroupOrpheusFrames(const std::vector<std::int32_t>& flat) {
    const std::size_t frames = flat.size() / 7;
    std::vector<std::int32_t> coarse;   // level 0, 1 per frame
    std::vector<std::int32_t> mid;       // level 1, 2 per frame
    std::vector<std::int32_t> fine;      // level 2, 4 per frame
    coarse.reserve(frames);
    mid.reserve(frames * 2);
    fine.reserve(frames * 4);

    for (std::size_t j = 0; j < frames; ++j) {
        const std::size_t i = 7 * j;
        coarse.push_back(flat[i + 0]);
        mid.push_back(flat[i + 1]);
        mid.push_back(flat[i + 4]);
        fine.push_back(flat[i + 2]);
        fine.push_back(flat[i + 3]);
        fine.push_back(flat[i + 5]);
        fine.push_back(flat[i + 6]);
    }
    return {std::move(coarse), std::move(mid), std::move(fine)};
}

} // namespace mimirmind::runtime::audio
