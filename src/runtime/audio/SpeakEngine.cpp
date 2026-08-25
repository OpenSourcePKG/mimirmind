// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/audio/SpeakEngine.hpp"

#include "core/log/Log.hpp"
#include "runtime/InferenceEngine.hpp"
#include "runtime/audio/WavWriter.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <sstream>
#include <vector>

namespace mimirmind::runtime::audio {

namespace {

// Trim trailing near-silence from the decoded PCM. Orpheus tends to pad
// terse prompts (e.g. "Hello.") with a quiet tail after the utterance —
// the audio-token stream is valid and on-slot, the model just keeps
// emitting low-energy frames before EOS. Only the TAIL is trimmed;
// leading samples are untouched so the first phoneme's onset is never
// clipped. The threshold is relative to the utterance's peak window RMS
// (adapts to overall loudness), and a short margin after the last active
// window preserves the natural decay.
void trimTrailingSilence(std::vector<float>& pcm, std::size_t sampleRate) {
    if (pcm.empty() || sampleRate == 0) {
        return;
    }
    const std::size_t win = std::max<std::size_t>(1, sampleRate / 50); // 20 ms
    auto windowRms = [&](std::size_t start) {
        const std::size_t end = std::min(pcm.size(), start + win);
        double acc = 0.0;
        for (std::size_t i = start; i < end; ++i) {
            acc += static_cast<double>(pcm[i]) * static_cast<double>(pcm[i]);
        }
        const std::size_t n = end - start;
        return n ? std::sqrt(acc / static_cast<double>(n)) : 0.0;
    };
    double peak = 0.0;
    for (std::size_t s = 0; s < pcm.size(); s += win) {
        peak = std::max(peak, windowRms(s));
    }
    if (peak <= 0.0) {
        return;
    }
    const double thresh = 0.03 * peak;   // 3% of peak counts as "silence"
    std::size_t lastActiveEnd = 0;
    for (std::size_t s = 0; s < pcm.size(); s += win) {
        if (windowRms(s) >= thresh) {
            lastActiveEnd = std::min(pcm.size(), s + win);
        }
    }
    if (lastActiveEnd == 0) {
        return;   // all silent — leave untouched rather than empty it
    }
    const std::size_t margin = std::min<std::size_t>(sampleRate / 20,
                                                     pcm.size()); // 50 ms decay
    const std::size_t keep = std::min(pcm.size(), lastActiveEnd + margin);
    if (keep < pcm.size()) {
        pcm.resize(keep);
    }
}

} // namespace

SpeakEngine::SpeakEngine(runtime::InferenceEngine& engine,
                         std::string_view codecPath)
    : _engine(engine) {
    _snac.load(codecPath, /*noise=*/false);
}

std::string SpeakEngine::resolveVoice(std::string_view voice) const {
    const auto& voices = orpheusVoices();
    if (!voice.empty() &&
        std::find(voices.begin(), voices.end(), voice) != voices.end()) {
        return std::string(voice);
    }
    return "tara";
}

std::vector<float>
SpeakEngine::synthesizePcm(std::string_view text, std::string_view voice,
                           std::size_t maxNewTokens) const {
    // Prompt = [start] + BPE("{voice}: {text}") + end[]   (Orpheus 3B format).
    const std::string prompt = resolveVoice(voice) + ": " + std::string(text);

    std::vector<std::int32_t> ids;
    ids.push_back(_ot.startToken);
    const auto body = _engine.tokenizer().encode(prompt, /*addBos=*/false);
    ids.insert(ids.end(), body.begin(), body.end());
    for (const std::int32_t e : _ot.endTokens) ids.push_back(e);

    GenerateParams gp;
    gp.maxNewTokens          = maxNewTokens != 0 ? maxNewTokens : 1200;
    gp.sampling.temperature  = 0.6F;
    gp.sampling.topP         = 0.8F;
    gp.sampling.topK         = 0;
    gp.sampling.seed         = 0;
    // Orpheus REQUIRES a repetition penalty (1.3) — without it the acoustic LM
    // degenerates into repeated audio tokens that fall out of the 7-per-frame
    // slot alignment and get dropped, yielding little/no audio. Apply it over a
    // wide window covering the audio-token history.
    gp.sampling.repetitionPenalty = 1.3F;
    gp.sampling.penaltyWindow     = 1024;
    gp.stopIds               = {_ot.audioEos};

    {
        std::ostringstream os;
        os << "prompt(" << ids.size() << ")=[";
        for (std::size_t i = 0; i < ids.size() && i < 12; ++i) os << ids[i] << ",";
        os << "]";
        MM_LOG_INFO("tts", "speak: {}", os.str());
    }

    const std::vector<std::int32_t> gen = _engine.generate(ids, gp);

    {
        std::ostringstream os;
        os << "gen(" << gen.size() << ") first=[";
        for (std::size_t i = 0; i < gen.size() && i < 24; ++i) os << gen[i] << ",";
        os << "]";
        MM_LOG_INFO("tts", "speak: {}", os.str());
    }

    // Emitted ids -> flat audio codes. Mirror the reference: the position index
    // (hence the 4096-block slot) advances only on a KEPT code; codes must be
    // strictly positive.
    std::vector<std::int32_t> flat;
    flat.reserve(gen.size());
    std::size_t count = 0;
    for (const std::int32_t tid : gen) {
        if (tid == _ot.audioEos) break;
        const std::int32_t code = orpheusTokenToCode(tid, count, _ot);
        if (code > 0) {
            flat.push_back(code);
            ++count;
        }
    }

    MM_LOG_INFO("tts", "speak: {} valid audio codes ({} frames)", flat.size(),
                flat.size() / 7);
    const auto levels = regroupOrpheusFrames(flat);
    if (levels.size() != 3 || levels[2].empty()) {
        return {};
    }
    std::vector<float> pcm = _snac.decode(levels);
    trimTrailingSilence(pcm, sampleRate());
    return pcm;
}

std::vector<std::byte>
SpeakEngine::synthesizeWav(std::string_view text, std::string_view voice,
                           std::size_t maxNewTokens) const {
    const std::vector<float> pcm = synthesizePcm(text, voice, maxNewTokens);
    return WavWriter::encodePcm16(pcm, sampleRate());
}

} // namespace mimirmind::runtime::audio
