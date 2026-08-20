// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/audio/SpeakEngine.hpp"

#include "core/log/Log.hpp"
#include "runtime/InferenceEngine.hpp"
#include "runtime/audio/WavWriter.hpp"

#include <algorithm>
#include <sstream>

namespace mimirmind::runtime::audio {

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
    return _snac.decode(levels);
}

std::vector<std::byte>
SpeakEngine::synthesizeWav(std::string_view text, std::string_view voice,
                           std::size_t maxNewTokens) const {
    const std::vector<float> pcm = synthesizePcm(text, voice, maxNewTokens);
    return WavWriter::encodePcm16(pcm, sampleRate());
}

} // namespace mimirmind::runtime::audio
