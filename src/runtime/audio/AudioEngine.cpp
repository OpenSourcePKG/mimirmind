// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/audio/AudioEngine.hpp"

#include "compute/dsp/MelSpectrogram.hpp"
#include "runtime/audio/WavReader.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mimirmind::runtime::audio {

namespace {

[[noreturn]] void fail(std::string_view msg) {
    std::ostringstream os;
    os << "AudioEngine: " << msg;
    throw std::runtime_error(os.str());
}

// Resolve a Whisper control token from the tokenizer's added-tokens by its
// exact text (e.g. "<|startoftranscript|>"). Throws if the checkpoint's
// tokenizer does not define it.
std::int32_t requireSpecial(const model::Tokenizer& tok, std::string_view text) {
    const std::int32_t id = tok.findToken(text);
    if (id < 0) {
        std::ostringstream os;
        os << "tokenizer is missing required special token '" << text << "'";
        fail(os.str());
    }
    return id;
}

} // namespace

AudioEngine::AudioEngine(std::string_view modelDir,
                         compute::ComputeOps& ops,
                         compute::ComputeMatmul& matmul)
    : _runner{_model, ops, matmul} {
    // BF16 linear weights -> tensor-core GEMM at M>1 (encoder + prefill decode
    // steps); conv/embeddings/norms stay F32. Same rounding tolerance the embed
    // and rerank paths accept.
    _model.load(modelDir, ops, core::gguf::GgmlType::BF16);

    // Whisper's tokenizer is GPT-2 byte-level BPE — exactly what the shared
    // Tokenizer's HF-json path speaks. Resolves detokenization AND the control
    // token ids below, so the decode logic never hard-codes a vocab.
    _tokenizer.loadFromHfJson(std::string{modelDir});

    _special.sot          = requireSpecial(_tokenizer, "<|startoftranscript|>");
    _special.eot          = requireSpecial(_tokenizer, "<|endoftext|>");
    _special.transcribe   = requireSpecial(_tokenizer, "<|transcribe|>");
    _special.translate    = requireSpecial(_tokenizer, "<|translate|>");
    _special.noTimestamps = requireSpecial(_tokenizer, "<|notimestamps|>");
    _special.langEn       = requireSpecial(_tokenizer, "<|en|>");
}

std::int32_t AudioEngine::languageToken(std::string_view language) const {
    const std::string lang = language.empty() ? "en" : std::string{language};
    std::ostringstream tag;
    tag << "<|" << lang << "|>";
    const std::int32_t id = _tokenizer.findToken(tag.str());
    if (id < 0) {
        std::ostringstream os;
        os << "unknown Whisper language code '" << lang << "' (no " << tag.str()
           << " token in the tokenizer)";
        fail(os.str());
    }
    return id;
}

std::string AudioEngine::transcribeMono(std::span<const float> mono,
                                        std::string_view language) const {
    if (mono.empty()) {
        return {};
    }
    // Whisper is trained exclusively on 30-second windows: pad-or-trim the
    // audio to exactly N_SAMPLES = 30 * 16000 = 480000 samples before the mel
    // frontend (openai/whisper pad_or_trim). This yields the fixed 3000 mel
    // frames / 1500 audio-ctx positions the encoder positional embeddings and
    // the decoder were trained on. Without it the encoder sees a shorter
    // positional range than training and greedy decode fails to emit EOT
    // (runaway repetition of the transcript). Windowing audio longer than 30 s
    // into multiple segments is a later increment.
    constexpr std::size_t kWhisperSampleRate    = 16000;
    constexpr std::size_t kWhisperWindowSamples = 30 * kWhisperSampleRate;
    std::vector<float>    window(kWhisperWindowSamples, 0.0F);
    const std::size_t     nCopy = std::min(mono.size(), kWhisperWindowSamples);
    std::copy_n(mono.begin(), nCopy, window.begin());

    compute::dsp::MelConfig melCfg{};
    melCfg.nMels = static_cast<int>(_model.config().numMelBins);
    const compute::dsp::MelSpectrogram mel =
        compute::dsp::logMelSpectrogram(window, melCfg);
    if (mel.nFrames == 0) {
        return {};
    }

    WhisperDecodeOptions opt{};
    opt.special    = _special;
    opt.translate  = false;   // task = transcribe
    opt.timestamps = false;
    // Empty or "auto" language -> let the model detect it (OpenAI Whisper
    // behaviour). Forcing the wrong language (e.g. "en" on German audio)
    // produces hallucinated output. A concrete code forces that language.
    if (language.empty() || language == "auto") {
        opt.detectLanguage = true;
    } else {
        opt.langToken = languageToken(language);
    }

    const std::vector<std::int32_t> ids =
        _runner.transcribeGreedy(mel.data.data(),
                                 static_cast<std::size_t>(mel.nMels),
                                 static_cast<std::size_t>(mel.nFrames), opt);

    // The returned ids include the forced prompt prefix (SOT / lang / task /
    // no-timestamps) and any timestamp/control tokens. Every Whisper control
    // token sits at or above <|endoftext|>; text tokens are strictly below it.
    // Keep the text tokens only, then detokenize.
    std::vector<std::int32_t> textIds;
    textIds.reserve(ids.size());
    for (const std::int32_t id : ids) {
        if (id < _special.eot) {
            textIds.push_back(id);
        }
    }
    return _tokenizer.decode(textIds, /*skipSpecial=*/true);
}

std::string AudioEngine::transcribe(std::span<const std::byte> wav,
                                    std::string_view language) const {
    // WAV -> mono 16 kHz f32 (the log-mel front-end's fixed input rate).
    const AudioClip clip = WavReader::decodeToMono(wav, 16000);
    return transcribeMono(clip.samples, language);
}

std::string AudioEngine::transcribeFile(std::string_view path,
                                        std::string_view language) const {
    const AudioClip mono = WavReader::resampleTo(WavReader::decodeFile(path), 16000);
    return transcribeMono(mono.samples, language);
}

} // namespace mimirmind::runtime::audio
