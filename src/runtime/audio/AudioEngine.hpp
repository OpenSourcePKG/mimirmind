// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "model/Tokenizer.hpp"
#include "runtime/audio/WhisperDecode.hpp"
#include "runtime/audio/WhisperModel.hpp"
#include "runtime/audio/WhisperRunner.hpp"

#include <cstddef>
#include <span>
#include <string>
#include <string_view>

namespace mimirmind::compute {
class ComputeOps;
class ComputeMatmul;
} // namespace mimirmind::compute

namespace mimirmind::runtime::audio {

/**
 * A loaded Whisper-class ASR model wrapped as a single serving unit behind
 * /v1/audio/transcriptions — the audio-modality sibling of EmbedEngine /
 * RerankEngine. Given audio bytes it returns transcribed text.
 *
 * Owns everything the transcription cascade needs: the Whisper weights
 * (WhisperModel), a GPT-2 byte-level BPE tokenizer loaded from the checkpoint's
 * HF `tokenizer.json` (Whisper's tokenizer is exactly the byte-level BPE the
 * shared Tokenizer already speaks), and the WhisperRunner forward over the
 * shared ComputeOps / ComputeMatmul. The log-mel front-end is host-side and
 * stateless, so it is invoked per call rather than stored.
 *
 * Pipeline (WAV bytes -> text):
 *   WavReader.decodeToMono(16k) -> log-mel [nMels x nFrames]
 *   -> WhisperRunner.transcribeGreedy -> token ids
 *   -> strip special tokens -> Tokenizer.decode -> text.
 *
 * Special-token ids (SOT / language / task / no-timestamps / EOT) are resolved
 * from the tokenizer's added-tokens at load time, not hard-coded — the decode
 * logic stays checkpoint-independent. Not thread-safe: the caller serialises
 * calls per engine (same contract as EmbedEngine).
 */
class AudioEngine {
public:
    /// `modelDir` holds config.json, the Whisper safetensors (single or
    /// sharded) and tokenizer.json. Loads everything up front. Throws on a
    /// non-Whisper config, a missing/malformed file, or an unresolvable
    /// special token.
    AudioEngine(std::string_view modelDir,
                compute::ComputeOps& ops,
                compute::ComputeMatmul& matmul);

    /// Transcribe WAV bytes (a whole file's contents). `language` is a Whisper
    /// language code ("en", "de", ...); empty forces English. Greedy, no
    /// timestamps. Returns the detokenized text (special tokens stripped).
    [[nodiscard]] std::string
    transcribe(std::span<const std::byte> wav, std::string_view language = "en") const;

    /// Convenience: read a .wav file from disk and transcribe it.
    [[nodiscard]] std::string
    transcribeFile(std::string_view path, std::string_view language = "en") const;

    [[nodiscard]] std::size_t numMelBins() const noexcept {
        return _model.config().numMelBins;
    }

private:
    /// Resolve a Whisper language code ("en") to its language token id via the
    /// tokenizer's "<|en|>" added-token. Throws if absent.
    [[nodiscard]] std::int32_t languageToken(std::string_view language) const;

    /// Shared tail: mel front-end -> greedy decode -> strip control tokens ->
    /// detokenize. `mono` must already be 16 kHz mono f32.
    [[nodiscard]] std::string
    transcribeMono(std::span<const float> mono, std::string_view language) const;

    WhisperModel          _model;
    model::Tokenizer      _tokenizer;
    WhisperSpecialTokens  _special{};
    mutable WhisperRunner _runner;
};

} // namespace mimirmind::runtime::audio
