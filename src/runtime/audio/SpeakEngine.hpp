// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "runtime/audio/OrpheusCodes.hpp"
#include "runtime/audio/SnacDecoder.hpp"

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace mimirmind::runtime {
class InferenceEngine;
}

namespace mimirmind::runtime::audio {

/**
 * Text-to-speech engine for the Orpheus pipeline (Llama-3.2 acoustic LM + SNAC
 * 24 kHz codec), behind /v1/audio/speech (roadmap 8.13.2). The analogue of the
 * STT AudioEngine: it ties the reusable pieces together —
 *
 *   text (+voice) -> Orpheus prompt -> InferenceEngine.generate (Llama-3.2,
 *   sampled) -> audio token ids -> OrpheusCodes (de-offset + 7-per-frame
 *   regroup) -> SnacDecoder -> 24 kHz PCM -> WavWriter -> WAV bytes.
 *
 * The acoustic model is a STANDARD Llama, so it runs on the existing
 * InferenceEngine unchanged (no net-new acoustic code) on either backend; the
 * only codec-specific work lives in SnacDecoder / OrpheusCodes. The engine is
 * borrowed (non-owning) and kept alive by ServeMode for the process lifetime;
 * SpeakEngine owns only the (small, host-side) SNAC decoder weights. Not
 * thread-safe — one synthesize() at a time (InferenceEngine has one KV cache).
 */
class SpeakEngine {
public:
    /// `engine` must already have an Orpheus (Llama-3.2) checkpoint loaded.
    /// `codecPath` is the converted SNAC-24kHz safetensors (see
    /// scripts/convert-snac.py). Throws if the codec fails to load.
    SpeakEngine(runtime::InferenceEngine& engine, std::string_view codecPath);

    /// Synthesize `text` in `voice` (one of orpheusVoices(); empty/unknown ->
    /// "tara") to 24 kHz mono f32 PCM in [-1, 1]. Empty if the model emits no
    /// audio. `maxNewTokens` bounds the acoustic decode (0 => default 1200).
    [[nodiscard]] std::vector<float>
    synthesizePcm(std::string_view text, std::string_view voice,
                  std::size_t maxNewTokens = 0) const;

    /// Same, encoded as a complete 24 kHz PCM16 WAV file.
    [[nodiscard]] std::vector<std::byte>
    synthesizeWav(std::string_view text, std::string_view voice,
                  std::size_t maxNewTokens = 0) const;

    [[nodiscard]] int sampleRate() const noexcept { return _snac.sampleRate(); }

private:
    [[nodiscard]] std::string resolveVoice(std::string_view voice) const;

    runtime::InferenceEngine& _engine;
    SnacDecoder               _snac;
    OrpheusTokens             _ot;
};

} // namespace mimirmind::runtime::audio
