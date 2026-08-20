// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace mimirmind::runtime::audio {

/**
 * Minimal RIFF/WAVE encoder — the inverse of WavReader, the output container
 * for the TTS path (/v1/audio/speech). Deliberately narrow: writes canonical
 * PCM (little-endian, integer) mono or multi-channel WAV, which every client
 * and the OpenAI `response_format:"wav"`/`"pcm"` shapes expect.
 *
 * Input is mono float32 in [-1, 1] (the canonical shape our audio path uses);
 * samples are clamped to [-1, 1] then quantised. The default is 16-bit PCM
 * (format tag 1) — the standard TTS wire format. The header is the standard
 * 44-byte canonical WAVE layout (RIFF / "WAVE" / "fmt " (16) / "data").
 */
class WavWriter {
public:
    /// Encode mono f32 PCM [-1, 1] at `sampleRate` Hz into a complete WAV file
    /// (44-byte header + 16-bit little-endian PCM data). Samples are clamped to
    /// [-1, 1] before quantisation. Returns the full file bytes.
    [[nodiscard]] static std::vector<std::byte>
    encodePcm16(std::span<const float> mono, int sampleRate);

    /// Raw little-endian 16-bit PCM samples (no header) — the OpenAI
    /// `response_format:"pcm"` payload. Same clamping/quantisation as
    /// encodePcm16 but without the RIFF wrapper.
    [[nodiscard]] static std::vector<std::byte>
    encodeRawPcm16(std::span<const float> mono);
};

} // namespace mimirmind::runtime::audio
