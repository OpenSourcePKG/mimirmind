// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace mimirmind::runtime::audio {

/**
 * A decoded audio clip: mono float32 PCM in [-1, 1] at a known sample rate.
 * This is the canonical shape the Whisper front-end consumes — the log-mel
 * front-end expects 16 kHz mono, so callers that need Whisper input run the
 * clip through `resampleTo(16000)` (or ask WavReader to do it for them).
 */
struct AudioClip {
    std::vector<float> samples;      // mono f32, [-1, 1]
    int                sampleRate{0};

    [[nodiscard]] std::size_t frames() const noexcept { return samples.size(); }
    [[nodiscard]] double      seconds() const noexcept {
        return sampleRate > 0 ? static_cast<double>(samples.size()) / sampleRate : 0.0;
    }
};

/**
 * Minimal RIFF/WAVE decoder — the first container MimirMind's ASR path
 * accepts. Deliberately narrow (WAV only; mp3/ogg/flac via a small decoder are
 * a later increment), but robust about the WAV variants that matter:
 *
 *   - PCM integer 8/16/24/32-bit (format tag 1)
 *   - IEEE float 32/64-bit (format tag 3)
 *   - WAVE_FORMAT_EXTENSIBLE (tag 0xFFFE) — the real sub-format is read from
 *     the extension GUID's first two bytes (1 = PCM, 3 = float)
 *   - Any channel count — down-mixed to mono by averaging the channels
 *   - Chunk ordering / unknown chunks (LIST, fact, ...) — skipped by walking
 *     the chunk table rather than assuming `data` sits at a fixed offset
 *
 * Everything is normalised to mono f32 in [-1, 1]. Sample rate is preserved as
 * read; use `resampleTo` to reach Whisper's 16 kHz. All errors are reported by
 * throwing std::runtime_error with a message naming the malformed field — the
 * decoder never reads past the buffer it was given.
 */
class WavReader {
public:
    /// Decode WAV bytes (a whole file's contents) into a mono f32 clip at the
    /// file's native sample rate. Throws std::runtime_error on a malformed or
    /// unsupported container.
    [[nodiscard]] static AudioClip decode(std::span<const std::byte> wav);

    /// Read a .wav file from disk and decode it. Throws if the file cannot be
    /// opened or is malformed.
    [[nodiscard]] static AudioClip decodeFile(std::string_view path);

    /// Decode `wav` and resample to `targetRate` (default 16 kHz) mono f32 —
    /// the exact shape the Whisper log-mel front-end expects.
    [[nodiscard]] static AudioClip
    decodeToMono(std::span<const std::byte> wav, int targetRate = 16000);

    /// Linear-interpolation resample of a mono clip to `targetRate`. A no-op
    /// (returns a copy) when the clip already sits at the target rate or is
    /// empty. Linear is deliberately simple — good enough to feed the mel
    /// front-end; a windowed-sinc polyphase resampler is a later quality step.
    [[nodiscard]] static AudioClip resampleTo(const AudioClip& clip, int targetRate);
};

} // namespace mimirmind::runtime::audio
