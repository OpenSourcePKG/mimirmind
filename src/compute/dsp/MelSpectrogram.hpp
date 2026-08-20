// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace mimirmind::compute::dsp {

/**
 * Parameters of the Whisper log-mel front-end.
 *
 * Defaults match the Whisper tiny/base/small/medium checkpoints
 * (n_mels = 80). Whisper large-v3 uses n_mels = 128 — set nMels
 * accordingly; everything else is shared across the family.
 */
struct MelConfig {
    int sampleRate = 16000;
    int nFft       = 400;
    int hopLength  = 160;
    int nMels      = 80;

    // Number of one-sided FFT bins (DC .. Nyquist inclusive) = nFft/2 + 1.
    // 201 for nFft = 400.
    int nBins() const { return nFft / 2 + 1; }
};

/**
 * Result of logMelSpectrogram: a row-major [nMels x nFrames] matrix
 * (mel-major — row m holds mel band m across all time frames) plus its
 * shape. Element (m, t) lives at data[m * nFrames + t].
 */
struct MelSpectrogram {
    std::vector<float> data;
    int nMels   = 0;
    int nFrames = 0;
};

/**
 * Slaney mel filterbank — byte-for-byte the librosa recipe Whisper uses:
 *
 *   librosa.filters.mel(sr, n_fft, n_mels, fmin=0, fmax=sr/2,
 *                       htk=False, norm='slaney')
 *
 * Triangular filters on the Slaney (HTK=false) mel scale, area-normalised
 * so each filter integrates to a constant regardless of its bandwidth
 * (the "slaney" norm: weight *= 2 / (hz[m+2] - hz[m])).
 *
 * Returns [nMels x nBins] row-major, nBins = nFft/2 + 1.
 */
std::vector<float> melFilterbank(const MelConfig& cfg);

/**
 * Whisper log-mel spectrogram of mono f32 PCM sampled at cfg.sampleRate.
 *
 * Pipeline (matches openai/whisper audio.log_mel_spectrogram exactly):
 *   1. Reflect-pad the signal by nFft/2 on each side (torch.stft
 *      center=True semantics).
 *   2. Framed STFT: hop = hopLength, window = periodic Hann of length nFft,
 *      power spectrum |X|^2 over the nBins one-sided bins.
 *   3. Drop the trailing STFT frame (Whisper's stft[..., :-1]) so
 *      nFrames = nSamples / hopLength.
 *   4. Apply the Slaney mel filterbank -> [nMels x nFrames].
 *   5. log10(clamp(., 1e-10)); floor at (globalMax - 8); (. + 4) / 4.
 *
 * The STFT here is a direct DFT (host reference, correctness first). An
 * FFT / GPU port is a separate optimisation increment — the front-end is
 * tiny next to the encoder.
 *
 * Returns [nMels x nFrames]; an empty result if pcm is shorter than one hop.
 */
MelSpectrogram logMelSpectrogram(std::span<const float> pcm, const MelConfig& cfg);

} // namespace mimirmind::compute::dsp
