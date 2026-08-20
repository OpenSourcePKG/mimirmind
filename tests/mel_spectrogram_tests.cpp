// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

// Pure-CPU sanity tests for the Whisper log-mel front-end
// (compute/dsp/MelSpectrogram). These check internal self-consistency and
// the Whisper recipe's invariants. Byte-level parity against whisper.cpp's
// mel is a separate on-box gate (needs the oracle + fixture WAVs).

#include "TestFramework.hpp"

#include "compute/dsp/MelSpectrogram.hpp"

#include <cmath>
#include <span>
#include <vector>

using namespace mimirmind::compute::dsp;

namespace {

constexpr double kPi = 3.14159265358979323846;

// Slaney mel-center frequency (Hz) of filter m, i.e. its triangle apex.
// Mirrors the edge computation in melFilterbank for cross-checking.
double melCenterHz(const MelConfig& cfg, int m) {
    auto hzToMel = [](double f) {
        const double fSp = 200.0 / 3.0;
        const double minLogHz = 1000.0;
        const double minLogMel = minLogHz / fSp;
        const double logStep = std::log(6.4) / 27.0;
        return (f >= minLogHz) ? minLogMel + std::log(f / minLogHz) / logStep
                               : f / fSp;
    };
    auto melToHz = [](double mel) {
        const double fSp = 200.0 / 3.0;
        const double minLogHz = 1000.0;
        const double minLogMel = minLogHz / fSp;
        const double logStep = std::log(6.4) / 27.0;
        return (mel >= minLogMel) ? minLogHz * std::exp(logStep * (mel - minLogMel))
                                  : fSp * mel;
    };
    const double melMin = hzToMel(0.0);
    const double melMax = hzToMel(cfg.sampleRate / 2.0);
    const double mel = melMin + (melMax - melMin) * (m + 1) / (cfg.nMels + 1);
    return melToHz(mel);
}

} // namespace

TEST(mel_filterbank_shape_and_nonnegative) {
    MelConfig cfg;                       // 16k / 400 / 160 / 80
    const std::vector<float> fb = melFilterbank(cfg);
    EXPECT_EQ(fb.size(), static_cast<std::size_t>(cfg.nMels) * cfg.nBins());
    bool anyPositive = false;
    for (float w : fb) {
        EXPECT_TRUE(w >= 0.0f);
        if (w > 0.0f) {
            anyPositive = true;
        }
    }
    EXPECT_TRUE(anyPositive);
}

TEST(mel_filterbank_every_filter_has_energy) {
    // Each triangular mel filter must have a strictly positive total weight —
    // an empty filter means a mis-placed band edge.
    MelConfig cfg;
    const std::vector<float> fb = melFilterbank(cfg);
    for (int m = 0; m < cfg.nMels; ++m) {
        float sum = 0.0f;
        for (int k = 0; k < cfg.nBins(); ++k) {
            sum += fb[static_cast<std::size_t>(m) * cfg.nBins() + k];
        }
        EXPECT_TRUE(sum > 0.0f);
    }
}

TEST(logmel_silence_is_uniform_floor) {
    // All-zero input: every mel bin hits the 1e-10 floor -> log10 = -10 ->
    // globalMax = -10, floor = -18, so every value = (-10 + 4)/4 = -1.5.
    MelConfig cfg;
    std::vector<float> pcm(16000, 0.0f);   // 1 s of silence
    const MelSpectrogram mel = logMelSpectrogram(std::span<const float>(pcm), cfg);
    EXPECT_EQ(mel.nMels, cfg.nMels);
    EXPECT_EQ(mel.nFrames, 16000 / cfg.hopLength);
    EXPECT_EQ(mel.data.size(), static_cast<std::size_t>(mel.nMels) * mel.nFrames);
    for (float v : mel.data) {
        EXPECT_NEAR(v, -1.5f, 1e-4f);
    }
}

TEST(logmel_frame_count_matches_whisper) {
    MelConfig cfg;
    // 2.0 s -> 32000 samples -> 32000/160 = 200 frames (trailing frame dropped).
    std::vector<float> pcm(32000, 0.0f);
    const MelSpectrogram mel = logMelSpectrogram(std::span<const float>(pcm), cfg);
    EXPECT_EQ(mel.nFrames, 200);
}

TEST(logmel_short_input_is_empty) {
    MelConfig cfg;
    std::vector<float> pcm(100, 0.0f);     // < one hop of 160
    const MelSpectrogram mel = logMelSpectrogram(std::span<const float>(pcm), cfg);
    EXPECT_EQ(mel.nFrames, 0);
    EXPECT_TRUE(mel.data.empty());
}

TEST(logmel_tone_peaks_at_expected_mel_band) {
    // A pure 1 kHz tone should concentrate energy in the mel band whose
    // centre frequency is near 1 kHz. Check the argmax mel of a mid frame.
    MelConfig cfg;
    const int sr = cfg.sampleRate;
    const double freq = 1000.0;
    std::vector<float> pcm(sr);            // 1 s
    for (int n = 0; n < sr; ++n) {
        pcm[static_cast<std::size_t>(n)] =
            static_cast<float>(std::sin(2.0 * kPi * freq * n / sr));
    }
    const MelSpectrogram mel = logMelSpectrogram(std::span<const float>(pcm), cfg);
    EXPECT_TRUE(mel.nFrames > 10);

    const int t = mel.nFrames / 2;
    int argmax = 0;
    float best = -1e30f;
    for (int m = 0; m < mel.nMels; ++m) {
        const float v = mel.data[static_cast<std::size_t>(m) * mel.nFrames + t];
        if (v > best) {
            best = v;
            argmax = m;
        }
    }
    const double centerHz = melCenterHz(cfg, argmax);
    EXPECT_TRUE(centerHz > 800.0);
    EXPECT_TRUE(centerHz < 1250.0);
}

TEST(logmel_all_finite_and_bounded) {
    MelConfig cfg;
    const int sr = cfg.sampleRate;
    std::vector<float> pcm(sr / 2);
    for (int n = 0; n < static_cast<int>(pcm.size()); ++n) {
        pcm[static_cast<std::size_t>(n)] =
            0.3f * static_cast<float>(std::sin(2.0 * kPi * 440.0 * n / sr));
    }
    const MelSpectrogram mel = logMelSpectrogram(std::span<const float>(pcm), cfg);
    for (float v : mel.data) {
        EXPECT_TRUE(std::isfinite(v));
        // After (max(logspec, gmax-8) + 4)/4 the upper bound is (gmax+4)/4;
        // gmax <= ~ log10(power) which stays modest for a unit-ish tone.
        EXPECT_TRUE(v <= 10.0f);
    }
}
