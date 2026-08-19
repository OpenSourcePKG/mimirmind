// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "compute/dsp/MelSpectrogram.hpp"

#include <algorithm>
#include <cmath>

namespace mimirmind::compute::dsp {

namespace {

constexpr double kPi = 3.14159265358979323846;

// Slaney (HTK=false) mel scale, matching librosa.hz_to_mel / mel_to_hz.
// Linear below 1 kHz, logarithmic above, joined continuously.
double hzToMel(double f) {
    const double fSp        = 200.0 / 3.0;          // mels per Hz (linear region)
    const double minLogHz   = 1000.0;
    const double minLogMel  = minLogHz / fSp;       // 15.0
    const double logStep    = std::log(6.4) / 27.0;
    if (f >= minLogHz) {
        return minLogMel + std::log(f / minLogHz) / logStep;
    }
    return f / fSp;
}

double melToHz(double mel) {
    const double fSp        = 200.0 / 3.0;
    const double minLogHz   = 1000.0;
    const double minLogMel  = minLogHz / fSp;       // 15.0
    const double logStep    = std::log(6.4) / 27.0;
    if (mel >= minLogMel) {
        return minLogHz * std::exp(logStep * (mel - minLogMel));
    }
    return fSp * mel;
}

// numpy/torch 'reflect' padding index map: mirror around the edges,
// excluding the edge samples themselves. n must be >= 1.
int reflectIndex(int i, int n) {
    if (n == 1) {
        return 0;
    }
    const int period = 2 * (n - 1);
    int m = i % period;
    if (m < 0) {
        m += period;
    }
    return (m < n) ? m : (period - m);
}

} // namespace

std::vector<float> melFilterbank(const MelConfig& cfg) {
    const int    nBins = cfg.nBins();
    const double fMax  = cfg.sampleRate / 2.0;

    // nMels + 2 band edges, evenly spaced on the mel scale, back in Hz.
    std::vector<double> edgeHz(static_cast<std::size_t>(cfg.nMels) + 2);
    const double melMin = hzToMel(0.0);
    const double melMax = hzToMel(fMax);
    for (int i = 0; i < cfg.nMels + 2; ++i) {
        const double mel = melMin + (melMax - melMin) * i / (cfg.nMels + 1);
        edgeHz[static_cast<std::size_t>(i)] = melToHz(mel);
    }

    // One-sided FFT bin centre frequencies: linspace(0, sr/2, nBins).
    std::vector<double> binHz(static_cast<std::size_t>(nBins));
    for (int k = 0; k < nBins; ++k) {
        binHz[static_cast<std::size_t>(k)] = fMax * k / (nBins - 1);
    }

    std::vector<float> weights(static_cast<std::size_t>(cfg.nMels) * nBins, 0.0f);
    for (int m = 0; m < cfg.nMels; ++m) {
        const double lo    = edgeHz[static_cast<std::size_t>(m)];
        const double ce    = edgeHz[static_cast<std::size_t>(m) + 1];
        const double hi    = edgeHz[static_cast<std::size_t>(m) + 2];
        const double enorm = 2.0 / (hi - lo);       // Slaney area normalisation
        for (int k = 0; k < nBins; ++k) {
            const double f     = binHz[static_cast<std::size_t>(k)];
            const double lower = (f - lo) / (ce - lo);
            const double upper = (hi - f) / (hi - ce);
            double w = std::min(lower, upper);
            if (w < 0.0) {
                w = 0.0;
            }
            weights[static_cast<std::size_t>(m) * nBins + k] =
                static_cast<float>(w * enorm);
        }
    }
    return weights;
}

MelSpectrogram logMelSpectrogram(std::span<const float> pcm, const MelConfig& cfg) {
    MelSpectrogram out;
    out.nMels = cfg.nMels;

    const int nSamples = static_cast<int>(pcm.size());
    const int hop      = cfg.hopLength;
    const int nFft     = cfg.nFft;
    const int nBins    = cfg.nBins();
    const int pad      = nFft / 2;

    // torch.stft(center=True) yields 1 + nSamples/hop frames; Whisper drops
    // the trailing one (stft[..., :-1]) -> nSamples/hop frames.
    const int nFrames = nSamples / hop;
    out.nFrames = nFrames;
    if (nFrames <= 0) {
        return out;
    }

    // Periodic Hann window (torch default), length nFft.
    std::vector<float> window(static_cast<std::size_t>(nFft));
    for (int n = 0; n < nFft; ++n) {
        window[static_cast<std::size_t>(n)] =
            static_cast<float>(0.5 - 0.5 * std::cos(2.0 * kPi * n / nFft));
    }

    // Precompute DFT twiddle tables [nBins x nFft] so the per-frame inner
    // loop is trig-free. Direct DFT because nFft = 400 is not a power of two;
    // an FFT is a later optimisation increment.
    std::vector<float> cosT(static_cast<std::size_t>(nBins) * nFft);
    std::vector<float> sinT(static_cast<std::size_t>(nBins) * nFft);
    for (int k = 0; k < nBins; ++k) {
        for (int n = 0; n < nFft; ++n) {
            const double ang = 2.0 * kPi * k * n / nFft;
            cosT[static_cast<std::size_t>(k) * nFft + n] = static_cast<float>(std::cos(ang));
            sinT[static_cast<std::size_t>(k) * nFft + n] = static_cast<float>(std::sin(ang));
        }
    }

    const std::vector<float> filterbank = melFilterbank(cfg);
    const float* filters = filterbank.data();

    out.data.assign(static_cast<std::size_t>(cfg.nMels) * nFrames, 0.0f);

    std::vector<float> frame(static_cast<std::size_t>(nFft));
    std::vector<float> power(static_cast<std::size_t>(nBins));

    for (int t = 0; t < nFrames; ++t) {
        // Windowed frame; padded index (t*hop + j) maps to original index
        // (t*hop + j - pad) via reflect padding.
        const int base = t * hop - pad;
        for (int j = 0; j < nFft; ++j) {
            const int src = reflectIndex(base + j, nSamples);
            frame[static_cast<std::size_t>(j)] =
                pcm[static_cast<std::size_t>(src)] * window[static_cast<std::size_t>(j)];
        }

        // One-sided power spectrum |X_k|^2.
        for (int k = 0; k < nBins; ++k) {
            const float* c = &cosT[static_cast<std::size_t>(k) * nFft];
            const float* s = &sinT[static_cast<std::size_t>(k) * nFft];
            float re = 0.0f;
            float im = 0.0f;
            for (int n = 0; n < nFft; ++n) {
                const float x = frame[static_cast<std::size_t>(n)];
                re += x * c[n];
                im -= x * s[n];
            }
            power[static_cast<std::size_t>(k)] = re * re + im * im;
        }

        // Mel projection into column t.
        for (int m = 0; m < cfg.nMels; ++m) {
            const float* w = &filters[static_cast<std::size_t>(m) * nBins];
            float acc = 0.0f;
            for (int k = 0; k < nBins; ++k) {
                acc += w[k] * power[static_cast<std::size_t>(k)];
            }
            out.data[static_cast<std::size_t>(m) * nFrames + t] = acc;
        }
    }

    // log10 with a 1e-10 floor, then Whisper's dynamic-range compression:
    // floor at (globalMax - 8 dB-ish) and rescale to roughly [-1, +...].
    float globalMax = -1e30f;
    for (float& v : out.data) {
        v = std::log10(std::max(v, 1e-10f));
        globalMax = std::max(globalMax, v);
    }
    const float floor = globalMax - 8.0f;
    for (float& v : out.data) {
        v = (std::max(v, floor) + 4.0f) / 4.0f;
    }

    return out;
}

} // namespace mimirmind::compute::dsp
