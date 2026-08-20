// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

// Pure-CPU tests for the host-side Whisper forward pieces: the conv stem, the
// forced-prompt token sequence, and greedy argmax. The device forward
// (WhisperRunner) needs a backend + a real checkpoint and is a separate on-box
// parity gate.

#include "TestFramework.hpp"

#include "runtime/audio/WhisperConvStem.hpp"
#include "runtime/audio/WhisperDecode.hpp"

#include <cmath>
#include <vector>

using namespace mimirmind::runtime::audio;

namespace {

float geluRef(float x) {
    return 0.5F * x * (1.0F + std::erf(x * 0.7071067811865476F));
}

} // namespace

TEST(convstem_output_shape) {
    // nFrames = 3000 -> nCtx = (3000 - 1) / 2 + 1 = 1500 (Whisper's 30s window).
    const std::size_t nMels = 4, nFrames = 3000, d = 2;
    std::vector<float> mel(nMels * nFrames, 0.1F);
    std::vector<float> c1w(d * nMels * 3, 0.01F), c1b(d, 0.0F);
    std::vector<float> c2w(d * d * 3, 0.01F), c2b(d, 0.0F);
    const ConvStemOutput s = whisperConvStem(mel.data(), nMels, nFrames, d,
                                             c1w.data(), c1b.data(),
                                             c2w.data(), c2b.data());
    EXPECT_EQ(s.nCtx, static_cast<std::size_t>(1500));
    EXPECT_EQ(s.dModel, static_cast<std::size_t>(2));
    EXPECT_EQ(s.data.size(), static_cast<std::size_t>(1500 * 2));
}

TEST(convstem_single_channel_numeric) {
    // dModel = nMels = 1, nFrames = 4: hand-check conv indexing/stride/transpose.
    const std::size_t nMels = 1, nFrames = 4, d = 1;
    const std::vector<float> mel = {1.0F, 2.0F, 3.0F, 4.0F};
    const float w0 = 0.5F, w1 = -0.25F, w2 = 0.75F, b1 = 0.1F;   // conv1 k=3
    const float u0 = 1.0F, u1 = 0.5F, u2 = -0.5F, b2 = 0.0F;     // conv2 k=3
    const std::vector<float> c1w = {w0, w1, w2};
    const std::vector<float> c1b = {b1};
    const std::vector<float> c2w = {u0, u1, u2};
    const std::vector<float> c2b = {b2};

    // Reference: conv1 (stride 1, pad 1) + gelu -> out1[4].
    auto in1 = [&](long i) -> float {
        return (i >= 0 && i < 4) ? mel[static_cast<std::size_t>(i)] : 0.0F;
    };
    float out1[4];
    for (long t = 0; t < 4; ++t) {
        out1[t] = geluRef(b1 + w0 * in1(t - 1) + w1 * in1(t) + w2 * in1(t + 1));
    }
    // conv2 (stride 2, pad 1) + gelu -> out2[2], nCtx = 2.
    auto in2 = [&](long i) -> float {
        return (i >= 0 && i < 4) ? out1[i] : 0.0F;
    };
    float out2[2];
    for (long t = 0; t < 2; ++t) {
        const long ce = t * 2;
        out2[t] = geluRef(b2 + u0 * in2(ce - 1) + u1 * in2(ce) + u2 * in2(ce + 1));
    }

    const ConvStemOutput s = whisperConvStem(mel.data(), nMels, nFrames, d,
                                             c1w.data(), c1b.data(),
                                             c2w.data(), c2b.data());
    EXPECT_EQ(s.nCtx, static_cast<std::size_t>(2));
    EXPECT_EQ(s.dModel, static_cast<std::size_t>(1));
    EXPECT_NEAR(s.data[0], out2[0], 1e-6F);
    EXPECT_NEAR(s.data[1], out2[1], 1e-6F);
}

TEST(prompt_tokens_default_transcribe) {
    WhisperDecodeOptions opt;   // en, transcribe, no timestamps
    const std::vector<std::int32_t> p = whisperInitialPromptTokens(opt);
    EXPECT_EQ(p.size(), static_cast<std::size_t>(4));
    EXPECT_EQ(p[0], 50258);   // sot
    EXPECT_EQ(p[1], 50259);   // en
    EXPECT_EQ(p[2], 50359);   // transcribe
    EXPECT_EQ(p[3], 50363);   // notimestamps
}

TEST(prompt_tokens_translate) {
    WhisperDecodeOptions opt;
    opt.translate = true;
    const std::vector<std::int32_t> p = whisperInitialPromptTokens(opt);
    EXPECT_EQ(p[2], 50358);   // translate
}

TEST(prompt_tokens_with_timestamps_drops_notimestamps) {
    WhisperDecodeOptions opt;
    opt.timestamps = true;
    const std::vector<std::int32_t> p = whisperInitialPromptTokens(opt);
    EXPECT_EQ(p.size(), static_cast<std::size_t>(3));
    EXPECT_EQ(p[2], 50359);
}

TEST(argmax_row_picks_max) {
    const std::vector<float> a = {0.1F, 0.5F, 0.3F, -1.0F};
    EXPECT_EQ(argmaxRow(std::span<const float>{a}), 1);
    const std::vector<float> b = {2.0F, 0.5F, 0.3F};
    EXPECT_EQ(argmaxRow(std::span<const float>{b}), 0);
    const std::vector<float> c = {-3.0F, -0.5F, -0.3F};
    EXPECT_EQ(argmaxRow(std::span<const float>{c}), 2);
}
