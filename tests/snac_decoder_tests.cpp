// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

// Pure-CPU tests for the SNAC decoder's numeric primitives (runtime/audio/
// SnacDecoder) — Conv1d (incl. dilation/groups/padding), ConvTranspose1d (incl.
// stride upsampling + output_padding), and the Snake activation — plus the
// decode() guard paths. Weight-level parity vs the reference snac_24khz is an
// on-box step (needs the converted checkpoint); these lock the host math.

#include "TestFramework.hpp"

#include "runtime/audio/SnacDecoder.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace mimirmind::runtime::audio;

TEST(snac_conv1d_pointwise_identity) {
    std::size_t tout = 0;
    // cin=1,T=3, k=1 weight [1], stride1 pad0 dil1 groups1 -> identity.
    auto y = SnacDecoder::conv1dTest({1.F, 2.F, 3.F}, 1, 3, {1.F}, {}, 1, 1, 1,
                                     0, 1, 1, tout);
    EXPECT_EQ(tout, static_cast<std::size_t>(3));
    EXPECT_NEAR(y[0], 1.F, 1e-6F);
    EXPECT_NEAR(y[1], 2.F, 1e-6F);
    EXPECT_NEAR(y[2], 3.F, 1e-6F);
}

TEST(snac_conv1d_k3_same_padding) {
    std::size_t tout = 0;
    // cin=1,T=4, k=3 weight [1,1,1] (moving sum), pad=1 -> 'same' length.
    auto y = SnacDecoder::conv1dTest({1.F, 2.F, 3.F, 4.F}, 1, 4, {1.F, 1.F, 1.F},
                                     {}, 1, 3, 1, 1, 1, 1, tout);
    EXPECT_EQ(tout, static_cast<std::size_t>(4));
    EXPECT_NEAR(y[0], 3.F, 1e-6F);   // 0+1+2
    EXPECT_NEAR(y[1], 6.F, 1e-6F);   // 1+2+3
    EXPECT_NEAR(y[2], 9.F, 1e-6F);   // 2+3+4
    EXPECT_NEAR(y[3], 7.F, 1e-6F);   // 3+4+0
}

TEST(snac_conv1d_depthwise_groups) {
    std::size_t tout = 0;
    // cin=cout=2, groups=2 (depthwise), k=1. weight [2,1,1] = {10,100}.
    // x = ch0:[1,2], ch1:[3,4] -> ch0*10, ch1*100.
    auto y = SnacDecoder::conv1dTest({1.F, 2.F, 3.F, 4.F}, 2, 2, {10.F, 100.F},
                                     {}, 2, 1, 1, 0, 1, 2, tout);
    EXPECT_EQ(tout, static_cast<std::size_t>(2));
    EXPECT_NEAR(y[0], 10.F, 1e-4F);
    EXPECT_NEAR(y[1], 20.F, 1e-4F);
    EXPECT_NEAR(y[2], 300.F, 1e-4F);
    EXPECT_NEAR(y[3], 400.F, 1e-4F);
}

TEST(snac_conv1d_bias_added) {
    std::size_t tout = 0;
    auto y = SnacDecoder::conv1dTest({1.F, 2.F}, 1, 2, {2.F}, {5.F}, 1, 1, 1, 0,
                                     1, 1, tout);
    EXPECT_NEAR(y[0], 7.F, 1e-6F);   // 1*2+5
    EXPECT_NEAR(y[1], 9.F, 1e-6F);   // 2*2+5
}

TEST(snac_convtranspose_stride2_upsamples) {
    std::size_t tout = 0;
    // The SNAC DecoderBlock geometry for stride s: k=2s, pad=ceil(s/2),
    // output_padding=s%2. For s=2: k=4, pad=1, outpad=0 -> Tout = 2*T exactly.
    // cin=cout=1, weight [1,1,4] = ones. x=[1,2] -> each input contributes to a
    // length-4 kernel window; length must be 4 (=2*T).
    auto y = SnacDecoder::convTranspose1dTest({1.F, 2.F}, 1, 2, {1.F, 1.F, 1.F,
                                              1.F}, {}, 1, 4, 2, 1, 0, tout);
    EXPECT_EQ(tout, static_cast<std::size_t>(4));   // stride upsampling 2*T
}

TEST(snac_convtranspose_simple_scatter) {
    std::size_t tout = 0;
    // cin=cout=1, k=2, stride=2, pad=0, outpad=0. x=[1,2], w=[1,1].
    // ti=0 scatters to [0,1]; ti=1 to [2,3]. -> [1,1,2,2], Tout=4.
    auto y = SnacDecoder::convTranspose1dTest({1.F, 2.F}, 1, 2, {1.F, 1.F}, {},
                                              1, 2, 2, 0, 0, tout);
    EXPECT_EQ(tout, static_cast<std::size_t>(4));
    EXPECT_NEAR(y[0], 1.F, 1e-6F);
    EXPECT_NEAR(y[1], 1.F, 1e-6F);
    EXPECT_NEAR(y[2], 2.F, 1e-6F);
    EXPECT_NEAR(y[3], 2.F, 1e-6F);
}

TEST(snac_snake_activation) {
    // snake(x) = x + 1/(alpha+1e-9) * sin(alpha*x)^2.
    // x=0 -> 0; x=pi/2, alpha=1 -> pi/2 + sin(pi/2)^2 = pi/2 + 1.
    const float halfPi = 1.57079632679F;
    auto y = SnacDecoder::snakeTest({0.F, halfPi}, 1, 2, {1.F});
    EXPECT_NEAR(y[0], 0.F, 1e-6F);
    EXPECT_NEAR(y[1], halfPi + 1.0F, 1e-5F);
}

TEST(snac_snake_per_channel_alpha) {
    // Two channels, distinct alpha. ch0 alpha=2 at x=0 -> 0. ch1 alpha=1 at
    // x=pi/2 -> pi/2 + 1.
    const float halfPi = 1.57079632679F;
    auto y = SnacDecoder::snakeTest({0.F, halfPi}, 2, 1, {2.F, 1.F});
    EXPECT_NEAR(y[0], 0.F, 1e-6F);
    EXPECT_NEAR(y[1], halfPi + 1.0F, 1e-5F);
}

TEST(snac_decode_requires_loaded) {
    SnacDecoder dec;
    EXPECT_TRUE(!dec.isLoaded());
    bool threw = false;
    try {
        (void)dec.decode({{0}, {0}, {0}});
    } catch (const std::runtime_error&) {
        threw = true;
    }
    EXPECT_TRUE(threw);
}
