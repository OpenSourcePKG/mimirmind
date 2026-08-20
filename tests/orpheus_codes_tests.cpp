// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

// Pure-CPU tests for the Orpheus audio-token bridge (runtime/audio/
// OrpheusCodes): token-id -> SNAC code de-offset, and the 7-per-frame ->
// 3-level regroup, both matching canopyai/Orpheus-TTS convert_to_audio /
// turn_token_into_id exactly. No model needed.

#include "TestFramework.hpp"

#include "runtime/audio/OrpheusCodes.hpp"

#include <cstdint>
#include <vector>

using namespace mimirmind::runtime::audio;

TEST(orpheus_token_to_code_slot0) {
    // code = tid - 128266 - (index%7)*4096. index 0 -> slot 0.
    EXPECT_EQ(orpheusTokenToCode(128266, 0), 0);
    EXPECT_EQ(orpheusTokenToCode(128266 + 5, 0), 5);
    EXPECT_EQ(orpheusTokenToCode(128266 + 4095, 0), 4095);
}

TEST(orpheus_token_to_code_slots) {
    // slot = index % 7 shifts the valid 4096-block.
    EXPECT_EQ(orpheusTokenToCode(128266 + 4096 + 7, 1), 7);      // slot 1
    EXPECT_EQ(orpheusTokenToCode(128266 + 6 * 4096 + 100, 6), 100); // slot 6
    EXPECT_EQ(orpheusTokenToCode(128266 + 3, 7), 3);  // index 7 wraps to slot 0
}

TEST(orpheus_token_to_code_invalid) {
    EXPECT_EQ(orpheusTokenToCode(128266 - 1, 0), -1);       // negative
    EXPECT_EQ(orpheusTokenToCode(128266 + 4096, 0), -1);    // == 4096 out of range at slot 0
    EXPECT_EQ(orpheusTokenToCode(128258, 0), -1);           // audio EOS
    EXPECT_EQ(orpheusTokenToCode(128266, 1), -1);           // right value, wrong slot
}

TEST(orpheus_regroup_one_frame) {
    // frame [f0..f6] -> coarse=[f0], mid=[f1,f4], fine=[f2,f3,f5,f6].
    std::vector<std::int32_t> flat{10, 11, 12, 13, 14, 15, 16};
    auto c = regroupOrpheusFrames(flat);
    EXPECT_EQ(c.size(), static_cast<std::size_t>(3));
    EXPECT_EQ(c[0].size(), static_cast<std::size_t>(1));
    EXPECT_EQ(c[1].size(), static_cast<std::size_t>(2));
    EXPECT_EQ(c[2].size(), static_cast<std::size_t>(4));
    EXPECT_EQ(c[0][0], 10);
    EXPECT_EQ(c[1][0], 11);  EXPECT_EQ(c[1][1], 14);
    EXPECT_EQ(c[2][0], 12);  EXPECT_EQ(c[2][1], 13);
    EXPECT_EQ(c[2][2], 15);  EXPECT_EQ(c[2][3], 16);
}

TEST(orpheus_regroup_two_frames_shape) {
    std::vector<std::int32_t> flat;
    for (int f = 0; f < 2; ++f)
        for (int k = 0; k < 7; ++k) flat.push_back(100 * f + k);
    auto c = regroupOrpheusFrames(flat);
    // N=2 frames -> {N, 2N, 4N} = {2,4,8}; and the SNAC vq_stride relation
    // coarse*4 == fine, mid*2 == fine.
    EXPECT_EQ(c[0].size(), static_cast<std::size_t>(2));
    EXPECT_EQ(c[1].size(), static_cast<std::size_t>(4));
    EXPECT_EQ(c[2].size(), static_cast<std::size_t>(8));
    EXPECT_EQ(c[0].size() * 4, c[2].size());
    EXPECT_EQ(c[1].size() * 2, c[2].size());
    // second frame's f4 lands at mid[3].
    EXPECT_EQ(c[1][2], 101);   // frame1 f1 = 100+1
    EXPECT_EQ(c[1][3], 104);   // frame1 f4 = 100+4
}

TEST(orpheus_regroup_drops_partial_frame) {
    std::vector<std::int32_t> flat{1, 2, 3, 4, 5, 6, 7, 8, 9};  // 9 -> 1 frame
    auto c = regroupOrpheusFrames(flat);
    EXPECT_EQ(c[0].size(), static_cast<std::size_t>(1));
    EXPECT_EQ(c[2].size(), static_cast<std::size_t>(4));
}

TEST(orpheus_voices_list) {
    EXPECT_EQ(orpheusVoices().size(), static_cast<std::size_t>(8));
    EXPECT_TRUE(orpheusVoices()[0] == "tara");
}
