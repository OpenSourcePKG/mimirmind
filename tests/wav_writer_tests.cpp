// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

// Pure-CPU tests for the RIFF/WAVE encoder (runtime/audio/WavWriter), the TTS
// output container. Round-trips encoded bytes back through WavReader and checks
// the canonical header fields. No device or checkpoint needed.

#include "TestFramework.hpp"

#include "runtime/audio/WavReader.hpp"
#include "runtime/audio/WavWriter.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

using namespace mimirmind::runtime::audio;

namespace {

std::uint32_t readLE(const std::vector<std::byte>& b, std::size_t off, std::size_t n) {
    std::uint32_t v = 0;
    for (std::size_t i = 0; i < n; ++i) {
        v |= static_cast<std::uint32_t>(static_cast<unsigned char>(b[off + i]))
             << (8 * i);
    }
    return v;
}

bool tagAt(const std::vector<std::byte>& b, std::size_t off, const char* t) {
    for (std::size_t i = 0; i < 4; ++i) {
        if (static_cast<char>(b[off + i]) != t[i]) return false;
    }
    return true;
}

} // namespace

TEST(wavwriter_header_fields) {
    std::vector<float> mono{0.0F, 0.5F, -0.5F, 1.0F};
    const auto wav = WavWriter::encodePcm16(mono, 24000);

    // 44-byte header + 4 samples * 2 bytes.
    EXPECT_EQ(wav.size(), static_cast<std::size_t>(44 + 8));
    EXPECT_TRUE(tagAt(wav, 0, "RIFF"));
    EXPECT_EQ(readLE(wav, 4, 4), 36u + 8u);       // file size - 8
    EXPECT_TRUE(tagAt(wav, 8, "WAVE"));
    EXPECT_TRUE(tagAt(wav, 12, "fmt "));
    EXPECT_EQ(readLE(wav, 16, 4), 16u);           // fmt size
    EXPECT_EQ(readLE(wav, 20, 2), 1u);            // PCM
    EXPECT_EQ(readLE(wav, 22, 2), 1u);            // mono
    EXPECT_EQ(readLE(wav, 24, 4), 24000u);        // sample rate
    EXPECT_EQ(readLE(wav, 28, 4), 48000u);        // byte rate = 24000*2
    EXPECT_EQ(readLE(wav, 32, 2), 2u);            // block align
    EXPECT_EQ(readLE(wav, 34, 2), 16u);           // bits
    EXPECT_TRUE(tagAt(wav, 36, "data"));
    EXPECT_EQ(readLE(wav, 40, 4), 8u);            // data bytes
}

TEST(wavwriter_roundtrip_via_reader) {
    std::vector<float> mono{0.0F, 0.5F, -0.5F, 0.999969F, -1.0F, 0.25F};
    const auto wav = WavWriter::encodePcm16(mono, 24000);

    const AudioClip clip =
        WavReader::decode(std::span<const std::byte>{wav.data(), wav.size()});
    EXPECT_EQ(clip.sampleRate, 24000);
    EXPECT_EQ(clip.samples.size(), mono.size());
    for (std::size_t i = 0; i < mono.size(); ++i) {
        // 16-bit quantisation error is bounded by 1 LSB ~ 3e-5.
        EXPECT_NEAR(clip.samples[i], mono[i], 1e-4F);
    }
}

TEST(wavwriter_clamps_out_of_range) {
    std::vector<float> mono{2.0F, -2.0F};
    const auto wav = WavWriter::encodePcm16(mono, 16000);
    const AudioClip clip =
        WavReader::decode(std::span<const std::byte>{wav.data(), wav.size()});
    EXPECT_EQ(clip.samples.size(), static_cast<std::size_t>(2));
    EXPECT_NEAR(clip.samples[0], 1.0F, 1e-4F);    // clamped to +1
    EXPECT_NEAR(clip.samples[1], -1.0F, 1e-4F);   // clamped to -1
}

TEST(wavwriter_raw_pcm16_no_header) {
    std::vector<float> mono{0.0F, 1.0F, -1.0F};
    const auto raw = WavWriter::encodeRawPcm16(mono);
    EXPECT_EQ(raw.size(), static_cast<std::size_t>(6));   // 3 samples * 2 bytes
    // 0.0 -> 0, +1.0 -> 32767 (0x7FFF), -1.0 -> -32767 (0x8001).
    EXPECT_EQ(readLE(raw, 0, 2), 0u);
    EXPECT_EQ(readLE(raw, 2, 2), 0x7FFFu);
    EXPECT_EQ(readLE(raw, 4, 2), 0x8001u);
}

TEST(wavwriter_empty_is_header_only) {
    std::vector<float> mono;
    const auto wav = WavWriter::encodePcm16(mono, 24000);
    EXPECT_EQ(wav.size(), static_cast<std::size_t>(44));
    EXPECT_EQ(readLE(wav, 40, 4), 0u);            // zero data bytes
}
