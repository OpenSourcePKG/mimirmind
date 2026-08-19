// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

// Pure-CPU tests for the RIFF/WAVE decoder + resampler (runtime/audio/WavReader)
// feeding the Whisper front-end. Covers the PCM/float variants, channel
// down-mix, chunk-table walking, malformed-input rejection, and linear
// resampling to 16 kHz. No device or checkpoint needed.

#include "TestFramework.hpp"

#include "runtime/audio/WavReader.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using namespace mimirmind::runtime::audio;

namespace {

// Minimal WAV byte-stream builder. `data` is the raw interleaved sample bytes;
// `fmtTag` 1 = PCM int, 3 = IEEE float. Emits a canonical
// RIFF/"fmt "/"data" container (16-byte fmt chunk).
std::vector<std::uint8_t> buildWav(std::uint16_t fmtTag, std::uint16_t channels,
                                   std::uint32_t rate, std::uint16_t bits,
                                   const std::vector<std::uint8_t>& data) {
    auto putU32 = [](std::vector<std::uint8_t>& v, std::uint32_t x) {
        v.push_back(static_cast<std::uint8_t>(x & 0xFF));
        v.push_back(static_cast<std::uint8_t>((x >> 8) & 0xFF));
        v.push_back(static_cast<std::uint8_t>((x >> 16) & 0xFF));
        v.push_back(static_cast<std::uint8_t>((x >> 24) & 0xFF));
    };
    auto putU16 = [](std::vector<std::uint8_t>& v, std::uint16_t x) {
        v.push_back(static_cast<std::uint8_t>(x & 0xFF));
        v.push_back(static_cast<std::uint8_t>((x >> 8) & 0xFF));
    };
    auto putTag = [](std::vector<std::uint8_t>& v, const char* t) {
        for (int i = 0; i < 4; ++i) v.push_back(static_cast<std::uint8_t>(t[i]));
    };

    const std::uint16_t blockAlign = channels * (bits / 8);
    const std::uint32_t byteRate   = rate * blockAlign;

    std::vector<std::uint8_t> v;
    putTag(v, "RIFF");
    putU32(v, static_cast<std::uint32_t>(36 + data.size())); // RIFF size
    putTag(v, "WAVE");
    putTag(v, "fmt ");
    putU32(v, 16);
    putU16(v, fmtTag);
    putU16(v, channels);
    putU32(v, rate);
    putU32(v, byteRate);
    putU16(v, blockAlign);
    putU16(v, bits);
    putTag(v, "data");
    putU32(v, static_cast<std::uint32_t>(data.size()));
    v.insert(v.end(), data.begin(), data.end());
    return v;
}

std::span<const std::byte> asBytes(const std::vector<std::uint8_t>& v) {
    return std::as_bytes(std::span<const std::uint8_t>(v.data(), v.size()));
}

// Little-endian sample byte emitters for the test data payload.
void pushS16(std::vector<std::uint8_t>& v, std::int16_t x) {
    const auto u = static_cast<std::uint16_t>(x);
    v.push_back(static_cast<std::uint8_t>(u & 0xFF));
    v.push_back(static_cast<std::uint8_t>((u >> 8) & 0xFF));
}
void pushF32(std::vector<std::uint8_t>& v, float x) {
    std::uint8_t b[4];
    std::memcpy(b, &x, 4);
    for (auto e : b) v.push_back(e);
}

} // namespace

TEST(wav_pcm16_mono_roundtrip) {
    std::vector<std::uint8_t> data;
    pushS16(data, 0);        // 0.0
    pushS16(data, 16384);    // +0.5
    pushS16(data, -16384);   // -0.5
    pushS16(data, 32767);    // ~+1.0
    const auto wav = buildWav(1, 1, 16000, 16, data);

    const AudioClip clip = WavReader::decode(asBytes(wav));
    EXPECT_EQ(clip.sampleRate, 16000);
    EXPECT_EQ(clip.samples.size(), static_cast<std::size_t>(4));
    EXPECT_NEAR(clip.samples[0], 0.0F, 1e-4F);
    EXPECT_NEAR(clip.samples[1], 0.5F, 1e-4F);
    EXPECT_NEAR(clip.samples[2], -0.5F, 1e-4F);
    EXPECT_NEAR(clip.samples[3], 32767.0F / 32768.0F, 1e-4F);
}

TEST(wav_pcm16_stereo_downmix) {
    // Two frames, L/R interleaved: frame0 (0.5, -0.5) -> 0; frame1 (0.5, 0.5) -> 0.5
    std::vector<std::uint8_t> data;
    pushS16(data, 16384);
    pushS16(data, -16384);
    pushS16(data, 16384);
    pushS16(data, 16384);
    const auto wav = buildWav(1, 2, 44100, 16, data);

    const AudioClip clip = WavReader::decode(asBytes(wav));
    EXPECT_EQ(clip.sampleRate, 44100);
    EXPECT_EQ(clip.samples.size(), static_cast<std::size_t>(2));
    EXPECT_NEAR(clip.samples[0], 0.0F, 1e-4F);
    EXPECT_NEAR(clip.samples[1], 0.5F, 1e-4F);
}

TEST(wav_float32_mono) {
    std::vector<std::uint8_t> data;
    pushF32(data, 0.25F);
    pushF32(data, -0.75F);
    const auto wav = buildWav(3, 1, 16000, 32, data);

    const AudioClip clip = WavReader::decode(asBytes(wav));
    EXPECT_EQ(clip.samples.size(), static_cast<std::size_t>(2));
    EXPECT_NEAR(clip.samples[0], 0.25F, 1e-6F);
    EXPECT_NEAR(clip.samples[1], -0.75F, 1e-6F);
}

TEST(wav_pcm8_unsigned) {
    // 8-bit WAV PCM is unsigned, centred at 128.
    std::vector<std::uint8_t> data{128, 192, 64};   // 0.0, +0.5, -0.5
    const auto wav = buildWav(1, 1, 8000, 8, data);

    const AudioClip clip = WavReader::decode(asBytes(wav));
    EXPECT_EQ(clip.samples.size(), static_cast<std::size_t>(3));
    EXPECT_NEAR(clip.samples[0], 0.0F, 1e-4F);
    EXPECT_NEAR(clip.samples[1], 0.5F, 1e-4F);
    EXPECT_NEAR(clip.samples[2], -0.5F, 1e-4F);
}

TEST(wav_pcm24_signed) {
    // 0x400000 = +0.5 * 2^23; LE bytes 00 00 40. Negative: 0xC00000 -> -0.5.
    std::vector<std::uint8_t> data{0x00, 0x00, 0x40, 0x00, 0x00, 0xC0};
    const auto wav = buildWav(1, 1, 16000, 24, data);

    const AudioClip clip = WavReader::decode(asBytes(wav));
    EXPECT_EQ(clip.samples.size(), static_cast<std::size_t>(2));
    EXPECT_NEAR(clip.samples[0], 0.5F, 1e-4F);
    EXPECT_NEAR(clip.samples[1], -0.5F, 1e-4F);
}

TEST(wav_skips_unknown_chunk_before_data) {
    // Insert a LIST chunk between fmt and data; the decoder must walk past it.
    std::vector<std::uint8_t> data;
    pushS16(data, 16384);
    auto wav = buildWav(1, 1, 16000, 16, data);

    // Splice a "LIST" chunk (id + size + 4-byte payload) right before "data".
    // "data" tag sits at offset 36 in the canonical layout.
    std::vector<std::uint8_t> list{'L', 'I', 'S', 'T', 4, 0, 0, 0, 1, 2, 3, 4};
    wav.insert(wav.begin() + 36, list.begin(), list.end());
    // Fix the RIFF size to include the inserted chunk.
    const std::uint32_t newSize = static_cast<std::uint32_t>(wav.size() - 8);
    wav[4] = static_cast<std::uint8_t>(newSize & 0xFF);
    wav[5] = static_cast<std::uint8_t>((newSize >> 8) & 0xFF);
    wav[6] = static_cast<std::uint8_t>((newSize >> 16) & 0xFF);
    wav[7] = static_cast<std::uint8_t>((newSize >> 24) & 0xFF);

    const AudioClip clip = WavReader::decode(asBytes(wav));
    EXPECT_EQ(clip.samples.size(), static_cast<std::size_t>(1));
    EXPECT_NEAR(clip.samples[0], 0.5F, 1e-4F);
}

TEST(wav_resample_8k_to_16k_linear) {
    AudioClip in;
    in.sampleRate = 8000;
    in.samples = {0.0F, 1.0F};   // ramp

    const AudioClip out = WavReader::resampleTo(in, 16000);
    EXPECT_EQ(out.sampleRate, 16000);
    EXPECT_EQ(out.samples.size(), static_cast<std::size_t>(4));
    // Positions 0, 1/3, 2/3, 1 over the [0,1] ramp.
    EXPECT_NEAR(out.samples[0], 0.0F, 1e-5F);
    EXPECT_NEAR(out.samples[1], 1.0F / 3.0F, 1e-5F);
    EXPECT_NEAR(out.samples[2], 2.0F / 3.0F, 1e-5F);
    EXPECT_NEAR(out.samples[3], 1.0F, 1e-5F);
}

TEST(wav_resample_noop_same_rate) {
    AudioClip in;
    in.sampleRate = 16000;
    in.samples = {0.1F, 0.2F, 0.3F};
    const AudioClip out = WavReader::resampleTo(in, 16000);
    EXPECT_EQ(out.samples.size(), static_cast<std::size_t>(3));
    EXPECT_NEAR(out.samples[2], 0.3F, 1e-6F);
}

TEST(wav_decode_to_mono_16k_end_to_end) {
    // 8 kHz stereo PCM16 -> decodeToMono should downmix AND resample to 16k.
    std::vector<std::uint8_t> data;
    for (int i = 0; i < 4; ++i) {   // 4 stereo frames
        pushS16(data, 16384);       // L = 0.5
        pushS16(data, 16384);       // R = 0.5
    }
    const auto wav = buildWav(1, 2, 8000, 16, data);
    const AudioClip clip = WavReader::decodeToMono(asBytes(wav), 16000);
    EXPECT_EQ(clip.sampleRate, 16000);
    // 4 input frames at 8k -> ~8 output frames at 16k.
    EXPECT_EQ(clip.samples.size(), static_cast<std::size_t>(8));
    EXPECT_NEAR(clip.samples[0], 0.5F, 1e-4F);
    EXPECT_NEAR(clip.samples[7], 0.5F, 1e-4F);
}

TEST(wav_rejects_non_riff) {
    std::vector<std::uint8_t> junk{'X', 'X', 'X', 'X', 0, 0, 0, 0, 'W', 'A', 'V', 'E'};
    bool threw = false;
    try {
        (void)WavReader::decode(asBytes(junk));
    } catch (const std::runtime_error&) {
        threw = true;
    }
    EXPECT_TRUE(threw);
}

TEST(wav_rejects_missing_data_chunk) {
    // Valid RIFF/WAVE/fmt but no data chunk.
    std::vector<std::uint8_t> data;   // empty -> buildWav still emits a data hdr
    auto wav = buildWav(1, 1, 16000, 16, data);
    // Truncate the trailing "data" header (last 8 bytes) so no data chunk exists.
    wav.resize(wav.size() - 8);
    bool threw = false;
    try {
        (void)WavReader::decode(asBytes(wav));
    } catch (const std::runtime_error&) {
        threw = true;
    }
    EXPECT_TRUE(threw);
}

TEST(wav_rejects_unsupported_format_tag) {
    std::vector<std::uint8_t> data{0, 0};
    const auto wav = buildWav(7, 1, 16000, 16, data);   // 7 = A-law, unsupported
    bool threw = false;
    try {
        (void)WavReader::decode(asBytes(wav));
    } catch (const std::runtime_error&) {
        threw = true;
    }
    EXPECT_TRUE(threw);
}
