// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/audio/WavWriter.hpp"

#include <algorithm>
#include <cmath>

namespace mimirmind::runtime::audio {

namespace {

// Append a little-endian integer of the given byte width to `out`.
void putLE(std::vector<std::byte>& out, std::uint32_t value, int bytes) {
    for (int i = 0; i < bytes; ++i) {
        out.push_back(static_cast<std::byte>((value >> (8 * i)) & 0xFFu));
    }
}

void putTag(std::vector<std::byte>& out, const char (&tag)[5]) {
    // tag is a 4-char C string literal ("RIFF") + NUL — write the 4 chars.
    for (int i = 0; i < 4; ++i) {
        out.push_back(static_cast<std::byte>(static_cast<unsigned char>(tag[i])));
    }
}

// Clamp to [-1, 1] and quantise to signed 16-bit. Symmetric scaling by 32767
// (not 32768) keeps +1.0 and -1.0 exactly representable and avoids overflow.
std::int16_t toI16(float s) {
    const float c = std::clamp(s, -1.0F, 1.0F);
    return static_cast<std::int16_t>(std::lround(c * 32767.0F));
}

void appendPcm16(std::vector<std::byte>& out, std::span<const float> mono) {
    for (const float s : mono) {
        const std::uint16_t u = static_cast<std::uint16_t>(toI16(s));
        out.push_back(static_cast<std::byte>(u & 0xFFu));
        out.push_back(static_cast<std::byte>((u >> 8) & 0xFFu));
    }
}

} // namespace

std::vector<std::byte>
WavWriter::encodeRawPcm16(std::span<const float> mono) {
    std::vector<std::byte> out;
    out.reserve(mono.size() * 2);
    appendPcm16(out, mono);
    return out;
}

std::vector<std::byte>
WavWriter::encodePcm16(std::span<const float> mono, int sampleRate) {
    const std::uint32_t rate     = sampleRate > 0
                                       ? static_cast<std::uint32_t>(sampleRate)
                                       : 24000u;
    const std::uint16_t channels = 1;
    const std::uint16_t bits     = 16;
    const std::uint16_t blockAlign = static_cast<std::uint16_t>(channels * bits / 8);
    const std::uint32_t byteRate   = rate * blockAlign;
    const std::uint32_t dataBytes  =
        static_cast<std::uint32_t>(mono.size()) * blockAlign;

    std::vector<std::byte> out;
    out.reserve(44 + dataBytes);

    // ---- RIFF header ----
    putTag(out, "RIFF");
    putLE(out, 36u + dataBytes, 4);   // file size - 8
    putTag(out, "WAVE");

    // ---- fmt chunk (PCM, 16 bytes) ----
    putTag(out, "fmt ");
    putLE(out, 16u, 4);               // fmt chunk size
    putLE(out, 1u, 2);                // audio format = 1 (PCM)
    putLE(out, channels, 2);
    putLE(out, rate, 4);
    putLE(out, byteRate, 4);
    putLE(out, blockAlign, 2);
    putLE(out, bits, 2);

    // ---- data chunk ----
    putTag(out, "data");
    putLE(out, dataBytes, 4);
    appendPcm16(out, mono);

    return out;
}

} // namespace mimirmind::runtime::audio
