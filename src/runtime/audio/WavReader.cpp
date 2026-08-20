// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/audio/WavReader.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <ios>
#include <iterator>
#include <sstream>
#include <stdexcept>

namespace mimirmind::runtime::audio {

namespace {

[[noreturn]] void fail(std::string_view msg) {
    std::ostringstream os;
    os << "WavReader: " << msg;
    throw std::runtime_error(os.str());
}

// Little-endian scalar reads over a byte span with explicit bounds checks —
// the container is untrusted input, so every read is guarded.
std::uint32_t readU32(std::span<const std::byte> b, std::size_t off) {
    if (off + 4 > b.size()) {
        fail("truncated: 32-bit read past end of buffer");
    }
    const auto* p = reinterpret_cast<const std::uint8_t*>(b.data() + off);
    return static_cast<std::uint32_t>(p[0]) |
           (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) |
           (static_cast<std::uint32_t>(p[3]) << 24);
}

std::uint16_t readU16(std::span<const std::byte> b, std::size_t off) {
    if (off + 2 > b.size()) {
        fail("truncated: 16-bit read past end of buffer");
    }
    const auto* p = reinterpret_cast<const std::uint8_t*>(b.data() + off);
    return static_cast<std::uint16_t>(static_cast<std::uint32_t>(p[0]) |
                                      (static_cast<std::uint32_t>(p[1]) << 8));
}

bool tagIs(std::span<const std::byte> b, std::size_t off, const char (&tag)[5]) {
    if (off + 4 > b.size()) {
        return false;
    }
    return std::memcmp(b.data() + off, tag, 4) == 0;
}

// Decode one interleaved sample at byte offset `off` for the given format and
// bit depth into a normalised float in [-1, 1].
float sampleToFloat(std::span<const std::byte> b, std::size_t off,
                    std::uint16_t fmt, std::uint16_t bits) {
    const auto* p = reinterpret_cast<const std::uint8_t*>(b.data() + off);
    if (fmt == 3) { // IEEE float
        if (bits == 32) {
            float v{};
            std::memcpy(&v, p, sizeof(v));
            return v;
        }
        if (bits == 64) {
            double v{};
            std::memcpy(&v, p, sizeof(v));
            return static_cast<float>(v);
        }
        fail("unsupported IEEE-float bit depth (only 32/64)");
    }
    // PCM integer.
    switch (bits) {
        case 8: {
            // 8-bit WAV PCM is unsigned, centred at 128.
            return (static_cast<float>(p[0]) - 128.0F) / 128.0F;
        }
        case 16: {
            const auto v = static_cast<std::int16_t>(
                static_cast<std::uint16_t>(p[0]) |
                (static_cast<std::uint16_t>(p[1]) << 8));
            return static_cast<float>(v) / 32768.0F;
        }
        case 24: {
            std::int32_t v = static_cast<std::int32_t>(p[0]) |
                             (static_cast<std::int32_t>(p[1]) << 8) |
                             (static_cast<std::int32_t>(p[2]) << 16);
            if (v & 0x00800000) {
                v |= ~0x00FFFFFF; // sign-extend 24 -> 32
            }
            return static_cast<float>(v) / 8388608.0F;
        }
        case 32: {
            const auto v = static_cast<std::int32_t>(
                static_cast<std::uint32_t>(p[0]) |
                (static_cast<std::uint32_t>(p[1]) << 8) |
                (static_cast<std::uint32_t>(p[2]) << 16) |
                (static_cast<std::uint32_t>(p[3]) << 24));
            return static_cast<float>(v) / 2147483648.0F;
        }
        default:
            fail("unsupported PCM bit depth (only 8/16/24/32)");
    }
}

} // namespace

AudioClip WavReader::decode(std::span<const std::byte> wav) {
    // RIFF header: "RIFF" <u32 size> "WAVE".
    if (wav.size() < 12) {
        fail("too small to be a WAV (need at least a 12-byte RIFF header)");
    }
    if (!tagIs(wav, 0, "RIFF")) {
        fail("missing 'RIFF' magic");
    }
    if (!tagIs(wav, 8, "WAVE")) {
        fail("missing 'WAVE' form type");
    }

    // Walk the chunk table from offset 12: [4-byte id][u32 size][payload],
    // payload padded to an even length. Do not assume any chunk order.
    std::uint16_t fmt{0};
    std::uint16_t channels{0};
    std::uint32_t sampleRate{0};
    std::uint16_t bits{0};
    bool          haveFmt{false};
    std::size_t   dataOff{0};
    std::size_t   dataLen{0};
    bool          haveData{false};

    std::size_t off = 12;
    while (off + 8 <= wav.size()) {
        const std::uint32_t chunkSize = readU32(wav, off + 4);
        const std::size_t   payload   = off + 8;
        if (tagIs(wav, off, "fmt ")) {
            if (chunkSize < 16 || payload + 16 > wav.size()) {
                fail("malformed 'fmt ' chunk");
            }
            fmt        = readU16(wav, payload + 0);
            channels   = readU16(wav, payload + 2);
            sampleRate = readU32(wav, payload + 4);
            bits       = readU16(wav, payload + 14);
            if (fmt == 0xFFFE) { // WAVE_FORMAT_EXTENSIBLE
                if (chunkSize < 40 || payload + 26 > wav.size()) {
                    fail("malformed EXTENSIBLE 'fmt ' chunk");
                }
                // Real format lives in the sub-format GUID's first 2 bytes.
                fmt = readU16(wav, payload + 24);
            }
            haveFmt = true;
        } else if (tagIs(wav, off, "data")) {
            dataOff  = payload;
            // Clamp to the actual buffer — some encoders write a 0 or oversized
            // data size; trust the buffer, not the declared length.
            dataLen  = chunkSize;
            if (dataOff + dataLen > wav.size()) {
                dataLen = wav.size() - dataOff;
            }
            haveData = true;
        }
        // Advance to the next chunk (payloads are word-aligned).
        std::size_t advance = 8 + chunkSize + (chunkSize & 1U);
        if (advance < 8) { // overflow guard
            break;
        }
        off += advance;
    }

    if (!haveFmt) {
        fail("no 'fmt ' chunk found");
    }
    if (!haveData) {
        fail("no 'data' chunk found");
    }
    if (fmt != 1 && fmt != 3) {
        fail("unsupported WAV format tag (only PCM=1 and IEEE-float=3)");
    }
    if (channels == 0) {
        fail("zero channels");
    }
    if (sampleRate == 0) {
        fail("zero sample rate");
    }
    const std::size_t bytesPerSample = bits / 8U;
    if (bytesPerSample == 0) {
        fail("zero bit depth");
    }
    const std::size_t frameBytes = bytesPerSample * channels;
    const std::size_t nFrames    = frameBytes ? dataLen / frameBytes : 0;

    AudioClip clip;
    clip.sampleRate = static_cast<int>(sampleRate);
    clip.samples.resize(nFrames);
    // Down-mix to mono by averaging the interleaved channels.
    const float invCh = 1.0F / static_cast<float>(channels);
    for (std::size_t f = 0; f < nFrames; ++f) {
        float acc = 0.0F;
        const std::size_t base = dataOff + f * frameBytes;
        for (std::uint16_t c = 0; c < channels; ++c) {
            acc += sampleToFloat(wav, base + c * bytesPerSample, fmt, bits);
        }
        clip.samples[f] = acc * invCh;
    }
    return clip;
}

AudioClip WavReader::decodeFile(std::string_view path) {
    std::ifstream in(std::string{path}, std::ios::binary);
    if (!in) {
        std::ostringstream os;
        os << "cannot open WAV file '" << path << "'";
        fail(os.str());
    }
    std::vector<char> raw((std::istreambuf_iterator<char>(in)),
                          std::istreambuf_iterator<char>());
    return decode(std::as_bytes(std::span<const char>(raw.data(), raw.size())));
}

AudioClip WavReader::resampleTo(const AudioClip& clip, int targetRate) {
    if (targetRate <= 0) {
        fail("resample target rate must be positive");
    }
    if (clip.sampleRate == targetRate || clip.samples.empty()) {
        AudioClip out = clip;
        out.sampleRate = targetRate;
        return out;
    }
    const double ratio = static_cast<double>(targetRate) / clip.sampleRate;
    const std::size_t inN  = clip.samples.size();
    const std::size_t outN = static_cast<std::size_t>(
        std::llround(static_cast<double>(inN) * ratio));

    AudioClip out;
    out.sampleRate = targetRate;
    out.samples.resize(outN);
    if (outN == 0) {
        return out;
    }
    if (inN == 1) {
        // Degenerate: a single input sample maps to a constant.
        std::fill(out.samples.begin(), out.samples.end(), clip.samples[0]);
        return out;
    }
    // Map each output index back to a fractional input position and linearly
    // interpolate between the two bracketing input samples.
    const double step = static_cast<double>(inN - 1) /
                        static_cast<double>(outN > 1 ? outN - 1 : 1);
    for (std::size_t i = 0; i < outN; ++i) {
        const double pos = static_cast<double>(i) * step;
        std::size_t  i0  = static_cast<std::size_t>(pos);
        if (i0 >= inN - 1) {
            out.samples[i] = clip.samples[inN - 1];
            continue;
        }
        const double frac = pos - static_cast<double>(i0);
        const float  a    = clip.samples[i0];
        const float  b    = clip.samples[i0 + 1];
        out.samples[i]    = a + static_cast<float>(frac) * (b - a);
    }
    return out;
}

AudioClip WavReader::decodeToMono(std::span<const std::byte> wav, int targetRate) {
    return resampleTo(decode(wav), targetRate);
}

} // namespace mimirmind::runtime::audio
