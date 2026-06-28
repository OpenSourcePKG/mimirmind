#include "compute/Dequant.hpp"

#include "runtime/Log.hpp"

#include <cstring>
#include <stdexcept>
#include <string>

namespace mimirmind::compute {

float halfToFloat(std::uint16_t h) noexcept {
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000U) << 16;
    const std::uint32_t exp  = (static_cast<std::uint32_t>(h) >> 10) & 0x1FU;
    const std::uint32_t mant = static_cast<std::uint32_t>(h) & 0x3FFU;

    std::uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;                          // +/- 0
        } else {
            // Subnormal half → normalised float.
            std::uint32_t m = mant;
            int e = -14;
            while ((m & 0x400U) == 0) {
                m <<= 1U;
                --e;
            }
            m &= 0x3FFU;
            bits = sign
                 | (static_cast<std::uint32_t>(e + 127) << 23)
                 | (m << 13);
        }
    } else if (exp == 0x1FU) {
        bits = sign | 0x7F800000U | (mant << 13);  // inf / nan
    } else {
        bits = sign
             | (static_cast<std::uint32_t>(exp + (127 - 15)) << 23)
             | (mant << 13);
    }

    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

float bf16ToFloat(std::uint16_t b) noexcept {
    const std::uint32_t bits = static_cast<std::uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

namespace {

void dequantF32(const void* src, std::size_t nelements, float* dst) {
    std::memcpy(dst, src, nelements * sizeof(float));
}

void dequantF16(const void* src, std::size_t nelements, float* dst) {
    const auto* in = static_cast<const std::uint16_t*>(src);
    for (std::size_t i = 0; i < nelements; ++i) {
        dst[i] = halfToFloat(in[i]);
    }
}

void dequantBf16(const void* src, std::size_t nelements, float* dst) {
    const auto* in = static_cast<const std::uint16_t*>(src);
    for (std::size_t i = 0; i < nelements; ++i) {
        dst[i] = bf16ToFloat(in[i]);
    }
}

// ----- Q4_K --------------------------------------------------------------
//
// Super-block of 256 elements, 144 bytes:
//   fp16 d         — scale-of-scales (2 B)
//   fp16 dmin      — scale-of-mins   (2 B)
//   uint8 scales[12]  — eight 6-bit scales + eight 6-bit mins, packed
//   uint8 qs[128]     — 256 nibbles (4-bit quants), little-nibble first
//
// Sub-blocks of 32: the packing follows ggml's get_scale_min_k4. For
// j in [0..3] the scale lives in scales[j] & 0x3F and the min in
// scales[j+4] & 0x3F. For j in [4..7] the high two bits live in the top
// of scales[j-4] and scales[j]. The element value is
//   v = d * scale * q - dmin * min
// per sub-block.

constexpr std::size_t kQ4KBlockElements = 256;
constexpr std::size_t kQ4KBlockBytes    = 144;

void getScaleMinK4(int j,
                   const std::uint8_t* q,
                   std::uint8_t&       outScale,
                   std::uint8_t&       outMin) noexcept {
    if (j < 4) {
        outScale = static_cast<std::uint8_t>(q[j]     & 0x3FU);
        outMin   = static_cast<std::uint8_t>(q[j + 4] & 0x3FU);
    } else {
        outScale = static_cast<std::uint8_t>(
            (q[j + 4] & 0x0FU) | ((q[j - 4] >> 6) << 4));
        outMin   = static_cast<std::uint8_t>(
            (q[j + 4] >> 4)    | ((q[j    ] >> 6) << 4));
    }
}

void dequantQ4K(const void* src, std::size_t nelements, float* dst) {
    if (nelements % kQ4KBlockElements != 0) {
        throw std::runtime_error(
            "dequant Q4_K: nelements=" + std::to_string(nelements) +
            " is not a multiple of " + std::to_string(kQ4KBlockElements));
    }
    const std::size_t nblocks = nelements / kQ4KBlockElements;
    const auto* base = static_cast<const std::uint8_t*>(src);

    for (std::size_t b = 0; b < nblocks; ++b) {
        const auto* block = base + b * kQ4KBlockBytes;

        std::uint16_t dHalf, dminHalf;
        std::memcpy(&dHalf,    block,     sizeof(std::uint16_t));
        std::memcpy(&dminHalf, block + 2, sizeof(std::uint16_t));
        const float d    = halfToFloat(dHalf);
        const float dmin = halfToFloat(dminHalf);

        const std::uint8_t* scales = block + 4;     // 12 bytes
        const std::uint8_t* qs     = block + 16;    // 128 bytes (256 nibbles)

        // Eight sub-blocks of 32 elements, processed two at a time
        // (lower-nibble half then upper-nibble half of the same 32 bytes).
        for (int j = 0; j < 8; j += 2) {
            std::uint8_t sLo, mLo, sHi, mHi;
            getScaleMinK4(j,     scales, sLo, mLo);
            getScaleMinK4(j + 1, scales, sHi, mHi);
            const float d1 = d * static_cast<float>(sLo);
            const float m1 = dmin * static_cast<float>(mLo);
            const float d2 = d * static_cast<float>(sHi);
            const float m2 = dmin * static_cast<float>(mHi);

            const std::uint8_t* q = qs + (j / 2) * 32;
            for (int l = 0; l < 32; ++l) {
                *dst++ = d1 * static_cast<float>(q[l] & 0x0FU) - m1;
            }
            for (int l = 0; l < 32; ++l) {
                *dst++ = d2 * static_cast<float>(q[l] >> 4)    - m2;
            }
        }
    }
}

} // namespace

void dequantToF32(model::GgmlType type,
                  const void*     src,
                  std::size_t     nelements,
                  float*          dst) {
    switch (type) {
        case model::GgmlType::F32:  dequantF32(src,  nelements, dst); return;
        case model::GgmlType::F16:  dequantF16(src,  nelements, dst); return;
        case model::GgmlType::BF16: dequantBf16(src, nelements, dst); return;
        case model::GgmlType::Q4_K: dequantQ4K(src,  nelements, dst); return;

        case model::GgmlType::F64:
        case model::GgmlType::I8:
        case model::GgmlType::I16:
        case model::GgmlType::I32:
        case model::GgmlType::I64:
        case model::GgmlType::Q4_0:
        case model::GgmlType::Q4_1:
        case model::GgmlType::Q5_0:
        case model::GgmlType::Q5_1:
        case model::GgmlType::Q8_0:
        case model::GgmlType::Q8_1:
        case model::GgmlType::Q2_K:
        case model::GgmlType::Q3_K:
        case model::GgmlType::Q5_K:
        case model::GgmlType::Q6_K:
        case model::GgmlType::Q8_K:
        case model::GgmlType::Unknown:
        default:
            break;
    }
    MM_LOG_ERROR("dequant", "type {} not yet implemented",
                 model::typeInfo(type).name);
    throw std::runtime_error(
        "dequantToF32: ggml type '" +
        std::string{model::typeInfo(type).name} + "' not yet implemented");
}

} // namespace mimirmind::compute