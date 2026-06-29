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

// ----- Q8_0 --------------------------------------------------------------
//
// Block of 32 elements, 34 bytes:
//   fp16  d        — block scale (2 B)
//   int8  qs[32]   — 32 signed 8-bit quants (32 B)
// value[i] = d * qs[i]

constexpr std::size_t kQ80BlockElements = 32;
constexpr std::size_t kQ80BlockBytes    = 34;

void dequantQ80(const void* src, std::size_t nelements, float* dst) {
    if (nelements % kQ80BlockElements != 0) {
        throw std::runtime_error(
            "dequant Q8_0: nelements=" + std::to_string(nelements) +
            " is not a multiple of " + std::to_string(kQ80BlockElements));
    }
    const std::size_t nblocks = nelements / kQ80BlockElements;
    const auto* base = static_cast<const std::uint8_t*>(src);

    for (std::size_t b = 0; b < nblocks; ++b) {
        const auto* block = base + b * kQ80BlockBytes;

        std::uint16_t dHalf;
        std::memcpy(&dHalf, block, sizeof(std::uint16_t));
        const float d = halfToFloat(dHalf);

        const auto* qs = reinterpret_cast<const std::int8_t*>(block + 2);
        for (std::size_t l = 0; l < kQ80BlockElements; ++l) {
            *dst++ = d * static_cast<float>(qs[l]);
        }
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

// ----- Q6_K --------------------------------------------------------------
//
// Super-block of 256 elements, 210 bytes:
//   uint8 ql[128]    — lower 4 bits of 256 6-bit quants
//   uint8 qh[64]     — upper 2 bits of 256 6-bit quants (4 quants/byte)
//   int8  scales[16] — 16 signed scales, one per 16-element sub-block
//   fp16  d          — super-block scale
//
// Per element: q = ((ql_nibble) | (qh_pair << 4)) - 32  in [-32..31]
// value = d * scale[sub_block] * q

constexpr std::size_t kQ6KBlockElements = 256;
constexpr std::size_t kQ6KBlockBytes    = 210;

void dequantQ6K(const void* src, std::size_t nelements, float* dst) {
    if (nelements % kQ6KBlockElements != 0) {
        throw std::runtime_error(
            "dequant Q6_K: nelements=" + std::to_string(nelements) +
            " is not a multiple of " + std::to_string(kQ6KBlockElements));
    }
    const std::size_t nblocks = nelements / kQ6KBlockElements;
    const auto* base = static_cast<const std::uint8_t*>(src);

    for (std::size_t b = 0; b < nblocks; ++b) {
        const auto* block = base + b * kQ6KBlockBytes;

        const std::uint8_t* ql = block;            // 128 bytes
        const std::uint8_t* qh = block + 128;      // 64 bytes
        const auto*         sc = reinterpret_cast<const std::int8_t*>(block + 192); // 16 bytes
        std::uint16_t dHalf;
        std::memcpy(&dHalf, block + 208, sizeof(std::uint16_t));
        const float d = halfToFloat(dHalf);

        float*       y          = dst;
        const std::uint8_t* qlp = ql;
        const std::uint8_t* qhp = qh;
        const std::int8_t*  scp = sc;

        // Two 128-element halves per super-block.
        for (int half = 0; half < 2; ++half) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const auto q1 = static_cast<std::int8_t>(
                    (qlp[l +  0] & 0x0FU) | (((qhp[l] >> 0) & 0x03U) << 4)) - 32;
                const auto q2 = static_cast<std::int8_t>(
                    (qlp[l + 32] & 0x0FU) | (((qhp[l] >> 2) & 0x03U) << 4)) - 32;
                const auto q3 = static_cast<std::int8_t>(
                    (qlp[l +  0] >> 4U)   | (((qhp[l] >> 4) & 0x03U) << 4)) - 32;
                const auto q4 = static_cast<std::int8_t>(
                    (qlp[l + 32] >> 4U)   | (((qhp[l] >> 6) & 0x03U) << 4)) - 32;
                y[l +  0] = d * static_cast<float>(scp[is + 0]) * static_cast<float>(q1);
                y[l + 32] = d * static_cast<float>(scp[is + 2]) * static_cast<float>(q2);
                y[l + 64] = d * static_cast<float>(scp[is + 4]) * static_cast<float>(q3);
                y[l + 96] = d * static_cast<float>(scp[is + 6]) * static_cast<float>(q4);
            }
            y   += 128;
            qlp += 64;
            qhp += 32;
            scp += 8;
        }

        dst += kQ6KBlockElements;
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
        case model::GgmlType::Q8_0: dequantQ80(src,  nelements, dst); return;
        case model::GgmlType::Q4_K: dequantQ4K(src,  nelements, dst); return;
        case model::GgmlType::Q6_K: dequantQ6K(src,  nelements, dst); return;

        case model::GgmlType::F64:
        case model::GgmlType::I8:
        case model::GgmlType::I16:
        case model::GgmlType::I32:
        case model::GgmlType::I64:
        case model::GgmlType::Q4_0:
        case model::GgmlType::Q4_1:
        case model::GgmlType::Q5_0:
        case model::GgmlType::Q5_1:
        case model::GgmlType::Q8_1:
        case model::GgmlType::Q2_K:
        case model::GgmlType::Q3_K:
        case model::GgmlType::Q5_K:
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