// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/modelopt/NvFp4Reference.hpp"

#include <cmath>
#include <limits>

namespace mimirmind::core::modelopt {

float e2m1ToFloat(std::uint8_t nib) noexcept {
    // Magnitudes indexed by the low 3 bits; bit 3 is the sign.
    static constexpr float kMag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float v = kMag[nib & 0x7u];
    return (nib & 0x8u) ? -v : v;
}

float e4m3ToFloat(std::uint8_t byte) noexcept {
    const std::uint32_t s = (byte >> 7) & 0x1u;
    const std::uint32_t e = (byte >> 3) & 0xFu;
    const std::uint32_t m = byte & 0x7u;
    const float sign = s ? -1.0f : 1.0f;

    if (e == 0u) {
        // Subnormal: 2^(1-bias) * (m/8), bias = 7 -> 2^-6.
        return sign * std::ldexp(static_cast<float>(m) / 8.0f, -6);
    }
    if (e == 0xFu && m == 0x7u) {
        // The only NaN codes in E4M3 (S.1111.111). No infinities.
        return std::numeric_limits<float>::quiet_NaN();
    }
    // Normal: 2^(e-bias) * (1 + m/8).
    return sign * std::ldexp(1.0f + static_cast<float>(m) / 8.0f,
                             static_cast<int>(e) - 7);
}

void dequantNvfp4(const std::uint8_t* packed,
                  const std::uint8_t* blockScale,
                  float               globalScale,
                  std::uint64_t       rows,
                  std::uint64_t       in,
                  float*              out) noexcept {
    const std::uint64_t packedCols = in / 2;
    const std::uint64_t ksf        = in / 16;

    for (std::uint64_t r = 0; r < rows; ++r) {
        const std::uint8_t* pRow = packed + r * packedCols;
        const std::uint8_t* sRow = blockScale + r * ksf;
        float*              oRow = out + r * in;

        for (std::uint64_t j = 0; j < in; ++j) {
            const std::uint8_t byte = pRow[j / 2];
            const std::uint8_t nib  = (j & 1u) ? (byte >> 4) : (byte & 0x0Fu);
            const float blk = e4m3ToFloat(sRow[j / 16]);
            oRow[j] = globalScale * blk * e2m1ToFloat(nib);
        }
    }
}

void dequantFp8(const std::uint8_t* weight,
                float               weightScale,
                std::uint64_t       n,
                float*              out) noexcept {
    for (std::uint64_t i = 0; i < n; ++i) {
        out[i] = weightScale * e4m3ToFloat(weight[i]);
    }
}

} // namespace mimirmind::core::modelopt