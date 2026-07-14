// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "compute/quant/Q4K.hpp"

#include "compute/Dequant.hpp"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace mimirmind::compute::quant {

const Q4K& Q4K::instance() noexcept {
    static const Q4K inst;
    return inst;
}

core::gguf::GgmlType Q4K::ggmlType() const noexcept {
    return core::gguf::GgmlType::Q4_K;
}

std::string_view Q4K::name() const noexcept {
    return "Q4_K";
}

std::size_t Q4K::blockElements() const noexcept {
    return kBlockElements;
}

std::size_t Q4K::blockBytes() const noexcept {
    return kBlockBytes;
}

std::string_view Q4K::gpuMatmulModule() const noexcept {
    return "matmul_q4k_vec";
}

std::string_view Q4K::gpuMatmulGemmModule() const noexcept {
    return "matmul_q4k_gemm";
}

std::size_t Q4K::gpuMatmulGemmMTile() const noexcept {
    return kGemmMTile;
}

void Q4K::getScaleMinK4(int j,
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

void Q4K::dequantToF32(const void* src,
                       std::size_t nelements,
                       float*      dst) const {
    if (nelements % kBlockElements != 0) {
        throw std::runtime_error(
            "dequant Q4_K: nelements=" + std::to_string(nelements) +
            " is not a multiple of " + std::to_string(kBlockElements));
    }
    const std::size_t nblocks = nelements / kBlockElements;
    const auto* base = static_cast<const std::uint8_t*>(src);

    for (std::size_t b = 0; b < nblocks; ++b) {
        const auto* block = base + b * kBlockBytes;

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

} // namespace mimirmind::compute::quant