// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "compute/quant/Q5_1.hpp"

#include "compute/Dequant.hpp"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace mimirmind::compute::quant {

const Q5_1& Q5_1::instance() noexcept {
    static const Q5_1 inst;
    return inst;
}

core::gguf::GgmlType Q5_1::ggmlType() const noexcept {
    return core::gguf::GgmlType::Q5_1;
}

std::string_view Q5_1::name() const noexcept {
    return "Q5_1";
}

std::size_t Q5_1::blockElements() const noexcept {
    return kBlockElements;
}

std::size_t Q5_1::blockBytes() const noexcept {
    return kBlockBytes;
}

void Q5_1::dequantToF32(const void* src,
                        std::size_t nelements,
                        float*      dst) const {
    if (nelements % kBlockElements != 0) {
        throw std::runtime_error(
            "dequant Q5_1: nelements=" + std::to_string(nelements) +
            " is not a multiple of " + std::to_string(kBlockElements));
    }
    const std::size_t nblocks = nelements / kBlockElements;
    const auto*       base    = static_cast<const std::uint8_t*>(src);

    // Matches llama.cpp `dequantize_row_q5_1` bit-for-bit. Same split
    // as Q5_0 (element j: qs[j] low nibble + qh bit j, element j+16:
    // qs[j] high nibble + qh bit j+16), but unsigned quant + minimum.
    constexpr std::size_t kHalfBlock = kBlockElements / 2;   // 16

    for (std::size_t b = 0; b < nblocks; ++b) {
        const auto* block = base + b * kBlockBytes;

        std::uint16_t dHalf;
        std::memcpy(&dHalf, block, sizeof(std::uint16_t));
        const float d = halfToFloat(dHalf);

        std::uint16_t mHalf;
        std::memcpy(&mHalf, block + 2, sizeof(std::uint16_t));
        const float m = halfToFloat(mHalf);

        // qh is 4 little-endian bytes; memcpy handles alignment safely
        // (blocks are packed, no natural 4-byte alignment guaranteed).
        std::uint32_t qh;
        std::memcpy(&qh, block + 4, sizeof(qh));

        const auto* qs = block + 8;
        for (std::size_t j = 0; j < kHalfBlock; ++j) {
            const std::uint8_t xhLo =
                static_cast<std::uint8_t>((qh >> j) & 0x1U) << 4;
            const std::uint8_t xhHi =
                static_cast<std::uint8_t>((qh >> (j + 16)) & 0x1U) << 4;

            const std::uint8_t x0 =
                static_cast<std::uint8_t>((qs[j] & 0x0FU) | xhLo);
            const std::uint8_t x1 =
                static_cast<std::uint8_t>((qs[j] >>   4)  | xhHi);

            dst[j]              = d * static_cast<float>(x0) + m;
            dst[j + kHalfBlock] = d * static_cast<float>(x1) + m;
        }
        dst += kBlockElements;
    }
}

} // namespace mimirmind::compute::quant
