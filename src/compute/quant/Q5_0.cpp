// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "compute/quant/Q5_0.hpp"

#include "compute/Dequant.hpp"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace mimirmind::compute::quant {

const Q5_0& Q5_0::instance() noexcept {
    static const Q5_0 inst;
    return inst;
}

core::gguf::GgmlType Q5_0::ggmlType() const noexcept {
    return core::gguf::GgmlType::Q5_0;
}

std::string_view Q5_0::name() const noexcept {
    return "Q5_0";
}

std::size_t Q5_0::blockElements() const noexcept {
    return kBlockElements;
}

std::size_t Q5_0::blockBytes() const noexcept {
    return kBlockBytes;
}

void Q5_0::dequantToF32(const void* src,
                        std::size_t nelements,
                        float*      dst) const {
    if (nelements % kBlockElements != 0) {
        throw std::runtime_error(
            "dequant Q5_0: nelements=" + std::to_string(nelements) +
            " is not a multiple of " + std::to_string(kBlockElements));
    }
    const std::size_t nblocks = nelements / kBlockElements;
    const auto*       base    = static_cast<const std::uint8_t*>(src);

    // Matches llama.cpp `dequantize_row_q5_0` bit-for-bit. The block
    // splits into two halves: element j uses qs[j] low nibble + qh bit j,
    // element j+16 uses qs[j] high nibble + qh bit j+16.
    constexpr std::size_t kHalfBlock = kBlockElements / 2;   // 16

    for (std::size_t b = 0; b < nblocks; ++b) {
        const auto* block = base + b * kBlockBytes;

        std::uint16_t dHalf;
        std::memcpy(&dHalf, block, sizeof(std::uint16_t));
        const float d = halfToFloat(dHalf);

        // qh is 4 little-endian bytes; memcpy handles alignment safely
        // (blocks are packed, no natural 4-byte alignment guaranteed).
        std::uint32_t qh;
        std::memcpy(&qh, block + 2, sizeof(qh));

        const auto* qs = block + 6;
        for (std::size_t j = 0; j < kHalfBlock; ++j) {
            const std::uint8_t xhLo =
                static_cast<std::uint8_t>((qh >> j) & 0x1U) << 4;
            const std::uint8_t xhHi =
                static_cast<std::uint8_t>((qh >> (j + 16)) & 0x1U) << 4;

            const std::int32_t x0 =
                static_cast<std::int32_t>((qs[j] & 0x0FU) | xhLo) - 16;
            const std::int32_t x1 =
                static_cast<std::int32_t>((qs[j] >>   4)  | xhHi) - 16;

            dst[j]              = d * static_cast<float>(x0);
            dst[j + kHalfBlock] = d * static_cast<float>(x1);
        }
        dst += kBlockElements;
    }
}

} // namespace mimirmind::compute::quant