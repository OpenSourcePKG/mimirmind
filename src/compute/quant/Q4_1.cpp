// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "compute/quant/Q4_1.hpp"

#include "compute/Dequant.hpp"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace mimirmind::compute::quant {

const Q4_1& Q4_1::instance() noexcept {
    static const Q4_1 inst;
    return inst;
}

core::gguf::GgmlType Q4_1::ggmlType() const noexcept {
    return core::gguf::GgmlType::Q4_1;
}

std::string_view Q4_1::name() const noexcept {
    return "Q4_1";
}

std::size_t Q4_1::blockElements() const noexcept {
    return kBlockElements;
}

std::size_t Q4_1::blockBytes() const noexcept {
    return kBlockBytes;
}

void Q4_1::dequantToF32(const void* src,
                        std::size_t nelements,
                        float*      dst) const {
    if (nelements % kBlockElements != 0) {
        throw std::runtime_error(
            "dequant Q4_1: nelements=" + std::to_string(nelements) +
            " is not a multiple of " + std::to_string(kBlockElements));
    }
    const std::size_t nblocks = nelements / kBlockElements;
    const auto*       base    = static_cast<const std::uint8_t*>(src);

    // Matches llama.cpp `dequantize_row_q4_1` bit-for-bit.
    constexpr std::size_t kHalfBlock = kBlockElements / 2;   // 16

    for (std::size_t b = 0; b < nblocks; ++b) {
        const auto* block = base + b * kBlockBytes;

        std::uint16_t dHalf;
        std::memcpy(&dHalf, block, sizeof(std::uint16_t));
        const float d = halfToFloat(dHalf);

        std::uint16_t mHalf;
        std::memcpy(&mHalf, block + 2, sizeof(std::uint16_t));
        const float m = halfToFloat(mHalf);

        const auto* qs = block + 4;
        for (std::size_t j = 0; j < kHalfBlock; ++j) {
            dst[j] =
                d * static_cast<float>(qs[j] & 0x0FU) + m;
            dst[j + kHalfBlock] =
                d * static_cast<float>(qs[j] >>   4)  + m;
        }
        dst += kBlockElements;
    }
}

} // namespace mimirmind::compute::quant
