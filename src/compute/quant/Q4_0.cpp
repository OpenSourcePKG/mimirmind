// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "compute/quant/Q4_0.hpp"

#include "compute/Dequant.hpp"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace mimirmind::compute::quant {

const Q4_0& Q4_0::instance() noexcept {
    static const Q4_0 inst;
    return inst;
}

core::gguf::GgmlType Q4_0::ggmlType() const noexcept {
    return core::gguf::GgmlType::Q4_0;
}

std::string_view Q4_0::name() const noexcept {
    return "Q4_0";
}

std::size_t Q4_0::blockElements() const noexcept {
    return kBlockElements;
}

std::size_t Q4_0::blockBytes() const noexcept {
    return kBlockBytes;
}

void Q4_0::dequantToF32(const void* src,
                        std::size_t nelements,
                        float*      dst) const {
    if (nelements % kBlockElements != 0) {
        throw std::runtime_error(
            "dequant Q4_0: nelements=" + std::to_string(nelements) +
            " is not a multiple of " + std::to_string(kBlockElements));
    }
    const std::size_t nblocks = nelements / kBlockElements;
    const auto*       base    = static_cast<const std::uint8_t*>(src);

    // Matches llama.cpp `dequantize_row_q4_0` bit-for-bit.
    constexpr std::size_t kHalfBlock = kBlockElements / 2;   // 16

    for (std::size_t b = 0; b < nblocks; ++b) {
        const auto* block = base + b * kBlockBytes;

        std::uint16_t dHalf;
        std::memcpy(&dHalf, block, sizeof(std::uint16_t));
        const float d = halfToFloat(dHalf);

        const auto* qs = block + 2;
        for (std::size_t j = 0; j < kHalfBlock; ++j) {
            const std::int32_t x0 =
                static_cast<std::int32_t>(qs[j] & 0x0FU) - 8;
            const std::int32_t x1 =
                static_cast<std::int32_t>(qs[j] >>   4)  - 8;

            dst[j]              = d * static_cast<float>(x0);
            dst[j + kHalfBlock] = d * static_cast<float>(x1);
        }
        dst += kBlockElements;
    }
}

} // namespace mimirmind::compute::quant
