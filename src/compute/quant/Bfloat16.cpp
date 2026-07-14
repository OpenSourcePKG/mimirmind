// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "compute/quant/Bfloat16.hpp"

#include "compute/Dequant.hpp"

#include <cstdint>

namespace mimirmind::compute::quant {

const Bfloat16& Bfloat16::instance() noexcept {
    static const Bfloat16 inst;
    return inst;
}

core::gguf::GgmlType Bfloat16::ggmlType() const noexcept {
    return core::gguf::GgmlType::BF16;
}

std::string_view Bfloat16::name() const noexcept {
    return "BF16";
}

std::size_t Bfloat16::blockElements() const noexcept {
    return 1;
}

std::size_t Bfloat16::blockBytes() const noexcept {
    return sizeof(std::uint16_t);
}

void Bfloat16::dequantToF32(const void* src,
                            std::size_t nelements,
                            float*      dst) const {
    const auto* in = static_cast<const std::uint16_t*>(src);
    for (std::size_t i = 0; i < nelements; ++i) {
        dst[i] = bf16ToFloat(in[i]);
    }
}

} // namespace mimirmind::compute::quant