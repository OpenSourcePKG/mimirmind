// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "compute/QuantType.hpp"

namespace mimirmind::compute::quant {

/**
 * Q4_1 — block of 32 elements, 20 bytes:
 *   fp16      d        block scale       (2 B, offset 0)
 *   fp16      m        block minimum     (2 B, offset 2)
 *   uint8_t   qs[16]   4-bit packed      (16 B, offset 4)
 *
 * Same nibble packing as Q4_0, but the 4-bit quant stays unsigned and
 * the block carries an explicit minimum instead of the implicit -8:
 *   value[j]      = d * (qs[j] & 0x0F) + m
 *   value[j + 16] = d * (qs[j] >> 4)   + m
 *
 * No native L0 or HIP matmul kernel exists today; consumers dispatch
 * via `compute::matmul` (CPU fallback). Registered for dequant
 * coverage alongside Q4_0/Q5_1 (legacy 32-block GGUF types).
 */
class Q4_1 final : public QuantType {
public:
    [[nodiscard]] static const Q4_1& instance() noexcept;

    [[nodiscard]] core::gguf::GgmlType ggmlType()      const noexcept override;
    [[nodiscard]] std::string_view     name()          const noexcept override;
    [[nodiscard]] std::size_t          blockElements() const noexcept override;
    [[nodiscard]] std::size_t          blockBytes()    const noexcept override;

    void dequantToF32(const void* src,
                      std::size_t nelements,
                      float*      dst) const override;

private:
    Q4_1() = default;

    static constexpr std::size_t kBlockElements = 32;
    static constexpr std::size_t kBlockBytes    = 20;
};

} // namespace mimirmind::compute::quant
