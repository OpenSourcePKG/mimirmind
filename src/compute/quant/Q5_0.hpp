// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "compute/QuantType.hpp"

namespace mimirmind::compute::quant {

/**
 * Q5_0 — block of 32 elements, 22 bytes:
 *   fp16      d        block scale                (2 B, offset 0)
 *   uint32_t  qh       high-bit mask, 1 bit /elem (4 B, offset 2)
 *   uint8_t   qs[16]   low-4 bits packed          (16 B, offset 6)
 *
 * qs[j] low nibble  = element j
 * qs[j] high nibble = element (j + 16)
 * qh bit j          = high (5th) bit of element j
 * qh bit (j + 16)   = high (5th) bit of element (j + 16)
 *
 * Reconstructed:
 *   value[j]      = d * (((qs[j] & 0x0F) | ((qh >> j)        & 1) << 4) - 16)
 *   value[j + 16] = d * (((qs[j] >> 4)   | ((qh >> (j + 16)) & 1) << 4) - 16)
 *
 * No native L0 or HIP matmul kernel exists today; consumers dispatch
 * via `compute::matmul` (CPU fallback). Registered so at-least dequant
 * and CPU-side matmul work for models whose attention/FFN weights are
 * stored as Q5_0 (e.g. Qwen 2.5 Q4_K_M mixed layout).
 */
class Q5_0 final : public QuantType {
public:
    [[nodiscard]] static const Q5_0& instance() noexcept;

    [[nodiscard]] core::gguf::GgmlType ggmlType()      const noexcept override;
    [[nodiscard]] std::string_view     name()          const noexcept override;
    [[nodiscard]] std::size_t          blockElements() const noexcept override;
    [[nodiscard]] std::size_t          blockBytes()    const noexcept override;

    void dequantToF32(const void* src,
                      std::size_t nelements,
                      float*      dst) const override;

private:
    Q5_0() = default;

    static constexpr std::size_t kBlockElements = 32;
    static constexpr std::size_t kBlockBytes    = 22;
};

} // namespace mimirmind::compute::quant