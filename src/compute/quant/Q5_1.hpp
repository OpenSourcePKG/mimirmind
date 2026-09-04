// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "compute/QuantType.hpp"

namespace mimirmind::compute::quant {

/**
 * Q5_1 — block of 32 elements, 24 bytes:
 *   fp16      d        block scale                (2 B, offset 0)
 *   fp16      m        block minimum              (2 B, offset 2)
 *   uint32_t  qh       high-bit mask, 1 bit /elem (4 B, offset 4)
 *   uint8_t   qs[16]   low-4 bits packed          (16 B, offset 8)
 *
 * Same nibble/high-bit packing as Q5_0, but the 5-bit quant stays
 * unsigned and the block carries an explicit minimum instead of the
 * implicit -16 offset:
 *   value[j]      = d * ((qs[j] & 0x0F) | ((qh >> j)        & 1) << 4) + m
 *   value[j + 16] = d * ((qs[j] >> 4)   | ((qh >> (j + 16)) & 1) << 4) + m
 *
 * No native L0 or HIP matmul kernel exists today; consumers dispatch
 * via `compute::matmul` (CPU fallback). Registered so at-least dequant
 * works for GGUF files that store tensors as Q5_1 — llama.cpp's
 * quantizer falls back to legacy 32-block types for tensors whose row
 * length is not divisible by 256 (e.g. the 704-column
 * `ffn_down_exps` expert projections in gemma-4-26B-A4B sub-Q6 mixes).
 */
class Q5_1 final : public QuantType {
public:
    [[nodiscard]] static const Q5_1& instance() noexcept;

    [[nodiscard]] core::gguf::GgmlType ggmlType()      const noexcept override;
    [[nodiscard]] std::string_view     name()          const noexcept override;
    [[nodiscard]] std::size_t          blockElements() const noexcept override;
    [[nodiscard]] std::size_t          blockBytes()    const noexcept override;

    void dequantToF32(const void* src,
                      std::size_t nelements,
                      float*      dst) const override;

private:
    Q5_1() = default;

    static constexpr std::size_t kBlockElements = 32;
    static constexpr std::size_t kBlockBytes    = 24;
};

} // namespace mimirmind::compute::quant
