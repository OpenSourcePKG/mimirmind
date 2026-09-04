// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "compute/QuantType.hpp"

namespace mimirmind::compute::quant {

/**
 * Q4_0 — block of 32 elements, 18 bytes:
 *   fp16      d        block scale       (2 B, offset 0)
 *   uint8_t   qs[16]   4-bit packed      (16 B, offset 2)
 *
 * qs[j] low nibble  = element j
 * qs[j] high nibble = element (j + 16)
 *
 * Reconstructed:
 *   value[j]      = d * ((qs[j] & 0x0F) - 8)
 *   value[j + 16] = d * ((qs[j] >> 4)   - 8)
 *
 * No native L0 or HIP matmul kernel exists today; consumers dispatch
 * via `compute::matmul` (CPU fallback). Registered so at-least dequant
 * works for GGUF files that store tensors as Q4_0 — llama.cpp's
 * quantizer falls back to legacy 32-block types for tensors whose row
 * length is not divisible by 256 (e.g. the 704-column
 * `ffn_down_exps` expert projections in gemma-4-26B-A4B sub-Q6 mixes).
 */
class Q4_0 final : public QuantType {
public:
    [[nodiscard]] static const Q4_0& instance() noexcept;

    [[nodiscard]] core::gguf::GgmlType ggmlType()      const noexcept override;
    [[nodiscard]] std::string_view     name()          const noexcept override;
    [[nodiscard]] std::size_t          blockElements() const noexcept override;
    [[nodiscard]] std::size_t          blockBytes()    const noexcept override;

    void dequantToF32(const void* src,
                      std::size_t nelements,
                      float*      dst) const override;

private:
    Q4_0() = default;

    static constexpr std::size_t kBlockElements = 32;
    static constexpr std::size_t kBlockBytes    = 18;
};

} // namespace mimirmind::compute::quant
