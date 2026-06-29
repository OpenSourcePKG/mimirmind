#pragma once

#include "compute/QuantType.hpp"

namespace mimirmind::compute::quant {

/**
 * Q8_0 — block of 32 elements, 34 bytes:
 *   fp16  d        block scale (2 B)
 *   int8  qs[32]   32 signed 8-bit quants (32 B)
 * value[i] = d * qs[i].
 *
 * No GPU matmul kernel yet — falls back to the CPU matmul (which uses
 * a double accumulator and is bit-exact vs llama.cpp's CPU dequant).
 */
class Q8_0 final : public QuantType {
public:
    [[nodiscard]] static const Q8_0& instance() noexcept;

    [[nodiscard]] model::GgmlType  ggmlType()      const noexcept override;
    [[nodiscard]] std::string_view name()          const noexcept override;
    [[nodiscard]] std::size_t      blockElements() const noexcept override;
    [[nodiscard]] std::size_t      blockBytes()    const noexcept override;

    void dequantToF32(const void* src,
                      std::size_t nelements,
                      float*      dst) const override;

private:
    Q8_0() = default;

    static constexpr std::size_t kBlockElements = 32;
    static constexpr std::size_t kBlockBytes    = 34;
};

} // namespace mimirmind::compute::quant