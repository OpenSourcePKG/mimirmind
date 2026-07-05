#pragma once

#include "compute/QuantType.hpp"

#include <cstdint>

namespace mimirmind::compute::quant {

/**
 * Q5_K — super-block of 256 elements, 176 bytes:
 *   fp16  d          scale-of-scales   (2 B)
 *   fp16  dmin       scale-of-mins     (2 B)
 *   uint8 scales[12] eight 6-bit scales + eight 6-bit mins, packed
 *                    (identical layout to Q4_K's `scales` field)
 *   uint8 qh[32]     high bit of each of 256 5-bit quants
 *                    (256 bits packed one-per-quant, four 64-element
 *                    sub-super-blocks share the same 32 bytes with a
 *                    2-bit-per-iteration mask shift)
 *   uint8 qs[128]    lower 4 bits of the 256 quants (256 nibbles,
 *                    lo-nibble first inside each byte)
 *
 * Per element: q = (qs_nibble) | (qh_bit << 4)   in [0..31]
 *              value = d * scale * q - dmin * min
 *
 * The scale/min packing matches Q4_K's `getScaleMinK4` byte-for-byte
 * so Q5_K and Q4_K share that helper.
 *
 * M9.11 speculative decoding: the Gemma 4 E4B GGUF (bartowski Q4_K_M)
 * uses Q5_K for the attention-projection tensors (attn_k, attn_output),
 * so a draft forward pass hits this kernel on every layer.
 */
class Q5K final : public QuantType {
public:
    [[nodiscard]] static const Q5K& instance() noexcept;

    [[nodiscard]] model::GgmlType  ggmlType()            const noexcept override;
    [[nodiscard]] std::string_view name()                const noexcept override;
    [[nodiscard]] std::size_t      blockElements()       const noexcept override;
    [[nodiscard]] std::size_t      blockBytes()          const noexcept override;
    [[nodiscard]] std::string_view gpuMatmulModule()     const noexcept override;

    void dequantToF32(const void* src,
                      std::size_t nelements,
                      float*      dst) const override;

private:
    Q5K() = default;

    static constexpr std::size_t kBlockElements = 256;
    static constexpr std::size_t kBlockBytes    = 176;
};

} // namespace mimirmind::compute::quant