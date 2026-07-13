#pragma once

#include "compute/QuantType.hpp"

namespace mimirmind::compute::quant {

/**
 * Q6_K — super-block of 256 elements, 210 bytes:
 *   uint8 ql[128]    lower 4 bits of 256 6-bit quants
 *   uint8 qh[64]     upper 2 bits of 256 6-bit quants (4 quants/byte)
 *   int8  scales[16] 16 signed scales, one per 16-element sub-block
 *   fp16  d          super-block scale
 * Per element: q = ((ql_nibble) | (qh_pair << 4)) - 32  in [-32..31]
 * value = d * scale[sub_block] * q
 */
class Q6K final : public QuantType {
public:
    [[nodiscard]] static const Q6K& instance() noexcept;

    [[nodiscard]] core::gguf::GgmlType  ggmlType()            const noexcept override;
    [[nodiscard]] std::string_view name()                const noexcept override;
    [[nodiscard]] std::size_t      blockElements()       const noexcept override;
    [[nodiscard]] std::size_t      blockBytes()          const noexcept override;
    [[nodiscard]] std::string_view gpuMatmulModule()     const noexcept override;
    [[nodiscard]] std::string_view gpuMatmulGemmModule() const noexcept override;
    [[nodiscard]] std::size_t      gpuMatmulGemmMTile()  const noexcept override;

    void dequantToF32(const void* src,
                      std::size_t nelements,
                      float*      dst) const override;

private:
    Q6K() = default;

    static constexpr std::size_t kBlockElements = 256;
    static constexpr std::size_t kBlockBytes    = 210;

    // Must match MATMUL_Q6K_GEMM_M_TILE in kernels/matmul_q6k_gemm.cl.
    static constexpr std::size_t kGemmMTile     = 8;
};

} // namespace mimirmind::compute::quant