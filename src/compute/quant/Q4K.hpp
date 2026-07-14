// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "compute/QuantType.hpp"

#include <cstdint>

namespace mimirmind::compute::quant {

/**
 * Q4_K — super-block of 256 elements, 144 bytes:
 *   fp16  d         scale-of-scales (2 B)
 *   fp16  dmin      scale-of-mins   (2 B)
 *   uint8 scales[12]  eight 6-bit scales + eight 6-bit mins, packed
 *   uint8 qs[128]     256 nibbles (4-bit quants), little-nibble first
 * value = d * scale * q - dmin * min  per 32-element sub-block.
 */
class Q4K final : public QuantType {
public:
    [[nodiscard]] static const Q4K& instance() noexcept;

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
    Q4K() = default;

    /// Unpack one of the eight 6-bit scale/min pairs from the 12-byte
    /// packed `scales` field. Matches ggml's get_scale_min_k4.
    static void getScaleMinK4(int j,
                              const std::uint8_t* q,
                              std::uint8_t&       outScale,
                              std::uint8_t&       outMin) noexcept;

    static constexpr std::size_t kBlockElements = 256;
    static constexpr std::size_t kBlockBytes    = 144;

    // Must match MATMUL_Q4K_GEMM_M_TILE in kernels/matmul_q4k_gemm.cl.
    static constexpr std::size_t kGemmMTile     = 8;
};

} // namespace mimirmind::compute::quant