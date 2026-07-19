// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "compute/QuantType.hpp"

namespace mimirmind::compute::quant {

/**
 * Q3_K — super-block of 256 elements, 110 bytes:
 *   uint8 hmask[32]   high 1 bit of 256 3-bit quants (packed 8 quants / byte)
 *   uint8 qs[64]      low 2 bits of 256 3-bit quants (packed 4 quants / byte)
 *   uint8 scales[12]  16 6-bit unsigned scales packed via kmask1/kmask2 layout
 *   fp16  d           super-block scale
 *
 * Per element the 3-bit quant is reconstructed as:
 *   low_2  = (qs[qs_index] >> shift) & 0x3
 *   high_1 = (hmask[hmask_index] >> bit_pos) & 0x1
 *   q      = low_2 - (high_1 == 1 ? 0 : 4)          // signed 3-bit in [-4, 3]
 *   value  = d * (scale[sub_block] - 32) * q
 *
 * See `dequantize_row_q3_K` in llama.cpp `ggml-quants.c` for the
 * reference algorithm — the 12→16 scales unpack uses the same
 * `kmask1=0x03030303 / kmask2=0x0f0f0f0f` trick.
 *
 * HIP path: native `matmul_q3k_vec` kernel in
 * `kernels_hip/matmul_q3k_vec.hip` (WAVE32 gfx1101, one warp per output
 * row, 32 lanes × 8 elements = full 256-element super-block, no
 * cross-lane work per block). Bit-parity verified by
 * `tools/hip-matmul-q3k-probe.cpp`.
 */
class Q3K final : public QuantType {
public:
    [[nodiscard]] static const Q3K& instance() noexcept;

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
    Q3K() = default;

    static constexpr std::size_t kBlockElements = 256;
    static constexpr std::size_t kBlockBytes    = 110;

    // No GEMM variant today — matmul_q3k_vec is the only planned HIP
    // kernel and it dispatches through the same vector path as Q4_K /
    // Q6_K siblings. Sentinel value; consumers query
    // gpuMatmulGemmMTile() only when gpuMatmulGemmModule() is non-empty.
    static constexpr std::size_t kGemmMTile     = 0;
};

} // namespace mimirmind::compute::quant
