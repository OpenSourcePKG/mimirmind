#pragma once

#include "compute/QuantType.hpp"

namespace mimirmind::compute::quant {

/**
 * Q8_0 — block of 32 elements, 34 bytes:
 *   fp16  d        block scale (2 B)
 *   int8  qs[32]   32 signed 8-bit quants (32 B)
 * value[i] = d * qs[i].
 *
 * GPU matmul kernel: matmul_q8_0_vec (M8.G). Pre-M8.G the type fell
 * back to compute::matmul (CPU, double-acc). Per-block precision is
 * substantially higher than Q4_K/Q6_K (single scale, no sub-bands)
 * so plain FP32 per-thread accumulation is used in the kernel.
 */
class Q8_0 final : public QuantType {
public:
    [[nodiscard]] static const Q8_0& instance() noexcept;

    [[nodiscard]] model::GgmlType  ggmlType()            const noexcept override;
    [[nodiscard]] std::string_view name()                const noexcept override;
    [[nodiscard]] std::size_t      blockElements()       const noexcept override;
    [[nodiscard]] std::size_t      blockBytes()          const noexcept override;
    [[nodiscard]] std::string_view gpuMatmulModule()     const noexcept override;
    [[nodiscard]] std::string_view gpuMatmulGemmModule() const noexcept override;
    [[nodiscard]] std::size_t      gpuMatmulGemmMTile()  const noexcept override;

    void dequantToF32(const void* src,
                      std::size_t nelements,
                      float*      dst) const override;

    /// Quantize `K` f32 values from `src` into Q8_0-encoded bytes at
    /// `dst`. `K` must be a multiple of 32. Writes `(K/32) * 34` bytes.
    /// Per-block: max-abs symmetric scaling into int8 with an fp16 d,
    /// matches ggml_quantize_row_q8_0. Used by the E4B backend to lift
    /// BF16 `per_layer_model_proj` and by `FusedQkvWeights` to
    /// requantize mismatched attn_q/k/v tensors so QKV fusion applies.
    static void quantizeRow(const float* src,
                            std::size_t  K,
                            void*        dst) noexcept;

private:
    Q8_0() = default;

    static constexpr std::size_t kBlockElements = 32;
    static constexpr std::size_t kBlockBytes    = 34;

    // Must match MATMUL_Q8_0_GEMM_M_TILE in kernels/matmul_q8_0_gemm.cl.
    static constexpr std::size_t kGemmMTile     = 8;
};

} // namespace mimirmind::compute::quant