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

    // -------------------------------------------------------------------
    // Reordered row layout (M8.K Q8_0-Reorder foundation).
    //
    // The native ggml Q8_0 layout interleaves the fp16 block scale with
    // the 32 int8 quants inside every 34-byte block:
    //
    //   native row (nBlocks = K/32):
    //     [d_0][qs_0[32]] [d_1][qs_1[32]] ... [d_{nBlocks-1}][qs...[32]]
    //
    // On Intel Xe iGPUs the 34-byte stride wrecks subgroup coalescing —
    // each block load skips 2 bytes past the previous 32-byte quant run,
    // breaking any wide vector load pattern. llama.cpp PR #21527 fixed
    // this in the SYCL backend by splitting scales from quants at
    // preprocessing time. Empirical result: Q8_0 dense mmvq bandwidth
    // utilisation jumped from ~21 % to ~66 % on Xe2/Battlemage — 3.1x
    // throughput on Qwen3.5-27B, a >2x on Qwen2.5-1.5B.
    //
    // Reordered row layout (bytes, per matmul row of length K):
    //   [ nBlocks * fp16 scales  (2 B each, contiguous) ]
    //   [ nBlocks * 32 int8 quants (32 B each, contiguous) ]
    //
    // Row size is unchanged (nBlocks * 34 B). The scales region ends at
    // offset (2 * nBlocks) which is 2-aligned; the quants region starts
    // there. For all K used by our production models K is a multiple of
    // 64 (E4B: 256, 2048, 5120, 8192; 26B-A4B: 512, 2816, 5760, ...),
    // so nBlocks is even and the quants region also starts on a 32-byte
    // boundary — the alignment the subgroup uchar32 loads want.
    //
    // These functions operate on ONE row at a time. Multi-row tensors
    // (matmul weights) call them per-row so each row lives contiguously
    // in its reordered form — matches the mmvq access pattern where one
    // workgroup consumes one output row's weights.
    //
    // Bit-parity contract: reorderRow → unreorderRow round-trips byte-
    // identical to the native input, and dequantRowFromReorderedToF32
    // produces the same f32 values as dequantToF32 on the native input.

    /// Transform `nativeRow` from ggml native layout (interleaved
    /// 34-byte blocks) to reordered layout (scales region followed by
    /// quants region). K must be a multiple of 32. `nativeRow` and
    /// `reorderedRow` must not alias.
    static void reorderRow(const void* nativeRow,
                           std::size_t K,
                           void*       reorderedRow) noexcept;

    /// Inverse of `reorderRow`. Transforms reordered layout back to
    /// native interleaved 34-byte blocks. Used for parity tests, weight
    /// dumps, and as a fallback if a specific downstream path needs the
    /// legacy layout.
    static void unreorderRow(const void* reorderedRow,
                             std::size_t K,
                             void*       nativeRow) noexcept;

    /// Dequantise `K` elements from a reordered row directly to f32.
    /// Semantically identical to `dequantToF32(nativeRow, K, dst)` when
    /// `reorderedRow` was produced by `reorderRow(nativeRow, K, ...)`.
    /// Provided as a CPU oracle for kernel parity tests and as a slow
    /// but correct reference for any host-side fallback.
    static void dequantRowFromReorderedToF32(const void* reorderedRow,
                                             std::size_t K,
                                             float*      dst) noexcept;

    /// Reorder every row of an [N, K] Q8_0 weight matrix in place.
    /// `base` points at the native-layout weight bytes in USM (or any
    /// host-writable memory) and is transformed byte-by-byte into the
    /// reordered layout. `rowScratch` is a caller-owned working buffer
    /// that must be at least `blockBytes(K)` bytes — the reorder can't
    /// be done truly in-place because scales and quants interleave
    /// within each 34-byte block, so we copy one row into the scratch,
    /// reorder from the scratch back into the row, and move on.
    ///
    /// Use case: load-time preprocess for Q8_0 matmul weights. The
    /// per-row scratch stays row-sized (nBlocks * 34 B, ≤ 34 KiB even
    /// at K=32768) so a single reusable buffer suffices for a whole
    /// model's Q8_0 tensor set.
    ///
    /// Bit-parity contract: after this call, running
    /// `dequantRowFromReorderedToF32(base + n*rowBytes, K, ...)` yields
    /// the same f32 output as running `dequantToF32` did on the same
    /// row before the call. The matmulQ8_0VecReorderAsync kernel picks
    /// this layout up 1:1.
    static void reorderMatrixInPlace(void*       base,
                                     std::size_t N,
                                     std::size_t K,
                                     void*       rowScratch) noexcept;

    /// Inverse of `reorderMatrixInPlace`. Only needed when a downstream
    /// path insists on native layout (e.g. the Q6_K/Q4_K matmul kernels
    /// hitting a Q8_0 tensor that was speculatively reordered). Kept
    /// as a safety valve so we can toggle the whole path off at any
    /// point without touching the loader.
    static void unreorderMatrixInPlace(void*       base,
                                       std::size_t N,
                                       std::size_t K,
                                       void*       rowScratch) noexcept;

    /// Byte count of one Q8_0 weight row of length `K`. Handy for
    /// sizing the per-row scratch used by reorderMatrixInPlace / unreorderMatrixInPlace.
    /// K must be a multiple of 32.
    [[nodiscard]] static constexpr std::size_t rowBytes(std::size_t K) noexcept {
        return (K / kBlockElements) * kBlockBytes;
    }

private:
    Q8_0() = default;

    static constexpr std::size_t kBlockElements = 32;
    static constexpr std::size_t kBlockBytes    = 34;

    // Must match MATMUL_Q8_0_GEMM_M_TILE in kernels/matmul_q8_0_gemm.cl.
    static constexpr std::size_t kGemmMTile     = 8;
};

} // namespace mimirmind::compute::quant