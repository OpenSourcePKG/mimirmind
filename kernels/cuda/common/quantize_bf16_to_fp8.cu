// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// BF16 -> blocked-FP8 (E4M3) weight quantizer (load-time, one-shot).
//
//   src: [rows, K]        BF16 dense weights, row-major (K % 32 == 0)
//   dst: [rows, K/32]      FP8 blocks (34 bytes each), row-major
//
// Keeps the attention projections 1-byte in a log-preserving format (E4M3),
// unlike a linear Q8_0 which crushes the small log-distributed weights and
// breaks coherence (q8_0-linear-requant lesson).
//
// PER-TENSOR scale (passed in as `scale` / `invScale = 1/scale`), NOT a
// per-block absmax. These weights are natively FP8 in the checkpoint —
// value = e4m3_orig * weight_scale. With scale == weight_scale (== absmax/448,
// since FP8 quant uses the full E4M3 range), e4m3(BF16 * invScale) lands back
// on the ORIGINAL e4m3 grid, so the round-trip is near-lossless — no
// double-quantisation error. A per-block absmax would re-snap to a different
// grid and add ~6% error, which the GatedDeltaNet recurrence amplifies over a
// sequence into degeneration. The same per-tensor `scale` is written into every
// block header so the matmul_fp8 kernels stay embedded-scale (no plumbing).
//
// Block layout (34 B): fp16 d (bytes 0..1) = scale, E4M3 qs[32] (bytes 2..33);
//   value[i] = d * e4m3_decode(qs[i]).
//
// Launch: grid( rows, K/32, 1 ), block( 32, 1, 1 ).

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>

#ifndef QUANTIZE_FP8_LOCAL
#define QUANTIZE_FP8_LOCAL 32
#endif

#define FP8_BLOCK_ELEMENTS 32
#define FP8_BLOCK_BYTES    34

extern "C" __global__ __launch_bounds__(QUANTIZE_FP8_LOCAL)
void quantize_bf16_to_fp8(
    const __nv_bfloat16* __restrict__ src,   // [rows, K]
          unsigned char* __restrict__ dst,   // [rows, K/32] blocked FP8
    const int                          K,        // multiple of 32
    const float                        scale,    // per-tensor block scale
    const float                        invScale) // 1 / scale
{
    const int row = blockIdx.x;
    const int blk = blockIdx.y;
    const int lid = threadIdx.x;

    const size_t srcIdx =
        static_cast<size_t>(row) * static_cast<size_t>(K)
      + static_cast<size_t>(blk) * FP8_BLOCK_ELEMENTS + lid;
    const float xv = __bfloat162float(src[srcIdx]);

    const int    nBlocksPerRow = K / FP8_BLOCK_ELEMENTS;
    const size_t blkBase =
        (static_cast<size_t>(row) * static_cast<size_t>(nBlocksPerRow)
       + static_cast<size_t>(blk)) * static_cast<size_t>(FP8_BLOCK_BYTES);

    if (lid == 0) {
        *reinterpret_cast<__half*>(dst + blkBase) = __float2half(scale);
    }
    const __nv_fp8_e4m3 q = static_cast<__nv_fp8_e4m3>(xv * invScale);
    dst[blkBase + 2 + lid] = q.__x;
}
