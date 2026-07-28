// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// BF16 -> Q8_0 weight quantizer (load-time, one-shot).
//
//   src:  [rows, K]      BF16 dense weights, row-major (K % 32 == 0)
//   dst:  [rows, K/32]    Q8_0 blocks, row-major
//
// Used by the NVFP4 loader to shrink the materialised BF16 attention
// projections to Q8_0 (34 B / 32 elems = ~1.06 B/elem vs 2 B/elem) so the
// memory-bound decode reads HALF the projection traffic. Q8_0's 8-bit
// mantissa captures the (FP8/BF16-origin) weight values with negligible
// loss, and the result feeds the existing matmul_q8_0_vec / _gemm kernels
// unchanged — no new matmul, no scale plumbing.
//
// Block layout (34 B, matches ggml block_q8_0 and kv_quant_commit_q8_0):
//   [0..1]   fp16 scale = absmax / 127
//   [2..33]  int8 quants = round(x / scale) clamped to [-127, 127]
//   all-zero block -> scale 0, quants 0 (round-trips to zero).
//
// Launch geometry:
//   dim3 grid ( rows, K/32, 1 )
//   dim3 block( QUANTIZE_Q8_0_LOCAL, 1, 1 )     // 32 == full block of elems

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#ifndef QUANTIZE_Q8_0_LOCAL
#define QUANTIZE_Q8_0_LOCAL 32
#endif

#define Q8_0_BLOCK_ELEMENTS 32
#define Q8_0_BLOCK_BYTES    34

extern "C" __global__ __launch_bounds__(QUANTIZE_Q8_0_LOCAL)
void quantize_bf16_to_q8_0(
    const __nv_bfloat16* __restrict__ src,   // [rows, K]
          unsigned char* __restrict__ dst,   // [rows, K/32] Q8_0
    const int                          K)    // must be multiple of 32
{
    const int row = blockIdx.x;
    const int blk = blockIdx.y;
    const int lid = threadIdx.x;

    const size_t srcIdx =
        static_cast<size_t>(row) * static_cast<size_t>(K)
      + static_cast<size_t>(blk) * Q8_0_BLOCK_ELEMENTS + lid;
    const float xv = __bfloat162float(src[srcIdx]);

    __shared__ float scratch[QUANTIZE_Q8_0_LOCAL];
    scratch[lid] = fabsf(xv);
    __syncthreads();
    for (int s = QUANTIZE_Q8_0_LOCAL >> 1; s > 0; s >>= 1) {
        if (lid < s) {
            scratch[lid] = fmaxf(scratch[lid], scratch[lid + s]);
        }
        __syncthreads();
    }
    const float absMax   = scratch[0];
    const float scale    = (absMax > 0.0f) ? (absMax * (1.0f / 127.0f)) : 0.0f;
    const float invScale = (absMax > 0.0f) ? (127.0f / absMax) : 0.0f;

    const int    nBlocksPerRow = K / Q8_0_BLOCK_ELEMENTS;
    const size_t blkBase =
        (static_cast<size_t>(row) * static_cast<size_t>(nBlocksPerRow)
       + static_cast<size_t>(blk)) * static_cast<size_t>(Q8_0_BLOCK_BYTES);

    if (lid == 0) {
        *reinterpret_cast<__half*>(dst + blkBase) = __float2half(scale);
    }

    int q = 0;
    if (absMax > 0.0f) {
        q = __float2int_rn(xv * invScale);
        q = max(-127, min(127, q));
    }
    dst[blkBase + 2 + lid] = static_cast<unsigned char>(static_cast<signed char>(q));
}
