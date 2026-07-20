// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
// Ported from kernels_hip/matmul_q8_0_vec_reorder.hip — Track 4 mechanical port, no functional change intended.

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Matrix-vector multiply with Q8_0 weights in REORDERED row layout.
//
// Structurally identical to matmul_q8_0_vec.hip — same launch
// geometry, same math, same fp32 accumulator — but the row-local
// layout is scales-then-quants instead of interleaved 34-byte blocks.
// The native kernel pays a scattered-load tax because the 34-byte
// block stride breaks subgroup coalescing (Scale @ offset 0, quants
// @ offset 2, next block's scale @ offset 34 — the 2-byte gap ejects
// every subgroup vector load).
//
// Reorder splits each row into two contiguous regions:
//   [ nBlocks * fp16 scales (2 B each,  2*nBlocks bytes total) ]
//   [ nBlocks * 32 int8 quants (32 B each, 32*nBlocks bytes total) ]
// Total row size unchanged (nBlocks * 34 B).
//
//   Y[n] = sum_{k=0..K-1} X[k] * dequant_q8_0(W_reordered, n, k)
//
//   X:  [K]     F32 dense vector (single token / M=1)
//   W:  [N, K]  Q8_0 reordered
//   Y:  [N]     F32
//
// Launch geometry identical to matmul_q8_0_vec.hip. See the GEMM
// port for the WAVE32-mapping story (WgSize=64 with explicit sg16
// via tid/16 + tid%16, warp16_reduce_sum via __shfl_xor width=16).

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifndef MATMUL_Q8_0_LOCAL
#define MATMUL_Q8_0_LOCAL 64
#endif

#ifndef MATMUL_Q8_0_SG
#define MATMUL_Q8_0_SG 16
#endif

#define MATMUL_Q8_0_OUTPUTS_PER_GROUP (MATMUL_Q8_0_LOCAL / MATMUL_Q8_0_SG)

#define Q8_0_BLOCK_ELEMENTS 32
#define Q8_0_BLOCK_BYTES    34

// 1024 elements = 32 blocks = 4 KiB SLM per workgroup.
#define X_TILE_ELEMENTS 1024

static __device__ __forceinline__ float warp16_reduce_sum(float v) {
    v += __shfl_xor(v, 8, 16);
    v += __shfl_xor(v, 4, 16);
    v += __shfl_xor(v, 2, 16);
    v += __shfl_xor(v, 1, 16);
    return v;
}

extern "C" __global__ __launch_bounds__(MATMUL_Q8_0_LOCAL)
void matmul_q8_0_vec_reorder(
    const float*         __restrict__ X,
    const unsigned char* __restrict__ W,
          float*         __restrict__ Y,
    const int                         K,
    const int                         N)
{
    __shared__ float xTile[X_TILE_ELEMENTS];

    const int  wg      = blockIdx.x;
    const int  tid     = threadIdx.x;
    const int  lsize   = blockDim.x;
    const int  sgInWg  = tid / MATMUL_Q8_0_SG;
    const int  sgLocal = tid % MATMUL_Q8_0_SG;
    const int  n       = wg * MATMUL_Q8_0_OUTPUTS_PER_GROUP + sgInWg;
    const bool active  = (n < N);
    const int  nBlocks = K / Q8_0_BLOCK_ELEMENTS;

    float sum = 0.0f;

    for (int tile = 0; tile < K; tile += X_TILE_ELEMENTS) {
        const int tileK = (X_TILE_ELEMENTS < K - tile)
                            ? X_TILE_ELEMENTS : (K - tile);
        for (int i = tid; i < tileK; i += lsize) {
            xTile[i] = X[tile + i];
        }
        __syncthreads();

        if (active) {
            // Row base: total row size unchanged (nBlocks * 34 B) so
            // the outer stride stays identical to the native kernel.
            // Only the WITHIN-row layout differs.
            const unsigned char* __restrict__ row =
                W + static_cast<size_t>(n) * static_cast<size_t>(nBlocks)
                  * static_cast<size_t>(Q8_0_BLOCK_BYTES);

            const __half* scales =
                reinterpret_cast<const __half*>(row);
            const signed char* quants =
                reinterpret_cast<const signed char*>(
                    row + 2 * static_cast<size_t>(nBlocks));

            const int blockStart   = tile / Q8_0_BLOCK_ELEMENTS;
            const int blocksInTile = X_TILE_ELEMENTS / Q8_0_BLOCK_ELEMENTS;
            const int blockEnd     = (blockStart + blocksInTile < nBlocks)
                                       ? (blockStart + blocksInTile)
                                       : nBlocks;

            for (int b = blockStart; b < blockEnd; ++b) {
                const float d = __half2float(scales[b]);
                const signed char* qs =
                    quants + b * Q8_0_BLOCK_ELEMENTS;

                const int xLocalBase = (b - blockStart) * Q8_0_BLOCK_ELEMENTS;

                for (int l = sgLocal; l < Q8_0_BLOCK_ELEMENTS;
                     l += MATMUL_Q8_0_SG) {
                    sum = __fmaf_rn(xTile[xLocalBase + l],
                                    d * static_cast<float>(qs[l]),
                                    sum);
                }
            }
        }

        __syncthreads();
    }

    sum = warp16_reduce_sum(sum);

    if (active && sgLocal == 0) {
        Y[n] = sum;
    }
}