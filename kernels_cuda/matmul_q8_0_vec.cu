// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
// Ported from kernels_hip/matmul_q8_0_vec.hip — Track 4 mechanical port, no functional change intended.

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Matrix-vector multiply with Q8_0 weights, on-the-fly dequant.
//
//   Y[n] = sum_{k=0..K-1} X[k] * dequant_q8_0(W, n, k)
//
//   X:  [K]     F32 dense vector (single token / M=1)
//   W:  [N, K]  Q8_0 — each row is (K/32) blocks of 34 bytes.
//   Y:  [N]     F32 dense vector
//
// HIP port of kernels/matmul_q8_0_vec.cl, restructured for the RDNA3
// warp size of 32 (vs Intel's Xe subgroup width of 16 in the L0
// reference). Each warp = one output row. Each lane in the warp
// handles exactly ONE Q8_0 quant per iterated block (32 lanes, 32
// quants per block — perfect fit, no per-lane inner loop needed).
//
// Launch:
//   dim3 grid ( ceil(N / OUTPUTS_PER_GROUP), 1, 1 )
//   dim3 block( MATMUL_Q8_0_LOCAL, 1, 1 )
//
// Config:
//   MATMUL_Q8_0_LOCAL             — total threads per block. Must be a
//                                   multiple of 32 (warpSize on RDNA3).
//                                   Default 128 = 4 warps.
//   MATMUL_Q8_0_OUTPUTS_PER_GROUP — derived: LOCAL / 32.
//   X_TILE_ELEMENTS               — how much of X is staged in LDS per
//                                   tile pass. 1024 floats = 4 KiB.
//
// Q8_0 block layout (matches ggml block_q8_0):
//   fp16    d        (2 B) — block scale
//   int8    qs[32]   (32 B) — signed 8-bit quants
//   value[i] = d * qs[i]

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifndef MATMUL_Q8_0_LOCAL
#define MATMUL_Q8_0_LOCAL 128
#endif

#define MATMUL_Q8_0_WARPS             (MATMUL_Q8_0_LOCAL / 32)
#define MATMUL_Q8_0_OUTPUTS_PER_GROUP MATMUL_Q8_0_WARPS

#define Q8_0_BLOCK_ELEMENTS 32
#define Q8_0_BLOCK_BYTES    34
#define X_TILE_ELEMENTS     1024

namespace {

// Full-warp inclusive reduction via shuffle-down. warpSize is 32 on
// RDNA3 (see lesson_rdna3_wave32.md in local memory), so five rounds
// of halving cover the whole warp with no LDS traffic. Returns the
// full sum in lane 0; other lanes hold partial state.
__device__ __forceinline__ float warpReduceSum(float v) {
    v += __shfl_down(v, 16);
    v += __shfl_down(v,  8);
    v += __shfl_down(v,  4);
    v += __shfl_down(v,  2);
    v += __shfl_down(v,  1);
    return v;
}

} // namespace

extern "C" __global__ __launch_bounds__(MATMUL_Q8_0_LOCAL)
void matmul_q8_0_vec(
    const float*         __restrict__ X,
    const unsigned char* __restrict__ W,
          float*         __restrict__ Y,
    const int                          K,
    const int                          N)
{
    __shared__ float xTile[X_TILE_ELEMENTS];

    const int wg      = blockIdx.x;
    const int tid     = threadIdx.x;
    const int lsize   = blockDim.x;
    const int warpId  = tid / 32;      // 0..MATMUL_Q8_0_WARPS-1
    const int laneId  = tid % 32;
    const int n       = wg * MATMUL_Q8_0_OUTPUTS_PER_GROUP + warpId;
    const bool active = (n < N);
    const int nBlocks = K / Q8_0_BLOCK_ELEMENTS;

    float sum = 0.0f;

    for (int tile = 0; tile < K; tile += X_TILE_ELEMENTS) {
        const int tileK = min(X_TILE_ELEMENTS, K - tile);

        // Cooperative load — all threads stripe over the tile regardless
        // of warp assignment.
        for (int i = tid; i < tileK; i += lsize) {
            xTile[i] = X[tile + i];
        }
        __syncthreads();

        if (active) {
            const unsigned char* row =
                W + static_cast<size_t>(n)
                  * static_cast<size_t>(nBlocks)
                  * Q8_0_BLOCK_BYTES;

            const int blockStart   = tile / Q8_0_BLOCK_ELEMENTS;
            const int blocksInTile = X_TILE_ELEMENTS / Q8_0_BLOCK_ELEMENTS;
            const int blockEnd     = min(blockStart + blocksInTile, nBlocks);

            for (int b = blockStart; b < blockEnd; ++b) {
                const unsigned char* block = row + b * Q8_0_BLOCK_BYTES;

                // fp16 scale sits at bytes [0..1] of the block.
                const __half*      d_ptr = reinterpret_cast<const __half*>(block);
                const float        d     = __half2float(d_ptr[0]);
                // Signed int8 quants at bytes [2..33].
                const signed char* qs    = reinterpret_cast<const signed char*>(block + 2);

                const int xLocalBase = (b - blockStart) * Q8_0_BLOCK_ELEMENTS;

                // 32 lanes × 1 quant each — no inner loop.
                const float xv = xTile[xLocalBase + laneId];
                const float dq = d * static_cast<float>(qs[laneId]);
                sum = __fmaf_rn(xv, dq, sum);
            }
        }

        __syncthreads();
    }

    // Warp reduce: each output row's warp collapses its 32 partial sums
    // into lane 0.
    sum = warpReduceSum(sum);

    if (active && laneId == 0) {
        Y[n] = sum;
    }
}