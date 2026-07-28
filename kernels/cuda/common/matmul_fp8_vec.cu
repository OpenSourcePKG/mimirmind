// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Matrix-vector multiply with blocked-FP8 (E4M3) weights (serving decode M=1).
//
//   Y[n] = sum_k X[k] * (d_block * e4m3(W[n,k]))
//
//   X:  [K]     F32 dense vector
//   W:  [N, K]  blocked FP8, each row K/32 blocks of 34 bytes
//   Y:  [N]     F32 dense vector
//
// Same block structure + warp layout as matmul_q8_0_vec (LOCAL=128 = 4 warps ×
// 32 lanes, one warp per output row, 32 quants per block = one per lane), but
// the payload is E4M3 (logarithmic) instead of int8. Reads HALF the bytes of
// the BF16 vec path. Used for the attention projections kept blocked-FP8.
//
// Block layout (34 B): fp16 d (bytes 0..1), E4M3 qs[32] (bytes 2..33);
//   value[i] = d * e4m3_decode(qs[i]).

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#ifndef MATMUL_FP8_LOCAL
#define MATMUL_FP8_LOCAL 128
#endif

#define MATMUL_FP8_WARPS             (MATMUL_FP8_LOCAL / 32)
#define MATMUL_FP8_OUTPUTS_PER_GROUP MATMUL_FP8_WARPS

#define FP8_BLOCK_ELEMENTS 32
#define FP8_BLOCK_BYTES    34
#define X_TILE_ELEMENTS    1024

namespace {

__device__ __forceinline__ float warpReduceSum(float v) {
    v += __shfl_down_sync(0xffffffffu, v, 16);
    v += __shfl_down_sync(0xffffffffu, v,  8);
    v += __shfl_down_sync(0xffffffffu, v,  4);
    v += __shfl_down_sync(0xffffffffu, v,  2);
    v += __shfl_down_sync(0xffffffffu, v,  1);
    return v;
}

} // namespace

extern "C" __global__ __launch_bounds__(MATMUL_FP8_LOCAL)
void matmul_fp8_vec(
    const float*         __restrict__ X,
    const unsigned char* __restrict__ W,   // [N, K] blocked FP8
          float*         __restrict__ Y,
    const int                          K,
    const int                          N)
{
    __shared__ float xTile[X_TILE_ELEMENTS];

    const int wg      = blockIdx.x;
    const int tid     = threadIdx.x;
    const int lsize   = blockDim.x;
    const int warpId  = tid / 32;
    const int laneId  = tid % 32;
    const int n       = wg * MATMUL_FP8_OUTPUTS_PER_GROUP + warpId;
    const bool active = (n < N);
    const int nBlocks = K / FP8_BLOCK_ELEMENTS;

    float sum = 0.0f;

    for (int tile = 0; tile < K; tile += X_TILE_ELEMENTS) {
        const int tileK = min(X_TILE_ELEMENTS, K - tile);
        for (int i = tid; i < tileK; i += lsize) {
            xTile[i] = X[tile + i];
        }
        __syncthreads();

        if (active) {
            const unsigned char* row =
                W + static_cast<size_t>(n) * static_cast<size_t>(nBlocks)
                  * FP8_BLOCK_BYTES;
            const int blockStart   = tile / FP8_BLOCK_ELEMENTS;
            const int blocksInTile = X_TILE_ELEMENTS / FP8_BLOCK_ELEMENTS;
            const int blockEnd     = min(blockStart + blocksInTile, nBlocks);
            for (int b = blockStart; b < blockEnd; ++b) {
                const unsigned char* block = row + b * FP8_BLOCK_BYTES;
                const float d = __half2float(*reinterpret_cast<const __half*>(block));
                const __nv_fp8_e4m3* qs =
                    reinterpret_cast<const __nv_fp8_e4m3*>(block + 2);
                const int xLocalBase = (b - blockStart) * FP8_BLOCK_ELEMENTS;
                const float xv = xTile[xLocalBase + laneId];
                const float dq = d * static_cast<float>(qs[laneId]);
                sum = __fmaf_rn(xv, dq, sum);
            }
        }
        __syncthreads();
    }

    sum = warpReduceSum(sum);
    if (active && laneId == 0) {
        Y[n] = sum;
    }
}
