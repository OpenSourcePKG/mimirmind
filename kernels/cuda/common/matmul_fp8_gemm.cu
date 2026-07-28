// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Batched matmul with blocked-FP8 (E4M3) weights (prefill / batched decode).
//
//   Y[m, n] = sum_k X[m, k] * (d_block * e4m3(W[n, k]))
//
//   X:  [M, K]  F32 activations (M <= GEMM_MAX_M)
//   W:  [N, K]  blocked FP8 (34 B / 32 elems)
//   Y:  [M, N]  F32 output
//
// FP8 analogue of matmul_bf16_gemm: each weight row is read ONCE and reused
// across the M activation rows, but the weight bytes are ~half (blocked E4M3).
// See matmul_fp8_vec for the block layout.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#ifndef MATMUL_FP8_GEMM_LOCAL
#define MATMUL_FP8_GEMM_LOCAL 128
#endif

#define MATMUL_FP8_GEMM_WARPS             (MATMUL_FP8_GEMM_LOCAL / 32)
#define MATMUL_FP8_GEMM_OUTPUTS_PER_GROUP MATMUL_FP8_GEMM_WARPS

#define FP8_BLOCK_ELEMENTS 32
#define FP8_BLOCK_BYTES    34
#define GEMM_MAX_M   16
#define GEMM_X_TILE  256   // multiple of 32 (block boundary)

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

extern "C" __global__ __launch_bounds__(MATMUL_FP8_GEMM_LOCAL)
void matmul_fp8_gemm(
    const float*         __restrict__ X,   // [M, K]
    const unsigned char* __restrict__ W,   // [N, K] blocked FP8
          float*         __restrict__ Y,   // [M, N]
    const int                          K,
    const int                          N,
    const int                          M)   // <= GEMM_MAX_M
{
    __shared__ float xTile[GEMM_MAX_M * GEMM_X_TILE];

    const int wg     = blockIdx.x;
    const int tid    = threadIdx.x;
    const int lsize  = blockDim.x;
    const int warpId = tid / 32;
    const int laneId = tid % 32;
    const int n      = wg * MATMUL_FP8_GEMM_OUTPUTS_PER_GROUP + warpId;
    const bool active = (n < N);
    const int nBlocks = K / FP8_BLOCK_ELEMENTS;

    float acc[GEMM_MAX_M];
#pragma unroll
    for (int m = 0; m < GEMM_MAX_M; ++m) acc[m] = 0.0f;

    for (int tile = 0; tile < K; tile += GEMM_X_TILE) {
        const int tileK = min(GEMM_X_TILE, K - tile);
        for (int i = tid; i < M * tileK; i += lsize) {
            const int m  = i / tileK;
            const int kk = i % tileK;
            xTile[m * GEMM_X_TILE + kk] = X[static_cast<size_t>(m) * K + tile + kk];
        }
        __syncthreads();

        if (active) {
            const unsigned char* row =
                W + static_cast<size_t>(n) * static_cast<size_t>(nBlocks)
                  * FP8_BLOCK_BYTES;
            const int blockStart   = tile / FP8_BLOCK_ELEMENTS;
            const int blocksInTile = GEMM_X_TILE / FP8_BLOCK_ELEMENTS;
            const int blockEnd     = min(blockStart + blocksInTile, nBlocks);
            for (int b = blockStart; b < blockEnd; ++b) {
                const unsigned char* block = row + b * FP8_BLOCK_BYTES;
                const float d = __half2float(*reinterpret_cast<const __half*>(block));
                const __nv_fp8_e4m3* qs =
                    reinterpret_cast<const __nv_fp8_e4m3*>(block + 2);
                const int xBase = (b - blockStart) * FP8_BLOCK_ELEMENTS;
                for (int e = laneId; e < FP8_BLOCK_ELEMENTS; e += 32) {
                    const float w = d * static_cast<float>(qs[e]);
                    const int kk = xBase + e;
                    for (int m = 0; m < M; ++m) {
                        acc[m] = __fmaf_rn(w, xTile[m * GEMM_X_TILE + kk], acc[m]);
                    }
                }
            }
        }
        __syncthreads();
    }

    for (int m = 0; m < M; ++m) {
        const float s = warpReduceSum(acc[m]);
        if (active && laneId == 0) {
            Y[static_cast<size_t>(m) * N + n] = s;
        }
    }
}
