// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Batched matmul with blocked-NVFP4 weights (prefill / batched decode).
//   Y[m,n] = sum_k X[m,k] * (scale_block * e2m1(W[n,k]))
// See matmul_nvfp4blk_vec for the 20-byte / 32-element super-block layout.
// Each weight row is read once and reused across the M activation rows.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifndef MATMUL_NVBLK_GEMM_LOCAL
#define MATMUL_NVBLK_GEMM_LOCAL 128
#endif

#define MATMUL_NVBLK_GEMM_WARPS             (MATMUL_NVBLK_GEMM_LOCAL / 32)
#define MATMUL_NVBLK_GEMM_OUTPUTS_PER_GROUP MATMUL_NVBLK_GEMM_WARPS

#define NVBLK_SUPER_ELEMENTS 32
#define NVBLK_SUPER_BYTES    20
#define GEMM_MAX_M   16
#define GEMM_X_TILE  256   // 8 supers

namespace {

__device__ __forceinline__ float dq_e2m1(unsigned nib) {
    const float mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float v = mag[nib & 0x7u];
    return (nib & 0x8u) ? -v : v;
}

__device__ __forceinline__ float warpReduceSum(float v) {
    v += __shfl_down_sync(0xffffffffu, v, 16);
    v += __shfl_down_sync(0xffffffffu, v,  8);
    v += __shfl_down_sync(0xffffffffu, v,  4);
    v += __shfl_down_sync(0xffffffffu, v,  2);
    v += __shfl_down_sync(0xffffffffu, v,  1);
    return v;
}

} // namespace

extern "C" __global__ __launch_bounds__(MATMUL_NVBLK_GEMM_LOCAL)
void matmul_nvfp4blk_gemm(
    const float*         __restrict__ X,   // [M, K]
    const unsigned char* __restrict__ W,   // [N, K] blocked NVFP4
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
    const int n      = wg * MATMUL_NVBLK_GEMM_OUTPUTS_PER_GROUP + warpId;
    const bool active = (n < N);
    const int nSuper = K / NVBLK_SUPER_ELEMENTS;

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
                W + static_cast<size_t>(n) * static_cast<size_t>(nSuper)
                  * NVBLK_SUPER_BYTES;
            const int superStart   = tile / NVBLK_SUPER_ELEMENTS;
            const int supersInTile = GEMM_X_TILE / NVBLK_SUPER_ELEMENTS;
            const int superEnd     = min(superStart + supersInTile, nSuper);
            for (int sp = superStart; sp < superEnd; ++sp) {
                const unsigned char* blk = row + sp * NVBLK_SUPER_BYTES;
                const float s0 = __half2float(*reinterpret_cast<const __half*>(blk));
                const float s1 = __half2float(*reinterpret_cast<const __half*>(blk + 2));
                const unsigned char* qs = blk + 4;
                const unsigned char byte = qs[laneId >> 1];
                const unsigned nib = (laneId & 1) ? (byte >> 4) : (byte & 0x0F);
                const float w = ((laneId < 16) ? s0 : s1) * dq_e2m1(nib);
                const int kk = (sp - superStart) * NVBLK_SUPER_ELEMENTS + laneId;
                for (int m = 0; m < M; ++m) {
                    acc[m] = __fmaf_rn(w, xTile[m * GEMM_X_TILE + kk], acc[m]);
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
