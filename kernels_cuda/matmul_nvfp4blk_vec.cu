// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Matrix-vector multiply with blocked-NVFP4 weights (serving decode M=1).
//
//   Y[n] = sum_k X[k] * (scale_block * e2m1(W[n,k]))
//
// Native 4-bit NVFP4 kept resident (repackage_nvfp4_to_blk): each row is
// in/32 super-blocks of 20 bytes = fp16 s0 | fp16 s1 | 16 packed E2M1 bytes
// (2 NVFP4 blocks = 32 elements; element e uses s0 if (e%32)<16 else s1). The
// e2m1 nibbles are the exact checkpoint values, so this is lossless vs the
// BF16 the loader would otherwise materialise, at ~0.625 B/elem (3.2x less
// than BF16). Warp layout as matmul_q8_0_vec (LOCAL=128, one warp/output row,
// 32 lanes = one 32-element super per pass).

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifndef MATMUL_NVBLK_LOCAL
#define MATMUL_NVBLK_LOCAL 128
#endif

#define MATMUL_NVBLK_WARPS             (MATMUL_NVBLK_LOCAL / 32)
#define MATMUL_NVBLK_OUTPUTS_PER_GROUP MATMUL_NVBLK_WARPS

#define NVBLK_SUPER_ELEMENTS 32
#define NVBLK_SUPER_BYTES    20
#define X_TILE_ELEMENTS      1024

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

extern "C" __global__ __launch_bounds__(MATMUL_NVBLK_LOCAL)
void matmul_nvfp4blk_vec(
    const float*         __restrict__ X,
    const unsigned char* __restrict__ W,   // [N, K] blocked NVFP4
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
    const int n       = wg * MATMUL_NVBLK_OUTPUTS_PER_GROUP + warpId;
    const bool active = (n < N);
    const int nSuper  = K / NVBLK_SUPER_ELEMENTS;

    float sum = 0.0f;

    for (int tile = 0; tile < K; tile += X_TILE_ELEMENTS) {
        const int tileK = min(X_TILE_ELEMENTS, K - tile);
        for (int i = tid; i < tileK; i += lsize) {
            xTile[i] = X[tile + i];
        }
        __syncthreads();

        if (active) {
            const unsigned char* row =
                W + static_cast<size_t>(n) * static_cast<size_t>(nSuper)
                  * NVBLK_SUPER_BYTES;
            const int superStart = tile / NVBLK_SUPER_ELEMENTS;
            const int supersInTile = X_TILE_ELEMENTS / NVBLK_SUPER_ELEMENTS;
            const int superEnd   = min(superStart + supersInTile, nSuper);
            for (int sp = superStart; sp < superEnd; ++sp) {
                const unsigned char* blk = row + sp * NVBLK_SUPER_BYTES;
                const float s0 = __half2float(*reinterpret_cast<const __half*>(blk));
                const float s1 = __half2float(*reinterpret_cast<const __half*>(blk + 2));
                const unsigned char* qs = blk + 4;   // 16 packed bytes
                // Lane laneId -> element laneId of this 32-elem super.
                const unsigned char byte = qs[laneId >> 1];
                const unsigned nib = (laneId & 1) ? (byte >> 4) : (byte & 0x0F);
                const float scale = (laneId < 16) ? s0 : s1;
                const int xLocalBase = (sp - superStart) * NVBLK_SUPER_ELEMENTS;
                const float xv = xTile[xLocalBase + laneId];
                sum = __fmaf_rn(xv, scale * dq_e2m1(nib), sum);
            }
        }
        __syncthreads();
    }

    sum = warpReduceSum(sum);
    if (active && laneId == 0) {
        Y[n] = sum;
    }
}
