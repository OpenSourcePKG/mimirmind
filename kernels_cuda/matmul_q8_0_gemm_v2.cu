// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
// Ported from kernels_hip/matmul_q8_0_gemm_v2.hip — Track 4 mechanical port, no functional change intended.

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// M8.K.1 revised — Q8_0 GEMM with reduced SLM footprint.
//
// Identical to matmul_q8_0_gemm.hip except X_TILE_ELEMENTS shrinks
// from 1024 to 256. That drops SLM from 32 KiB to 8 KiB per WG —
// 4× more workgroups resident on the scheduler than v1.
//
// Everything else (M_TILE=8, WG=64, SG=16, 4 outputs/WG) is
// deliberately unchanged so any perf delta is attributable to the
// SLM axis alone.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifndef MATMUL_Q8_0_V2_LOCAL
#define MATMUL_Q8_0_V2_LOCAL 64
#endif

#ifndef MATMUL_Q8_0_V2_SG
#define MATMUL_Q8_0_V2_SG 16
#endif

#ifndef MATMUL_Q8_0_GEMM_V2_M_TILE
#define MATMUL_Q8_0_GEMM_V2_M_TILE 8
#endif

#define MATMUL_Q8_0_V2_OUTPUTS_PER_GROUP \
    (MATMUL_Q8_0_V2_LOCAL / MATMUL_Q8_0_V2_SG)

#define Q8_0_BLOCK_ELEMENTS 32
#define Q8_0_BLOCK_BYTES    34

// 256 elements = 8 blocks per K-tile.
// SLM: X_TILE_ELEMENTS * M_TILE * 4 B = 256 * 8 * 4 = 8 KiB / WG.
#define X_TILE_ELEMENTS 256

static __device__ __forceinline__ float warp16_reduce_sum(float v) {
    v += __shfl_xor(v, 8, 16);
    v += __shfl_xor(v, 4, 16);
    v += __shfl_xor(v, 2, 16);
    v += __shfl_xor(v, 1, 16);
    return v;
}

extern "C" __global__ __launch_bounds__(MATMUL_Q8_0_V2_LOCAL)
void matmul_q8_0_gemm_v2(
    const float*         __restrict__ X,
    const unsigned char* __restrict__ W,
          float*         __restrict__ Y,
    const int                         K,
    const int                         N,
    const int                         M)
{
    __shared__ float xTile[X_TILE_ELEMENTS][MATMUL_Q8_0_GEMM_V2_M_TILE];

    const int  wgN     = blockIdx.x;
    const int  wgM     = blockIdx.y;
    const int  tid     = threadIdx.x;
    const int  lsize   = blockDim.x;
    const int  sgInWg  = tid / MATMUL_Q8_0_V2_SG;
    const int  sgLocal = tid % MATMUL_Q8_0_V2_SG;
    const int  n       = wgN * MATMUL_Q8_0_V2_OUTPUTS_PER_GROUP + sgInWg;
    const int  mBase   = wgM * MATMUL_Q8_0_GEMM_V2_M_TILE;
    const bool nActive = (n < N);
    const int  nBlocks = K / Q8_0_BLOCK_ELEMENTS;

    float sum[MATMUL_Q8_0_GEMM_V2_M_TILE];

    #pragma unroll
    for (int mm = 0; mm < MATMUL_Q8_0_GEMM_V2_M_TILE; ++mm) {
        sum[mm] = 0.0f;
    }

    for (int tile = 0; tile < K; tile += X_TILE_ELEMENTS) {
        const int tileK = (X_TILE_ELEMENTS < K - tile)
                            ? X_TILE_ELEMENTS : (K - tile);

        const int loadTotal = MATMUL_Q8_0_GEMM_V2_M_TILE * X_TILE_ELEMENTS;
        for (int idx = tid; idx < loadTotal; idx += lsize) {
            const int  mSlot = idx / X_TILE_ELEMENTS;
            const int  iSlot = idx - mSlot * X_TILE_ELEMENTS;
            const int  mAct  = mBase + mSlot;
            const bool valid = (mAct < M) && (iSlot < tileK);
            xTile[iSlot][mSlot] =
                valid ? X[static_cast<size_t>(mAct) * static_cast<size_t>(K)
                        + tile + iSlot]
                      : 0.0f;
        }
        __syncthreads();

        if (nActive) {
            const unsigned char* __restrict__ row =
                W + static_cast<size_t>(n) * static_cast<size_t>(nBlocks)
                  * static_cast<size_t>(Q8_0_BLOCK_BYTES);

            const int blockStart   = tile / Q8_0_BLOCK_ELEMENTS;
            const int blocksInTile = X_TILE_ELEMENTS / Q8_0_BLOCK_ELEMENTS;
            const int blockEnd     = (blockStart + blocksInTile < nBlocks)
                                       ? (blockStart + blocksInTile)
                                       : nBlocks;

            for (int b = blockStart; b < blockEnd; ++b) {
                const unsigned char* __restrict__ block =
                    row + b * Q8_0_BLOCK_BYTES;
                const float d =
                    __half2float(*reinterpret_cast<const __half*>(block));
                const signed char* qs =
                    reinterpret_cast<const signed char*>(block + 2);

                const int xLocalBase = (b - blockStart) * Q8_0_BLOCK_ELEMENTS;

                for (int l = sgLocal; l < Q8_0_BLOCK_ELEMENTS;
                     l += MATMUL_Q8_0_V2_SG) {
                    const float w = d * static_cast<float>(qs[l]);

                    #pragma unroll
                    for (int mm = 0; mm < MATMUL_Q8_0_GEMM_V2_M_TILE; ++mm) {
                        sum[mm] = __fmaf_rn(xTile[xLocalBase + l][mm],
                                            w, sum[mm]);
                    }
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int mm = 0; mm < MATMUL_Q8_0_GEMM_V2_M_TILE; ++mm) {
        float s = warp16_reduce_sum(sum[mm]);
        if (nActive && sgLocal == 0) {
            const int mAct = mBase + mm;
            if (mAct < M) {
                Y[static_cast<size_t>(mAct) * static_cast<size_t>(N) + n] = s;
            }
        }
    }
}