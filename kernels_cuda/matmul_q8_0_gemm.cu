// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
// Ported from kernels_hip/matmul_q8_0_gemm.hip — Track 4 mechanical port, no functional change intended.

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Batched matrix-matrix multiply with Q8_0 weights, on-the-fly
// dequant. M-way generalisation of matmul_q8_0_vec: processes M_TILE
// token rows per workgroup so the (already cheap) Q8_0 dequant plus
// the per-block scale multiply amortise across M_TILE mads.
//
//   Y[m, n] = sum_{k=0..K-1} X[m, k] * dequant_q8_0(W, n, k)
//
//   X:  [M, K]         F32 row-major
//   W:  [N, K/32]      Q8_0 blocks (34 B each: fp16 d + 32×int8)
//   Y:  [M, N]         F32 row-major
//
// Launch geometry:
//   dim3 block( MATMUL_Q8_0_LOCAL, 1, 1 )              // 64
//   dim3 grid ( ceil(N / OUTPUTS_PER_GROUP),           // 4 outputs/WG
//               ceil(M / MATMUL_Q8_0_GEMM_M_TILE),     // 8 rows/WG
//               1 )
//
// WAVE32 mapping: the L0 kernel had 64 threads split into 4 sub-groups
// of 16 lanes via `intel_reqd_sub_group_size(16)`. On RDNA3 warpSize=32
// so the same 64-thread WG is 2 waves. We model the "4 sub-groups of
// 16 lanes" pattern explicitly: sgInWg = tid / 16, sgLocal = tid % 16,
// and sub_group_reduce_add becomes a butterfly `__shfl_xor_sync(0xffffffffu, v, off, 16)`
// — the width=16 parameter scopes each reduction to its own 16-lane
// slice of the wave, so all four sub-groups reduce in parallel.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifndef MATMUL_Q8_0_LOCAL
#define MATMUL_Q8_0_LOCAL 64
#endif

#ifndef MATMUL_Q8_0_SG
#define MATMUL_Q8_0_SG 16
#endif

#ifndef MATMUL_Q8_0_GEMM_M_TILE
#define MATMUL_Q8_0_GEMM_M_TILE 8
#endif

#define MATMUL_Q8_0_OUTPUTS_PER_GROUP (MATMUL_Q8_0_LOCAL / MATMUL_Q8_0_SG)

#define Q8_0_BLOCK_ELEMENTS 32
#define Q8_0_BLOCK_BYTES    34

// 1024 elements = 32 blocks per K-tile.
// SLM: X_TILE_ELEMENTS * MATMUL_Q8_0_GEMM_M_TILE * 4 B
//    = 1024 * 8 * 4 = 32 KiB per workgroup.
#define X_TILE_ELEMENTS 1024

// Butterfly reduction over a 16-lane sub-group. width=16 makes each
// 16-lane group inside the wave reduce independently.
static __device__ __forceinline__ float warp16_reduce_sum(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 8, 16);
    v += __shfl_xor_sync(0xffffffffu, v, 4, 16);
    v += __shfl_xor_sync(0xffffffffu, v, 2, 16);
    v += __shfl_xor_sync(0xffffffffu, v, 1, 16);
    return v;
}

extern "C" __global__ __launch_bounds__(MATMUL_Q8_0_LOCAL)
void matmul_q8_0_gemm(
    const float*         __restrict__ X,
    const unsigned char* __restrict__ W,
          float*         __restrict__ Y,
    const int                         K,
    const int                         N,
    const int                         M)
{
    __shared__ float xTile[X_TILE_ELEMENTS][MATMUL_Q8_0_GEMM_M_TILE];

    const int  wgN     = blockIdx.x;
    const int  wgM     = blockIdx.y;
    const int  tid     = threadIdx.x;
    const int  lsize   = blockDim.x;
    const int  sgInWg  = tid / MATMUL_Q8_0_SG;          // 0..3
    const int  sgLocal = tid % MATMUL_Q8_0_SG;          // 0..15
    const int  n       = wgN * MATMUL_Q8_0_OUTPUTS_PER_GROUP + sgInWg;
    const int  mBase   = wgM * MATMUL_Q8_0_GEMM_M_TILE;
    const bool nActive = (n < N);
    const int  nBlocks = K / Q8_0_BLOCK_ELEMENTS;

    float sum[MATMUL_Q8_0_GEMM_M_TILE];

    #pragma unroll
    for (int mm = 0; mm < MATMUL_Q8_0_GEMM_M_TILE; ++mm) {
        sum[mm] = 0.0f;
    }

    for (int tile = 0; tile < K; tile += X_TILE_ELEMENTS) {
        const int tileK = (X_TILE_ELEMENTS < K - tile)
                            ? X_TILE_ELEMENTS : (K - tile);

        const int loadTotal = MATMUL_Q8_0_GEMM_M_TILE * X_TILE_ELEMENTS;
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

                // 16 sub-group lanes cover the 32-element block via
                // strided sweep; each lane dequantises 2 quants and
                // MACs into every M_TILE accumulator.
                for (int l = sgLocal; l < Q8_0_BLOCK_ELEMENTS;
                     l += MATMUL_Q8_0_SG) {
                    const float w = d * static_cast<float>(qs[l]);

                    #pragma unroll
                    for (int mm = 0; mm < MATMUL_Q8_0_GEMM_M_TILE; ++mm) {
                        sum[mm] = __fmaf_rn(xTile[xLocalBase + l][mm],
                                            w, sum[mm]);
                    }
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int mm = 0; mm < MATMUL_Q8_0_GEMM_M_TILE; ++mm) {
        float s = warp16_reduce_sum(sum[mm]);
        if (nActive && sgLocal == 0) {
            const int mAct = mBase + mm;
            if (mAct < M) {
                Y[static_cast<size_t>(mAct) * static_cast<size_t>(N) + n] = s;
            }
        }
    }
}