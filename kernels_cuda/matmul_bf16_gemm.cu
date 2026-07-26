// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Batched matrix-matrix multiply with BF16 weights (serving decode).
//
//   Y[m, n] = sum_{k} X[m, k] * bf16_to_f32(W[n, k])
//
//   X:  [M, K]  F32 dense activations (M = batch = nSeq, M <= GEMM_MAX_M)
//   W:  [N, K]  BF16 dense weights, row-major (2 bytes / element)
//   Y:  [M, N]  F32 dense output
//
// The point vs matmul_bf16_vec (which is a GEMV looped M times by the host):
// each weight row W[n,:] is read ONCE from global memory and reused across
// all M activation rows, so the (large) weight traffic is amortised by the
// batch. That turns a batched decode from ~M serial forwards into one
// memory-bound pass -> serving throughput scales with the batch.
//
// Layout: one warp per output column n (OUTPUTS_PER_GROUP warps per block).
// The M activation rows for the current K-tile are staged in shared memory
// so every warp reads them from SMEM; each lane keeps M running partial
// sums (registers) and the warp reduces them at the end.
//
// M is bounded by GEMM_MAX_M so the per-lane accumulator array is a small
// fixed size; the host chunks larger batches into GEMM_MAX_M-row launches
// (weights are still read once per chunk).
//
// Launch:
//   dim3 grid ( ceil(N / OUTPUTS_PER_GROUP), 1, 1 )
//   dim3 block( MATMUL_BF16_GEMM_LOCAL, 1, 1 )

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#ifndef MATMUL_BF16_GEMM_LOCAL
#define MATMUL_BF16_GEMM_LOCAL 128
#endif

#define MATMUL_BF16_GEMM_WARPS             (MATMUL_BF16_GEMM_LOCAL / 32)
#define MATMUL_BF16_GEMM_OUTPUTS_PER_GROUP MATMUL_BF16_GEMM_WARPS

// Max batch rows per launch. Larger batches are chunked by the host. Keeps
// the per-lane accumulator array + shared X tile bounded.
#define GEMM_MAX_M   16
#define GEMM_X_TILE  256

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

extern "C" __global__ __launch_bounds__(MATMUL_BF16_GEMM_LOCAL)
void matmul_bf16_gemm(
    const float*          __restrict__ X,   // [M, K]
    const __nv_bfloat16*  __restrict__ W,   // [N, K]
          float*          __restrict__ Y,   // [M, N]
    const int                          K,
    const int                          N,
    const int                          M)   // <= GEMM_MAX_M
{
    __shared__ float xTile[GEMM_MAX_M * GEMM_X_TILE];   // [M, tileK]

    const int wg     = blockIdx.x;
    const int tid    = threadIdx.x;
    const int lsize  = blockDim.x;
    const int warpId = tid / 32;
    const int laneId = tid % 32;
    const int n      = wg * MATMUL_BF16_GEMM_OUTPUTS_PER_GROUP + warpId;
    const bool active = (n < N);

    float acc[GEMM_MAX_M];
#pragma unroll
    for (int m = 0; m < GEMM_MAX_M; ++m) {
        acc[m] = 0.0f;
    }

    for (int tile = 0; tile < K; tile += GEMM_X_TILE) {
        const int tileK = min(GEMM_X_TILE, K - tile);

        // Stage X[0..M-1, tile..tile+tileK) into shared memory.
        for (int i = tid; i < M * tileK; i += lsize) {
            const int m  = i / tileK;
            const int kk = i % tileK;
            xTile[m * GEMM_X_TILE + kk] = X[static_cast<size_t>(m) * K + tile + kk];
        }
        __syncthreads();

        if (active) {
            const __nv_bfloat16* __restrict__ wRow =
                W + static_cast<size_t>(n) * static_cast<size_t>(K) + tile;

            for (int k = laneId; k < tileK; k += 32) {
                const float w = __bfloat162float(wRow[k]);
                for (int m = 0; m < M; ++m) {
                    acc[m] = __fmaf_rn(w, xTile[m * GEMM_X_TILE + k], acc[m]);
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
