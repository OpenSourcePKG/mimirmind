// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Fused dense FFN gate+up projection for T=1 decode — Q8_0 variant.
//
// Computes both projections of a single token and the SiLU gate in one
// launch, for the always-on Qwen3-Next shared expert (one dense expert per
// layer, Q8_0 gate/up weights):
//
//   Y[n] = silu( sum_k Wg[n,k]*x[k] ) * ( sum_k Wu[n,k]*x[k] )
//
//   X   : [K=dModel]       F32 token
//   Wg  : [N=nFf, K]  Q8_0  gate weights
//   Wu  : [N=nFf, K]  Q8_0  up   weights
//   Y   : [N=nFf]     F32    silu(gate)*up
//
// One warp per output n: two Q8_0 row·x dot products sharing the tiled x
// (32 lanes × 1 quant/block, like matmul_q8_0_vec), warp-reduced, then
// SiLU(gate)*up on lane 0. Replaces 2× matmul_q8_0_vec + silu_mul with a
// single kernel — a launch-reduction step for the shared-expert GEMVs.
//
// warpSize=32; LOCAL=128 = 4 warps, OUTPUTS_PER_GROUP=4.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifndef FFN_GU_Q8_LOCAL
#define FFN_GU_Q8_LOCAL 128
#endif

#define FFN_GU_Q8_WARPS             (FFN_GU_Q8_LOCAL / 32)
#define FFN_GU_Q8_OUTPUTS_PER_GROUP FFN_GU_Q8_WARPS

#define Q8_0_BLOCK_ELEMENTS 32
#define Q8_0_BLOCK_BYTES    34
#define X_TILE_ELEMENTS     1024

namespace {

__device__ __forceinline__ float warpReduceSum(float v) {
    v += __shfl_down_sync(0xffffffffu, v, 16);
    v += __shfl_down_sync(0xffffffffu, v,  8);
    v += __shfl_down_sync(0xffffffffu, v,  4);
    v += __shfl_down_sync(0xffffffffu, v,  2);
    v += __shfl_down_sync(0xffffffffu, v,  1);
    return v;
}

// This lane's contribution to one Q8_0 block dot (lane l -> quant l).
__device__ __forceinline__ float q8BlockLane(
    const unsigned char* block, float xv, int lane) {
    const float d = __half2float(*reinterpret_cast<const __half*>(block));
    const signed char* qs = reinterpret_cast<const signed char*>(block + 2);
    return xv * (d * static_cast<float>(qs[lane]));
}

} // namespace

extern "C" __global__ __launch_bounds__(FFN_GU_Q8_LOCAL)
void ffn_gate_up_fused_q8_0(
    const float*         __restrict__ X,    // [K]
    const unsigned char* __restrict__ Wg,   // [N, K] Q8_0
    const unsigned char* __restrict__ Wu,   // [N, K] Q8_0
          float*         __restrict__ Y,    // [N] silu(gate)*up
    const int                         K,
    const int                         N)
{
    __shared__ float xTile[X_TILE_ELEMENTS];

    const int wg      = blockIdx.x;
    const int tid     = threadIdx.x;
    const int lsize   = blockDim.x;
    const int warpId  = tid / 32;
    const int laneId  = tid % 32;
    const int n       = wg * FFN_GU_Q8_OUTPUTS_PER_GROUP + warpId;
    const bool active = (n < N);
    const int nBlocks = K / Q8_0_BLOCK_ELEMENTS;

    float gsum = 0.0f;
    float usum = 0.0f;

    for (int tile = 0; tile < K; tile += X_TILE_ELEMENTS) {
        const int tileK = min(X_TILE_ELEMENTS, K - tile);
        for (int i = tid; i < tileK; i += lsize) {
            xTile[i] = X[tile + i];
        }
        __syncthreads();

        if (active) {
            const unsigned char* gRow =
                Wg + static_cast<size_t>(n) * nBlocks * Q8_0_BLOCK_BYTES;
            const unsigned char* uRow =
                Wu + static_cast<size_t>(n) * nBlocks * Q8_0_BLOCK_BYTES;

            const int blockStart   = tile / Q8_0_BLOCK_ELEMENTS;
            const int blocksInTile = X_TILE_ELEMENTS / Q8_0_BLOCK_ELEMENTS;
            const int blockEnd     = min(blockStart + blocksInTile, nBlocks);

            for (int b = blockStart; b < blockEnd; ++b) {
                const float xv = xTile[(b - blockStart) * Q8_0_BLOCK_ELEMENTS + laneId];
                gsum += q8BlockLane(gRow + b * Q8_0_BLOCK_BYTES, xv, laneId);
                usum += q8BlockLane(uRow + b * Q8_0_BLOCK_BYTES, xv, laneId);
            }
        }

        __syncthreads();
    }

    gsum = warpReduceSum(gsum);
    usum = warpReduceSum(usum);

    if (active && laneId == 0) {
        const float g    = gsum;
        const float silu = g / (1.0f + __expf(-g));
        Y[n] = silu * usum;
    }
}