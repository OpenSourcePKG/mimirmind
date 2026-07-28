// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Ported from kernels_cuda/moe_gate_up_fused_k_q4k.cu — mechanical HIP port
// (cuda_runtime -> hip_runtime, __shfl_down_sync -> __shfl_down), no
// functional change. gfx1101 is wave32 so the 32-lane warp reduce and the
// "one warp per (k,f) output" mapping carry over unchanged.
//
// Fused MoE gate+up projection for T=1 decode — Q4_K variant.
//
// For each of the K routed experts, computes both the gate and the up
// projection of the single input token and folds them with SiLU into the
// K-strided activation buffer the fused-K down kernel consumes:
//
//   gateAct[k, f] = silu( sum_d Wg[expIdx[k]][f,d] * x[d] )
//                 *       sum_d Wu[expIdx[k]][f,d] * x[d]
//
//   x         : [dModel]           F32 token
//   Wg / Wu   : Q4_K expert banks, per expert [nFf, dModel]
//   gateAct   : [K, nFf]           F32 (output)
//
// One warp per (k, f) output: two Q4_K row·x dot products over dModel
// sharing the tiled x, warp-reduced, then SiLU(gate)*up on lane 0. Q4_K
// super-block layout matches matmul_q4k_vec.
//
// RDNA3 warpSize=32, LOCAL=128 = 4 warps, OUTPUTS_PER_GROUP=4.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#ifndef MOE_GU_LOCAL
#define MOE_GU_LOCAL 128
#endif

#define MOE_GU_WARPS             (MOE_GU_LOCAL / 32)
#define MOE_GU_OUTPUTS_PER_GROUP MOE_GU_WARPS

#define Q4K_BLOCK_ELEMENTS 256
#define Q4K_BLOCK_BYTES    144
#define X_TILE_ELEMENTS    1024

namespace {

__device__ __forceinline__ float warpReduceSum(float v) {
    v += __shfl_down(v, 16);
    v += __shfl_down(v,  8);
    v += __shfl_down(v,  4);
    v += __shfl_down(v,  2);
    v += __shfl_down(v,  1);
    return v;
}

__device__ __forceinline__ void getScaleMinK4(
    int j, const unsigned char* q, unsigned int& scale, unsigned int& min) {
    if (j < 4) {
        scale = static_cast<unsigned int>(q[j]     & 0x3Fu);
        min   = static_cast<unsigned int>(q[j + 4] & 0x3Fu);
    } else {
        scale = static_cast<unsigned int>(
            (q[j + 4] & 0x0Fu) | ((q[j - 4] >> 6) << 4));
        min   = static_cast<unsigned int>(
            (q[j + 4] >> 4)    | ((q[j    ] >> 6) << 4));
    }
}

// This lane's partial dot of one Q4_K super-block row with the tiled x.
// Element layout identical to matmul_q4k_vec: lane l covers the 8
// positions {j*32 + l : j in 0..7}; xBase points at this block's slice.
__device__ __forceinline__ float q4kBlockDot(
    const unsigned char* block, const float* xBase, int l) {
    const __half* d_ptr    = reinterpret_cast<const __half*>(block);
    const __half* dmin_ptr = reinterpret_cast<const __half*>(block + 2);
    const float d    = __half2float(d_ptr[0]);
    const float dmin = __half2float(dmin_ptr[0]);
    const unsigned char* scales = block + 4;
    const unsigned char* qs     = block + 16;

    float acc = 0.0f;
    #pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        const int jLo = 2 * pair;
        const int jHi = jLo + 1;
        unsigned int sLo, mLo, sHi, mHi;
        getScaleMinK4(jLo, scales, sLo, mLo);
        getScaleMinK4(jHi, scales, sHi, mHi);
        const float dLo  = d    * static_cast<float>(sLo);
        const float mmLo = dmin * static_cast<float>(mLo);
        const float dHi  = d    * static_cast<float>(sHi);
        const float mmHi = dmin * static_cast<float>(mHi);
        const unsigned char qb = qs[pair * 32 + l];
        const float wLo = dLo * static_cast<float>(qb & 0x0Fu) - mmLo;
        const float wHi = dHi * static_cast<float>(qb >> 4)    - mmHi;
        acc = __fmaf_rn(xBase[jLo * 32 + l], wLo, acc);
        acc = __fmaf_rn(xBase[jHi * 32 + l], wHi, acc);
    }
    return acc;
}

} // namespace

extern "C" __global__ __launch_bounds__(MOE_GU_LOCAL)
void moe_gate_up_fused_k_q4k(
    const float*         __restrict__ X,           // [dModel]
    const unsigned char* __restrict__ Wg,          // gate expert bank
    const unsigned char* __restrict__ Wu,          // up expert bank
    const int*           __restrict__ expIdx,      // [K]
          float*         __restrict__ gateActOut,  // [K, nFf]  silu(gate)*up
    const int                         dModel,
    const int                         nFf,
    const int                         kActive,
    const int                         expertBytesGate,
    const int                         expertBytesUp)
{
    __shared__ float xTile[X_TILE_ELEMENTS];

    const int wg      = blockIdx.x;
    const int tid     = threadIdx.x;
    const int lsize   = blockDim.x;
    const int warpId  = tid / 32;
    const int laneId  = tid % 32;
    const int o       = wg * MOE_GU_OUTPUTS_PER_GROUP + warpId;
    const bool active = (o < kActive * nFf);
    const int k       = active ? (o / nFf) : 0;
    const int f       = active ? (o % nFf) : 0;
    const int nBlocks = dModel / Q4K_BLOCK_ELEMENTS;
    const int rowBytes = nBlocks * Q4K_BLOCK_BYTES;

    // Per-output expert weight rows (row f of expert e's gate / up matrix).
    const unsigned char* gateRow = nullptr;
    const unsigned char* upRow   = nullptr;
    if (active) {
        const int e = expIdx[k];
        gateRow = Wg + static_cast<size_t>(e) * expertBytesGate
                     + static_cast<size_t>(f) * rowBytes;
        upRow   = Wu + static_cast<size_t>(e) * expertBytesUp
                     + static_cast<size_t>(f) * rowBytes;
    }

    float gsum = 0.0f;
    float usum = 0.0f;

    for (int tile = 0; tile < dModel; tile += X_TILE_ELEMENTS) {
        const int tileK = (X_TILE_ELEMENTS < dModel - tile)
                            ? X_TILE_ELEMENTS : (dModel - tile);
        for (int i = tid; i < tileK; i += lsize) {
            xTile[i] = X[tile + i];
        }
        __syncthreads();

        if (active) {
            const int blockStart   = tile / Q4K_BLOCK_ELEMENTS;
            const int blocksInTile = X_TILE_ELEMENTS / Q4K_BLOCK_ELEMENTS;
            const int blockEnd     = (blockStart + blocksInTile < nBlocks)
                                       ? (blockStart + blocksInTile) : nBlocks;

            for (int b = blockStart; b < blockEnd; ++b) {
                const float* xBase = xTile + (b - blockStart) * Q4K_BLOCK_ELEMENTS;
                gsum += q4kBlockDot(gateRow + b * Q4K_BLOCK_BYTES, xBase, laneId);
                usum += q4kBlockDot(upRow   + b * Q4K_BLOCK_BYTES, xBase, laneId);
            }
        }

        __syncthreads();
    }

    gsum = warpReduceSum(gsum);
    usum = warpReduceSum(usum);

    if (active && laneId == 0) {
        // SiLU(gate) * up  (matches compute::siluMul semantics).
        const float g = gsum;
        const float silu = g / (1.0f + __expf(-g));
        gateActOut[static_cast<size_t>(k) * nFf + f] = silu * usum;
    }
}
