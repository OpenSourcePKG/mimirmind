// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Batched fused MoE down-projection (blocked-NVFP4) — NVFP4 analogue of
// moe_down_fused_k_q6k_batched. Keeps the routed down experts native 4-bit
// (E2M1), lossless. Same launch/warp16 layout as the Q6_K variant; each
// 16-lane sub-group computes one dModel output, accumulating x[ff]*w over the
// ffPer intermediate dim across the active experts, RMW into accum.
//
// Weight row: nSuper = ffPer/32 super-blocks of 20 bytes (fp16 s0|fp16 s1|16
// packed E2M1). value[e in super] = (e<16 ? s0 : s1) * e2m1(nibble_e).
//
// Launch: grid = dim3(ceil(dModel / OUTPUTS_PER_GROUP), nSeq, 1),
//         block = MOE_DOWN_LOCAL.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifndef MOE_DOWN_LOCAL
#define MOE_DOWN_LOCAL 64
#endif

#ifndef MOE_DOWN_SG
#define MOE_DOWN_SG 16
#endif

#define MOE_DOWN_OUTPUTS_PER_GROUP (MOE_DOWN_LOCAL / MOE_DOWN_SG)

#define NVBLK_SUPER_ELEMENTS 32
#define NVBLK_SUPER_BYTES    20
#define X_TILE_ELEMENTS      1024

static __device__ __forceinline__ float warp16_reduce_sum(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 8, 16);
    v += __shfl_xor_sync(0xffffffffu, v, 4, 16);
    v += __shfl_xor_sync(0xffffffffu, v, 2, 16);
    v += __shfl_xor_sync(0xffffffffu, v, 1, 16);
    return v;
}

static __device__ __forceinline__ float dq_e2m1(unsigned nib) {
    const float mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float v = mag[nib & 0x7u];
    return (nib & 0x8u) ? -v : v;
}

extern "C" __global__ __launch_bounds__(MOE_DOWN_LOCAL)
void moe_down_fused_k_nvfp4blk_batched(
    const float*         __restrict__ X,          // [nSeq, K, ffPer]
    const unsigned char* __restrict__ W,          // blocked-NVFP4 expert bank
    const int*           __restrict__ expIdx,     // [nSeq, K]
    const float*         __restrict__ kw,         // [nSeq, K]
          float*         __restrict__ accum,      // [nSeq, dModel] RMW
    const int                         ffPer,
    const int                         dModel,
    const int                         kActive,
    const int                         expertBytes)
{
    __shared__ float xTile[X_TILE_ELEMENTS];

    const int seq = blockIdx.y;
    const float* __restrict__ Xb =
        X + static_cast<size_t>(seq) * kActive * ffPer;
    const int*   __restrict__ expIdxB = expIdx + static_cast<size_t>(seq) * kActive;
    const float* __restrict__ kwB     = kw     + static_cast<size_t>(seq) * kActive;
    float*       __restrict__ accumB  = accum  + static_cast<size_t>(seq) * dModel;

    const int  wg       = blockIdx.x;
    const int  tid      = threadIdx.x;
    const int  lsize    = blockDim.x;
    const int  sgInWg   = tid / MOE_DOWN_SG;
    const int  sgLocal  = tid % MOE_DOWN_SG;
    const int  n        = wg * MOE_DOWN_OUTPUTS_PER_GROUP + sgInWg;
    const bool active   = (n < dModel);
    const int  nSuper   = ffPer / NVBLK_SUPER_ELEMENTS;
    const int  rowBytes = nSuper * NVBLK_SUPER_BYTES;

    float accumSum = 0.0f;

    for (int k = 0; k < kActive; ++k) {
        const int   e   = expIdxB[k];
        const float ekw = kwB[k];

        const unsigned char* __restrict__ Wexpert =
            W + static_cast<size_t>(e) * static_cast<size_t>(expertBytes);
        const float* __restrict__ Xk =
            Xb + static_cast<size_t>(k) * static_cast<size_t>(ffPer);

        float sum = 0.0f;

        for (int tile = 0; tile < ffPer; tile += X_TILE_ELEMENTS) {
            const int tileK = (X_TILE_ELEMENTS < ffPer - tile)
                                ? X_TILE_ELEMENTS : (ffPer - tile);
            for (int i = tid; i < tileK; i += lsize) {
                xTile[i] = Xk[tile + i];
            }
            __syncthreads();

            if (active) {
                const unsigned char* __restrict__ row =
                    Wexpert + static_cast<size_t>(n)
                            * static_cast<size_t>(rowBytes);
                const int spStart  = tile / NVBLK_SUPER_ELEMENTS;
                const int spInTile = X_TILE_ELEMENTS / NVBLK_SUPER_ELEMENTS;
                const int spEnd    = (spStart + spInTile < nSuper)
                                       ? (spStart + spInTile) : nSuper;
                for (int sp = spStart; sp < spEnd; ++sp) {
                    const unsigned char* __restrict__ super =
                        row + sp * NVBLK_SUPER_BYTES;
                    const float s0 = __half2float(*reinterpret_cast<const __half*>(super));
                    const float s1 = __half2float(*reinterpret_cast<const __half*>(super + 2));
                    const unsigned char* qs = super + 4;
                    const int xLocalBase = (sp - spStart) * NVBLK_SUPER_ELEMENTS;
                    for (int el = sgLocal; el < NVBLK_SUPER_ELEMENTS; el += MOE_DOWN_SG) {
                        const unsigned char byte = qs[el >> 1];
                        const unsigned nib = (el & 1) ? (byte >> 4) : (byte & 0x0F);
                        const float w = ((el < 16) ? s0 : s1) * dq_e2m1(nib);
                        sum = __fmaf_rn(xTile[xLocalBase + el], w, sum);
                    }
                }
            }

            __syncthreads();
        }

        accumSum += sum * ekw;
    }

    accumSum = warp16_reduce_sum(accumSum);
    if (active && sgLocal == 0) {
        accumB[n] += accumSum;   // read-modify-write
    }
}
