// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Batched fused MoE down projection (BF16) — one launch for all nSeq decode
// tokens. BF16 analogue of moe_down_fused_k_q5k_batched (no dequant).
//
//   accum[s, n] += sum_{k} kw[s,k] * ( Wd[e_{s,k}][n,:] . gateAct[s,k,:] )
//
//   X (gateAct) : [nSeq, kActive, ffPer]  F32   seqStride = kActive*ffPer
//   W           : [nExperts, dModel, ffPer] BF16 expert bank (shared)
//                 per-expert element stride = dModel*ffPer, per-row = ffPer
//   expIdx      : [nSeq, kActive]  int32
//   kw          : [nSeq, kActive]  F32 (router weight * wScale)
//   accum       : [nSeq, dModel]   F32, read-modify-write (+=)
//
// One warp per output n; loops the kActive experts, dotting each expert's
// row Wd[e][n,:] against that expert's gateAct[s,k,:]. grid.y = seq so the
// whole batch is one launch. accum is +='d so the caller must pre-zero it
// (or add the shared-expert contribution afterwards, as runMoeFfn does).
//
// Launch: grid = dim3(ceil(dModel / OUTPUTS_PER_GROUP), nSeq, 1),
//         block = MOE_DOWN_BF16_LOCAL.

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#ifndef MOE_DOWN_BF16_LOCAL
#define MOE_DOWN_BF16_LOCAL 128
#endif

#define MOE_DOWN_BF16_WARPS             (MOE_DOWN_BF16_LOCAL / 32)
#define MOE_DOWN_BF16_OUTPUTS_PER_GROUP MOE_DOWN_BF16_WARPS
#define X_TILE_ELEMENTS 1024

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

extern "C" __global__ __launch_bounds__(MOE_DOWN_BF16_LOCAL)
void moe_down_fused_k_bf16_batched(
    const float*         __restrict__ X,        // gateAct [nSeq, kActive, ffPer]
    const __nv_bfloat16* __restrict__ W,        // [nExperts, dModel, ffPer]
    const int*           __restrict__ expIdx,   // [nSeq, kActive]
    const float*         __restrict__ kw,       // [nSeq, kActive]
          float*         __restrict__ accum,    // [nSeq, dModel] RMW
    const int                         ffPer,
    const int                         dModel,
    const int                         kActive)
{
    __shared__ float xTile[X_TILE_ELEMENTS];

    const int seq = blockIdx.y;
    const float* __restrict__ Xb      = X      + static_cast<size_t>(seq) * kActive * ffPer;
    const int*   __restrict__ expIdxB = expIdx + static_cast<size_t>(seq) * kActive;
    const float* __restrict__ kwB     = kw     + static_cast<size_t>(seq) * kActive;
    float*       __restrict__ accumB  = accum  + static_cast<size_t>(seq) * dModel;

    const int wg     = blockIdx.x;
    const int tid    = threadIdx.x;
    const int lsize  = blockDim.x;
    const int warpId = tid / 32;
    const int laneId = tid % 32;
    const int n      = wg * MOE_DOWN_BF16_OUTPUTS_PER_GROUP + warpId;
    const bool active = (n < dModel);

    const size_t expStride = static_cast<size_t>(dModel) * ffPer;

    float accumSum = 0.0f;

    for (int k = 0; k < kActive; ++k) {
        const int   e   = expIdxB[k];
        const float ekw = kwB[k];
        const float* __restrict__ Xk = Xb + static_cast<size_t>(k) * ffPer;
        const __nv_bfloat16* __restrict__ wRow = active
            ? (W + static_cast<size_t>(e) * expStride + static_cast<size_t>(n) * ffPer)
            : nullptr;

        float sum = 0.0f;
        for (int tile = 0; tile < ffPer; tile += X_TILE_ELEMENTS) {
            const int tileK = (X_TILE_ELEMENTS < ffPer - tile)
                                ? X_TILE_ELEMENTS : (ffPer - tile);
            for (int i = tid; i < tileK; i += lsize) {
                xTile[i] = Xk[tile + i];
            }
            __syncthreads();

            if (active) {
                for (int i = laneId; i < tileK; i += 32) {
                    sum = __fmaf_rn(xTile[i], __bfloat162float(wRow[tile + i]), sum);
                }
            }
            __syncthreads();
        }

        sum = warpReduceSum(sum);
        if (active && laneId == 0) {
            accumSum += sum * ekw;
        }
    }

    if (active && laneId == 0) {
        accumB[n] += accumSum;
    }
}
