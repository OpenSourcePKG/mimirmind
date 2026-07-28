// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Batched fused MoE gate+up projection (BF16) — one launch for all nSeq
// decode tokens, each with its own routed-expert list. BF16 analogue of
// moe_gate_up_fused_k_q4k_batched (no dequant — dense BF16 weights).
//
//   gateActOut[s, k, f] = silu( Wg[e_{s,k}][f,:] . x[s] ) * ( Wu[e_{s,k}][f,:] . x[s] )
//
//   X          : [nSeq, dModel]        F32           seqStride = dModel
//   expIdx     : [nSeq, kActive]       int32         seqStride = kActive
//   gateActOut : [nSeq, kActive, nFf]  F32           seqStride = kActive*nFf
//   Wg / Wu    : [nExperts, nFf, dModel] BF16 expert banks (shared)
//                per-expert element stride = nFf*dModel, per-row = dModel
//
// One warp per output (k,f). The nSeq tokens' K experts are all processed
// in ONE launch (grid.y = seq), replacing the nSeq*K*2 per-token GEMV
// launches of the runMoeFfn CPU-fallback-era path -> cuts launch overhead
// for the 256-expert qwen35moe MoE. Reads each expert weight row once per
// (seq,k,f); the routed reads themselves stay memory-bound.
//
// Launch: grid = dim3(ceil(kActive*nFf / OUTPUTS_PER_GROUP), nSeq, 1),
//         block = MOE_GU_BF16_LOCAL.

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#ifndef MOE_GU_BF16_LOCAL
#define MOE_GU_BF16_LOCAL 128
#endif

#define MOE_GU_BF16_WARPS             (MOE_GU_BF16_LOCAL / 32)
#define MOE_GU_BF16_OUTPUTS_PER_GROUP MOE_GU_BF16_WARPS
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

extern "C" __global__ __launch_bounds__(MOE_GU_BF16_LOCAL)
void moe_gate_up_fused_k_bf16_batched(
    const float*         __restrict__ X,           // [nSeq, dModel]
    const __nv_bfloat16* __restrict__ Wg,          // [nExperts, nFf, dModel]
    const __nv_bfloat16* __restrict__ Wu,          // [nExperts, nFf, dModel]
    const int*           __restrict__ expIdx,      // [nSeq, kActive]
          float*         __restrict__ gateActOut,  // [nSeq, kActive, nFf]
    const int                         dModel,
    const int                         nFf,
    const int                         kActive)
{
    __shared__ float xTile[X_TILE_ELEMENTS];

    const int seq = blockIdx.y;
    const float* __restrict__ Xseq          = X      + static_cast<size_t>(seq) * dModel;
    const int*   __restrict__ expIdxSeq     = expIdx + static_cast<size_t>(seq) * kActive;
    float*       __restrict__ gateActOutSeq =
        gateActOut + static_cast<size_t>(seq) * kActive * nFf;

    const int wg     = blockIdx.x;
    const int tid    = threadIdx.x;
    const int lsize  = blockDim.x;
    const int warpId = tid / 32;
    const int laneId = tid % 32;
    const int o      = wg * MOE_GU_BF16_OUTPUTS_PER_GROUP + warpId;
    const bool active = (o < kActive * nFf);
    const int k = active ? (o / nFf) : 0;
    const int f = active ? (o % nFf) : 0;

    const __nv_bfloat16* __restrict__ gateRow = nullptr;
    const __nv_bfloat16* __restrict__ upRow   = nullptr;
    if (active) {
        const int e = expIdxSeq[k];
        const size_t expStride = static_cast<size_t>(nFf) * dModel;
        const size_t rowOff    = static_cast<size_t>(e) * expStride
                               + static_cast<size_t>(f) * dModel;
        gateRow = Wg + rowOff;
        upRow   = Wu + rowOff;
    }

    float gsum = 0.0f;
    float usum = 0.0f;

    for (int tile = 0; tile < dModel; tile += X_TILE_ELEMENTS) {
        const int tileK = (X_TILE_ELEMENTS < dModel - tile)
                            ? X_TILE_ELEMENTS : (dModel - tile);
        for (int i = tid; i < tileK; i += lsize) {
            xTile[i] = Xseq[tile + i];
        }
        __syncthreads();

        if (active) {
            for (int i = laneId; i < tileK; i += 32) {
                const float xv = xTile[i];
                gsum = __fmaf_rn(xv, __bfloat162float(gateRow[tile + i]), gsum);
                usum = __fmaf_rn(xv, __bfloat162float(upRow[tile + i]),   usum);
            }
        }
        __syncthreads();
    }

    gsum = warpReduceSum(gsum);
    usum = warpReduceSum(usum);

    if (active && laneId == 0) {
        const float silu = gsum / (1.0f + __expf(-gsum));
        gateActOutSeq[static_cast<size_t>(k) * nFf + f] = silu * usum;
    }
}
