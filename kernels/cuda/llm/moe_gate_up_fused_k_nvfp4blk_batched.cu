// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Batched fused MoE gate+up projection (blocked-NVFP4) — NVFP4 analogue of
// moe_gate_up_fused_k_q4k_batched. Keeps the routed experts native 4-bit
// (E2M1) and LOSSLESS (repackaged from the checkpoint's NVFP4, not re-quantised
// like the Q4_K path). Same warp layout (one warp per (k,f) output over
// kActive*nFf, 32 lanes span a 32-element super), silu(gate)*up.
//
// Weight bank row: nSuper = dModel/32 super-blocks of 20 bytes each
//   fp16 s0 | fp16 s1 | 16 packed E2M1 bytes  (2 NVFP4 blocks = 32 elems)
//   value[e in super] = (e<16 ? s0 : s1) * e2m1(nibble_e)
//
// Launch: grid = dim3(ceil(kActive*nFf / OUTPUTS_PER_GROUP), nSeq, 1),
//         block = MOE_GU_LOCAL.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifndef MOE_GU_LOCAL
#define MOE_GU_LOCAL 128
#endif

#define MOE_GU_WARPS             (MOE_GU_LOCAL / 32)
#define MOE_GU_OUTPUTS_PER_GROUP MOE_GU_WARPS

#define NVBLK_SUPER_ELEMENTS 32
#define NVBLK_SUPER_BYTES    20
#define X_TILE_ELEMENTS      1024

namespace {

__device__ __forceinline__ float warpReduceSum(float v) {
    v += __shfl_down_sync(0xffffffffu, v, 16);
    v += __shfl_down_sync(0xffffffffu, v,  8);
    v += __shfl_down_sync(0xffffffffu, v,  4);
    v += __shfl_down_sync(0xffffffffu, v,  2);
    v += __shfl_down_sync(0xffffffffu, v,  1);
    return v;
}

__device__ __forceinline__ float dq_e2m1(unsigned nib) {
    const float mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float v = mag[nib & 0x7u];
    return (nib & 0x8u) ? -v : v;
}

// Contribution of element `l` (lane) of one 32-element super to the dot product.
__device__ __forceinline__ float nvblkSuperDot(
    const unsigned char* super, const float* xBase, int l) {
    const float s0 = __half2float(*reinterpret_cast<const __half*>(super));
    const float s1 = __half2float(*reinterpret_cast<const __half*>(super + 2));
    const unsigned char byte = super[4 + (l >> 1)];
    const unsigned nib = (l & 1) ? (byte >> 4) : (byte & 0x0F);
    const float w = ((l < 16) ? s0 : s1) * dq_e2m1(nib);
    return xBase[l] * w;
}

} // namespace

extern "C" __global__ __launch_bounds__(MOE_GU_LOCAL)
void moe_gate_up_fused_k_nvfp4blk_batched(
    const float*         __restrict__ X,           // [nSeq, dModel]
    const unsigned char* __restrict__ Wg,          // gate expert bank (shared)
    const unsigned char* __restrict__ Wu,          // up expert bank (shared)
    const int*           __restrict__ expIdx,      // [nSeq, kActive]
          float*         __restrict__ gateActOut,  // [nSeq, kActive, nFf]
    const int                         dModel,
    const int                         nFf,
    const int                         kActive,
    const int                         expertBytesGate,
    const int                         expertBytesUp)
{
    __shared__ float xTile[X_TILE_ELEMENTS];

    const int seq = blockIdx.y;
    const float* Xseq          = X          + static_cast<size_t>(seq) * dModel;
    const int*   expIdxSeq     = expIdx     + static_cast<size_t>(seq) * kActive;
    float*       gateActOutSeq = gateActOut
        + static_cast<size_t>(seq) * kActive * nFf;

    const int wg      = blockIdx.x;
    const int tid     = threadIdx.x;
    const int lsize   = blockDim.x;
    const int warpId  = tid / 32;
    const int laneId  = tid % 32;
    const int o       = wg * MOE_GU_OUTPUTS_PER_GROUP + warpId;
    const bool active = (o < kActive * nFf);
    const int k       = active ? (o / nFf) : 0;
    const int f       = active ? (o % nFf) : 0;
    const int nSuper  = dModel / NVBLK_SUPER_ELEMENTS;
    const int rowBytes = nSuper * NVBLK_SUPER_BYTES;

    const unsigned char* gateRow = nullptr;
    const unsigned char* upRow   = nullptr;
    if (active) {
        const int e = expIdxSeq[k];
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
            xTile[i] = Xseq[tile + i];
        }
        __syncthreads();

        if (active) {
            const int superStart   = tile / NVBLK_SUPER_ELEMENTS;
            const int supersInTile  = X_TILE_ELEMENTS / NVBLK_SUPER_ELEMENTS;
            const int superEnd     = (superStart + supersInTile < nSuper)
                                       ? (superStart + supersInTile) : nSuper;
            for (int sp = superStart; sp < superEnd; ++sp) {
                const float* xBase = xTile + (sp - superStart) * NVBLK_SUPER_ELEMENTS;
                gsum += nvblkSuperDot(gateRow + sp * NVBLK_SUPER_BYTES, xBase, laneId);
                usum += nvblkSuperDot(upRow   + sp * NVBLK_SUPER_BYTES, xBase, laneId);
            }
        }

        __syncthreads();
    }

    gsum = warpReduceSum(gsum);
    usum = warpReduceSum(usum);

    if (active && laneId == 0) {
        const float g = gsum;
        const float silu = g / (1.0f + __expf(-g));
        gateActOutSeq[static_cast<size_t>(k) * nFf + f] = silu * usum;
    }
}
