// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Batched fused MoE down-projection — FP4-tensor-core weight format
// (M-Cuda.MoeGroup Sub-Step E-d.5). Same warp16 layout as the blocked-NVFP4
// variant (each 16-lane sub-group computes one dModel output), but reads the
// shared TC banks instead of the 20-byte super bank:
//   - plain E2M1 nibbles [nExperts, dModel, ffPer/2] (row (e,n) = nib + (e*dModel+n)*(ffPer/2))
//   - swizzled UE4M3 SFB per expert slot (one scale / 16 elems over ffPer)
//   - per-expert F32 global (folded into the per-expert sum before * kw)
//
// value[e,n,k] = global[e] * ue4m3(SFB[swizzle(n, k/16)]) * e2m1(nibble)
//
// Launch: grid = dim3(ceil(dModel / OUTPUTS_PER_GROUP), nSeq, 1),
//         block = MOE_DOWN_LOCAL.

#include <cuda_runtime.h>

#ifndef MOE_DOWN_TC_LOCAL
#define MOE_DOWN_TC_LOCAL 64
#endif
#ifndef MOE_DOWN_TC_SG
#define MOE_DOWN_TC_SG 16
#endif

#define MOE_DOWN_TC_OUTPUTS_PER_GROUP (MOE_DOWN_TC_LOCAL / MOE_DOWN_TC_SG)
#define TC_CHUNK_ELEMENTS 32
#define X_TILE_ELEMENTS   1024

static __device__ __forceinline__ float warp16_reduce_sum(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 8, 16);
    v += __shfl_xor_sync(0xffffffffu, v, 4, 16);
    v += __shfl_xor_sync(0xffffffffu, v, 2, 16);
    v += __shfl_xor_sync(0xffffffffu, v, 1, 16);
    return v;
}

static __device__ __forceinline__ float dq_e2m1(unsigned nib) {
    const unsigned mm   = nib & 0x7u;
    const unsigned bits = (((mm >> 1) + 126u) << 23) | ((mm & 1u) << 22);
    float v = __uint_as_float(bits);
    v = (mm < 2u) ? (0.5f * static_cast<float>(mm)) : v;
    return (nib & 0x8u) ? -v : v;
}

static __device__ __forceinline__ float dq_ue4m3(unsigned char b) {
    const unsigned e = (b >> 3) & 0xFu;
    const unsigned m = b & 0x7u;
    if (e == 0u) return ldexpf(static_cast<float>(m) * (1.0f / 8.0f), -6);
    return ldexpf(1.0f + static_cast<float>(m) * (1.0f / 8.0f), static_cast<int>(e) - 7);
}

static __device__ __forceinline__ long tcSfOffset(int row, int blk, int ksfTiles) {
    const int  a_m = row & 127;
    const long b_m = row >> 7;
    const int  a_s = blk & 3;
    const long b_s = blk >> 2;
    return static_cast<long>((a_m & 31) * 16 + (a_m >> 5) * 4 + a_s)
         + 512L * (b_s + static_cast<long>(ksfTiles) * b_m);
}

extern "C" __global__ __launch_bounds__(MOE_DOWN_TC_LOCAL)
void moe_down_fused_k_nvfp4tc_batched(
    const float*         __restrict__ X,        // [nSeq, K, ffPer]
    const unsigned char* __restrict__ WNib,     // nibbles [nExperts, dModel, ffPer/2]
    const unsigned char* __restrict__ WSfb,     // swizzled SFB (per-expert slots)
    const float*         __restrict__ WGlobal,  // [nExperts] down global
    const int*           __restrict__ expIdx,   // [nSeq, K]
    const float*         __restrict__ kw,       // [nSeq, K]
          float*         __restrict__ accum,    // [nSeq, dModel] RMW
    const int                         ffPer,
    const int                         dModel,
    const int                         kActive,
    const int                         sfbStride) // swizzled bytes per expert
{
    __shared__ float xTile[X_TILE_ELEMENTS];

    const int seq = blockIdx.y;
    const float* __restrict__ Xb      = X      + static_cast<size_t>(seq) * kActive * ffPer;
    const int*   __restrict__ expIdxB = expIdx + static_cast<size_t>(seq) * kActive;
    const float* __restrict__ kwB     = kw     + static_cast<size_t>(seq) * kActive;
    float*       __restrict__ accumB  = accum  + static_cast<size_t>(seq) * dModel;

    const int  wg      = blockIdx.x;
    const int  tid     = threadIdx.x;
    const int  lsize   = blockDim.x;
    const int  sgInWg  = tid / MOE_DOWN_TC_SG;
    const int  sgLocal = tid % MOE_DOWN_TC_SG;
    const int  n       = wg * MOE_DOWN_TC_OUTPUTS_PER_GROUP + sgInWg;
    const bool active  = (n < dModel);
    const int  nChunks  = ffPer / TC_CHUNK_ELEMENTS;
    const int  rowNib   = ffPer / 2;
    const int  ksfTiles = ((ffPer / 16) + 3) / 4;

    float accumSum = 0.0f;

    for (int k = 0; k < kActive; ++k) {
        const int   e   = expIdxB[k];
        const float ekw = kwB[k];
        const unsigned char* __restrict__ nibExp =
            WNib + (static_cast<size_t>(e) * dModel) * rowNib;
        const unsigned char* __restrict__ sfbExp =
            WSfb + static_cast<size_t>(e) * sfbStride;
        const float* __restrict__ Xk = Xb + static_cast<size_t>(k) * ffPer;

        float sum = 0.0f;
        for (int tile = 0; tile < ffPer; tile += X_TILE_ELEMENTS) {
            const int tileK = (X_TILE_ELEMENTS < ffPer - tile)
                                ? X_TILE_ELEMENTS : (ffPer - tile);
            for (int i = tid; i < tileK; i += lsize) xTile[i] = Xk[tile + i];
            __syncthreads();

            if (active) {
                const unsigned char* __restrict__ nibRow =
                    nibExp + static_cast<size_t>(n) * rowNib;
                const int spStart  = tile / TC_CHUNK_ELEMENTS;
                const int spInTile = X_TILE_ELEMENTS / TC_CHUNK_ELEMENTS;
                const int spEnd    = (spStart + spInTile < nChunks)
                                       ? (spStart + spInTile) : nChunks;
                for (int sp = spStart; sp < spEnd; ++sp) {
                    const int xLocalBase = (sp - spStart) * TC_CHUNK_ELEMENTS;
                    for (int el = sgLocal; el < TC_CHUNK_ELEMENTS; el += MOE_DOWN_TC_SG) {
                        const int kIdx = sp * TC_CHUNK_ELEMENTS + el;
                        const unsigned char byte = nibRow[kIdx >> 1];
                        const unsigned nib = (el & 1) ? (byte >> 4) : (byte & 0x0Fu);
                        const float scale =
                            dq_ue4m3(sfbExp[tcSfOffset(n, kIdx >> 4, ksfTiles)]);
                        const float w = scale * dq_e2m1(nib);
                        sum = __fmaf_rn(xTile[xLocalBase + el], w, sum);
                    }
                }
            }
            __syncthreads();
        }
        accumSum += sum * WGlobal[e] * ekw;   // fold per-expert global, then router weight
    }

    accumSum = warp16_reduce_sum(accumSum);
    if (active && sgLocal == 0) accumB[n] += accumSum;
}
