// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Batched fused MoE gate+up projection — FP4-tensor-core weight format
// (M-Cuda.MoeGroup Sub-Step E-d.5). Same warp layout as the blocked-NVFP4
// variant (one warp per (k,f) output, 32 lanes span a 32-element chunk,
// silu(gate)*up), but reads the SAME banks the prefill grouped GEMM uses:
//   - plain E2M1 nibbles  [nExperts, nFf, dModel/2]  (row (e,f) = nib + (e*nFf+f)*(dModel/2))
//   - swizzled UE4M3 SFB  per expert slot (128x4 layout, one scale / 16 elems)
//   - per-expert F32 global (folded in once, after the reduce)
// so the decode path no longer needs the blocked-NVFP4 bank.
//
// value[e,f,k] = global[e] * ue4m3(SFB[swizzle(f, k/16)]) * e2m1(nibble)
//
// Launch: grid = dim3(ceil(kActive*nFf / OUTPUTS_PER_GROUP), nSeq, 1),
//         block = MOE_GU_LOCAL.

#include <cuda_runtime.h>

#ifndef MOE_GU_TC_LOCAL
#define MOE_GU_TC_LOCAL 128
#endif

#define MOE_GU_TC_WARPS             (MOE_GU_TC_LOCAL / 32)
#define MOE_GU_TC_OUTPUTS_PER_GROUP MOE_GU_TC_WARPS
#define TC_CHUNK_ELEMENTS 32
#define X_TILE_ELEMENTS   1024

namespace {

__device__ __forceinline__ float warpReduceSum(float v) {
    v += __shfl_down_sync(0xffffffffu, v, 16);
    v += __shfl_down_sync(0xffffffffu, v,  8);
    v += __shfl_down_sync(0xffffffffu, v,  4);
    v += __shfl_down_sync(0xffffffffu, v,  2);
    v += __shfl_down_sync(0xffffffffu, v,  1);
    return v;
}

// E2M1 magnitude decode (arithmetic, no local LUT). Bit-exact to
// {0,0.5,1,1.5,2,3,4,6}; bit 3 = sign.
__device__ __forceinline__ float dq_e2m1(unsigned nib) {
    const unsigned mm   = nib & 0x7u;
    const unsigned bits = (((mm >> 1) + 126u) << 23) | ((mm & 1u) << 22);
    float v = __uint_as_float(bits);
    v = (mm < 2u) ? (0.5f * static_cast<float>(mm)) : v;
    return (nib & 0x8u) ? -v : v;
}

// UE4M3 (= E4M3 with sign bit 0 for the non-negative block scales) decode,
// bias 7. Matches the act-quant kernel's stored scale and CUTLASS float_ue4m3_t.
__device__ __forceinline__ float dq_ue4m3(unsigned char b) {
    const unsigned e = (b >> 3) & 0xFu;
    const unsigned m = b & 0x7u;
    if (e == 0u) return ldexpf(static_cast<float>(m) * (1.0f / 8.0f), -6);
    return ldexpf(1.0f + static_cast<float>(m) * (1.0f / 8.0f), static_cast<int>(e) - 7);
}

// Swizzled SFB byte offset of scale-block `blk` for output row `row`
// (== core::modelopt::swizzledScaleOffset(row, blk, ksfTiles)).
__device__ __forceinline__ long tcSfOffset(int row, int blk, int ksfTiles) {
    const int  a_m = row & 127;
    const long b_m = row >> 7;
    const int  a_s = blk & 3;
    const long b_s = blk >> 2;
    return static_cast<long>((a_m & 31) * 16 + (a_m >> 5) * 4 + a_s)
         + 512L * (b_s + static_cast<long>(ksfTiles) * b_m);
}

// Dot contribution of element `l` (lane) of 32-element chunk `sp`, output row
// `f`, from the plain-nibble row + swizzled SFB slot (global applied later).
__device__ __forceinline__ float tcDot(
    const unsigned char* __restrict__ nibRow,
    const unsigned char* __restrict__ sfbSlot,
    int f, const float* __restrict__ xBase, int sp, int ksfTiles, int l) {
    const int kIdx = sp * TC_CHUNK_ELEMENTS + l;
    const unsigned char byte = nibRow[kIdx >> 1];
    const unsigned nib = (l & 1) ? (byte >> 4) : (byte & 0x0Fu);
    const float scale = dq_ue4m3(sfbSlot[tcSfOffset(f, kIdx >> 4, ksfTiles)]);
    return xBase[l] * (scale * dq_e2m1(nib));
}

} // namespace

extern "C" __global__ __launch_bounds__(MOE_GU_TC_LOCAL)
void moe_gate_up_fused_k_nvfp4tc_batched(
    const float*         __restrict__ X,           // [nSeq, dModel]
    const unsigned char* __restrict__ WgNib,       // gate nibbles [nExperts,nFf,dModel/2]
    const unsigned char* __restrict__ WuNib,       // up   nibbles
    const unsigned char* __restrict__ WgSfb,       // gate swizzled SFB (per-expert slots)
    const unsigned char* __restrict__ WuSfb,       // up   swizzled SFB
    const float*         __restrict__ WgGlobal,    // [nExperts] gate global
    const float*         __restrict__ WuGlobal,    // [nExperts] up   global
    const int*           __restrict__ expIdx,      // [nSeq, kActive]
          float*         __restrict__ gateActOut,  // [nSeq, kActive, nFf]
    const int                         dModel,
    const int                         nFf,
    const int                         kActive,
    const int                         sfbStride)   // swizzled bytes per expert
{
    __shared__ float xTile[X_TILE_ELEMENTS];

    const int seq = blockIdx.y;
    const float* Xseq          = X          + static_cast<size_t>(seq) * dModel;
    const int*   expIdxSeq      = expIdx     + static_cast<size_t>(seq) * kActive;
    float*       gateActOutSeq = gateActOut + static_cast<size_t>(seq) * kActive * nFf;

    const int wg     = blockIdx.x;
    const int tid    = threadIdx.x;
    const int lsize  = blockDim.x;
    const int warpId = tid / 32;
    const int laneId = tid % 32;
    const int o      = wg * MOE_GU_TC_OUTPUTS_PER_GROUP + warpId;
    const bool active = (o < kActive * nFf);
    const int k = active ? (o / nFf) : 0;
    const int f = active ? (o % nFf) : 0;
    const int nChunks  = dModel / TC_CHUNK_ELEMENTS;
    const int rowNib   = dModel / 2;                       // bytes per nibble row
    const int ksfTiles = ((dModel / 16) + 3) / 4;

    const unsigned char* gNib = nullptr; const unsigned char* uNib = nullptr;
    const unsigned char* gSfb = nullptr; const unsigned char* uSfb = nullptr;
    int e = 0;
    if (active) {
        e = expIdxSeq[k];
        const size_t rowOff = (static_cast<size_t>(e) * nFf + f) * rowNib;
        gNib = WgNib + rowOff;  uNib = WuNib + rowOff;
        gSfb = WgSfb + static_cast<size_t>(e) * sfbStride;
        uSfb = WuSfb + static_cast<size_t>(e) * sfbStride;
    }

    float gsum = 0.0f, usum = 0.0f;

    for (int tile = 0; tile < dModel; tile += X_TILE_ELEMENTS) {
        const int tileK = (X_TILE_ELEMENTS < dModel - tile)
                            ? X_TILE_ELEMENTS : (dModel - tile);
        for (int i = tid; i < tileK; i += lsize) xTile[i] = Xseq[tile + i];
        __syncthreads();

        if (active) {
            const int chunkStart = tile / TC_CHUNK_ELEMENTS;
            const int chunksInTile = X_TILE_ELEMENTS / TC_CHUNK_ELEMENTS;
            const int chunkEnd = (chunkStart + chunksInTile < nChunks)
                                   ? (chunkStart + chunksInTile) : nChunks;
            for (int sp = chunkStart; sp < chunkEnd; ++sp) {
                const float* xBase = xTile + (sp - chunkStart) * TC_CHUNK_ELEMENTS;
                gsum += tcDot(gNib, gSfb, f, xBase, sp, ksfTiles, laneId);
                usum += tcDot(uNib, uSfb, f, xBase, sp, ksfTiles, laneId);
            }
        }
        __syncthreads();
    }

    gsum = warpReduceSum(gsum);
    usum = warpReduceSum(usum);

    if (active && laneId == 0) {
        const float g = gsum * WgGlobal[e];              // fold per-expert global
        const float u = usum * WuGlobal[e];
        const float silu = g / (1.0f + __expf(-g));
        gateActOutSeq[static_cast<size_t>(k) * nFf + f] = silu * u;
    }
}
