// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Dynamic activation quantiser: F32 activations -> NVFP4 (E2M1 nibbles) +
// per-16 UE4M3 block scales in the Blackwell-swizzled scale-factor layout that
// the CUTLASS block-scaled tensor-core grouped GEMM consumes (M-Cuda.MoeGroup
// Sub-Step E-d, the FP4-TC path). This turns the W4A16 activation stream into
// the W4A4 operand the block_scale mma needs — the step that finally makes the
// grouped MoE GEMM compute-dense enough to beat the batched fused-K path on
// GB10. Mirrors vLLM's cvt_warp_fp16_to_fp4 recipe (nvfp4_utils.cuh), adapted
// to F32 input and mimirmind's kernel idiom.
//
//   in:    [M, K]         F32 activations, row-major (K % 16 == 0)
//   out:   [M, K/2]       U8, 2 E2M1 nibbles / byte (elem 2j low, 2j+1 high)
//   SFout: swizzled       U8 UE4M3, one scale per 16-element block, laid out
//                         [numMTiles, numKTiles, 32, 4, 4] (see offset math
//                         below). The caller MUST cudaMemset SFout to 0 first:
//                         padding rows/cols (M -> round_up(M,128), K/16 ->
//                         round_up(.,4)) are not visited here but TMA reads them.
//   gscale: per-tensor activation global scale (SFScaleVal). Pass 1.0f when the
//           activations are not pre-calibrated (W4A16 checkpoints); the matching
//           GEMM epilogue alpha is then 1/(gscale * weight_gscale).
//
// value reconstruction (what the mma computes): a_e2m1 * ue4m3(SF) == true_a *
// gscale, because SF = e4m3(gscale * blockAbsmax / 6) and the nibble stores
// true_a * gscale / SF. The GEMM's per-expert alpha undoes gscale*weight_gscale.
//
// Software E2M1 (round-to-nearest-even, saturate to 6.0): required anyway — our
// PTX baseline is compute_80, which has no cvt.rn.satfinite.e2m1x2.f32 (that is
// an sm_100+ instruction), and GB10/sm_121a lacks it in hardware too (vLLM PR
// #35947). One code path serves both.
//
// Launch: grid( M, ceil((K/16) / ACT_QUANT_GROUPS_PER_BLOCK), 1 ),
//         block( ACT_QUANT_GROUPS_PER_BLOCK, 1, 1 ).

#include <cuda_runtime.h>
#include <cuda_fp8.h>

#ifndef ACT_QUANT_NVFP4_LOCAL
#define ACT_QUANT_NVFP4_LOCAL 256
#endif

#define ACT_QUANT_GROUPS_PER_BLOCK ACT_QUANT_NVFP4_LOCAL
#define NVFP4_SF_VEC_SIZE 16   // one block scale per 16 elements (NVFP4 group)

namespace {

// Software E2M1 encode of |v| already divided into the target grid, with the
// grid's sign applied. Grid = {0,.5,1,1.5,2,3,4,6}; codes 0..7; RNE ties resolve
// to the even code (matches cvt.rn.satfinite.e2m1x2.f32 hardware behaviour).
__device__ __forceinline__ unsigned aq_e2m1(float v) {
    const float a = fabsf(v);
    unsigned nib;
    if      (a <= 0.25f) nib = 0u;   // tie .25 -> 0.0 (code0 even)
    else if (a <  0.75f) nib = 1u;   // 0.5
    else if (a <= 1.25f) nib = 2u;   // tie .75 & 1.25 -> 1.0 (code2 even)
    else if (a <  1.75f) nib = 3u;   // 1.5
    else if (a <= 2.5f)  nib = 4u;   // tie 1.75 & 2.5 -> 2.0 (code4 even)
    else if (a <  3.5f)  nib = 5u;   // 3.0
    else if (a <= 5.0f)  nib = 6u;   // tie 3.5 & 5.0 -> 4.0 (code6 even)
    else                 nib = 7u;   // 6.0 (saturate)
    return nib | ((v < 0.0f) ? 0x8u : 0x0u);
}

// Swizzled scale-factor byte offset for logical (row = mIdx, block = kIdx),
// SF layout [numMTiles, numKTiles, 32, 4, 4]. Identical decomposition to vLLM's
// cvt_quant_to_fp4_get_sf_out_offset so CUTLASS's Sm1xxBlkScaledConfig reads it.
__device__ __forceinline__ long aq_sf_offset(int mIdx, int kIdx, int numKTiles) {
    const int mTileIdx  = mIdx >> 7;         // mIdx / 128
    const int outerMIdx = mIdx & 31;         // mIdx % 32
    const int innerMIdx = (mIdx >> 5) & 3;   // (mIdx / 32) % 4
    const int kTileIdx  = kIdx >> 2;         // kIdx / 4
    const int innerKIdx = kIdx & 3;          // kIdx % 4
    return ((static_cast<long>(mTileIdx) * numKTiles + kTileIdx) << 9)
         | (static_cast<long>(outerMIdx) << 4)
         | (static_cast<long>(innerMIdx) << 2)
         | static_cast<long>(innerKIdx);
}

} // namespace

extern "C" __global__ __launch_bounds__(ACT_QUANT_NVFP4_LOCAL)
void moe_act_quant_nvfp4(
    const float*         __restrict__ in,      // [M, K]
          unsigned char* __restrict__ out,     // [M, K/2] packed E2M1
          unsigned char* __restrict__ SFout,   // swizzled UE4M3 (pre-zeroed)
    const float                        gscale, // per-tensor activation scale
    const int                          M,
    const int                          K)      // multiple of 16
{
    const int nBlocks   = K / NVFP4_SF_VEC_SIZE;     // 16-elem blocks per row
    const int row       = blockIdx.x;                // one row of activations
    const int blk       = blockIdx.y * blockDim.x + threadIdx.x; // block index
    if (row >= M || blk >= nBlocks) {
        return;
    }

    // numKTiles pads the K-scale count up to a multiple of 4 (the innermost
    // swizzle dim); rounded col count / 4 tiles, matching the CUTLASS layout.
    const int numKTiles = (nBlocks + 3) / 4;

    const int k0 = blk * NVFP4_SF_VEC_SIZE;
    const float* __restrict__ src = in + static_cast<long>(row) * K + k0;

    // Block absmax over the 16 elements.
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < NVFP4_SF_VEC_SIZE; ++i) {
        amax = fmaxf(amax, fabsf(src[i]));
    }

    // SF = e4m3(gscale * amax / 6); round-trip through E4M3 so the encode uses
    // the exact stored scale the mma will multiply back in.
    float sfVal = gscale * (amax * (1.0f / 6.0f));
    const __nv_fp8_e4m3 sfE4m3 = __nv_fp8_e4m3(sfVal);
    sfVal = static_cast<float>(sfE4m3);

    // out = true_a * gscale / SF  (so a_e2m1 * SF == true_a * gscale).
    const float outScale = (sfVal != 0.0f) ? (gscale / sfVal) : 0.0f;

    // Pack 16 E2M1 nibbles into 8 bytes at [row, k0/2 ..].
    unsigned char* __restrict__ dst = out + (static_cast<long>(row) * K + k0) / 2;
#pragma unroll
    for (int j = 0; j < NVFP4_SF_VEC_SIZE / 2; ++j) {
        const unsigned lo = aq_e2m1(src[2 * j]     * outScale);
        const unsigned hi = aq_e2m1(src[2 * j + 1] * outScale);
        dst[j] = static_cast<unsigned char>(lo | (hi << 4));
    }

    // Store the block scale at its swizzled address.
    SFout[aq_sf_offset(row, blk, numKTiles)] =
        *reinterpret_cast<const unsigned char*>(&sfE4m3);
}

// Row-mapped variant: quantise only `nRows` real rows, each read from and
// written to the (padded) row `rowMap[blockIdx.x]`. The FP4-TC grouped MoE path
// pads every expert to 128 rows so its SFA sub-tensor is tile-aligned; the
// dense kernel above then quantises all `maxPad = R + nExperts*128` rows, of
// which only R are real (at decode M this is ~64x waste). Quantising just the
// real rows is bit-identical for those rows — the padding rows keep the
// pre-zeroed SF (scale 0 -> activation 0) and their GEMM output is discarded, so
// their nibble bytes never contribute. Same swizzle offset math, `row` sourced
// from the map instead of blockIdx.x.
extern "C" __global__ __launch_bounds__(ACT_QUANT_NVFP4_LOCAL)
void moe_act_quant_nvfp4_rows(
    const float*         __restrict__ in,      // [maxPad, K] (real rows scattered in)
          unsigned char* __restrict__ out,     // [maxPad, K/2] packed E2M1
          unsigned char* __restrict__ SFout,   // swizzled UE4M3 (pre-zeroed)
    const float                        gscale, // per-tensor activation scale
    const int*           __restrict__ rowMap,  // [nRows] padded row per real row
    const int                          nRows,
    const int                          K)      // multiple of 16
{
    const int nBlocks = K / NVFP4_SF_VEC_SIZE;
    const int logical = blockIdx.x;
    const int blk     = blockIdx.y * blockDim.x + threadIdx.x;
    if (logical >= nRows || blk >= nBlocks) {
        return;
    }
    const int row       = rowMap[logical];
    const int numKTiles = (nBlocks + 3) / 4;

    const int k0 = blk * NVFP4_SF_VEC_SIZE;
    const float* __restrict__ src = in + static_cast<long>(row) * K + k0;

    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < NVFP4_SF_VEC_SIZE; ++i) {
        amax = fmaxf(amax, fabsf(src[i]));
    }

    float sfVal = gscale * (amax * (1.0f / 6.0f));
    const __nv_fp8_e4m3 sfE4m3 = __nv_fp8_e4m3(sfVal);
    sfVal = static_cast<float>(sfE4m3);
    const float outScale = (sfVal != 0.0f) ? (gscale / sfVal) : 0.0f;

    unsigned char* __restrict__ dst = out + (static_cast<long>(row) * K + k0) / 2;
#pragma unroll
    for (int j = 0; j < NVFP4_SF_VEC_SIZE / 2; ++j) {
        const unsigned lo = aq_e2m1(src[2 * j]     * outScale);
        const unsigned hi = aq_e2m1(src[2 * j + 1] * outScale);
        dst[j] = static_cast<unsigned char>(lo | (hi << 4));
    }
    SFout[aq_sf_offset(row, blk, numKTiles)] =
        *reinterpret_cast<const unsigned char*>(&sfE4m3);
}
