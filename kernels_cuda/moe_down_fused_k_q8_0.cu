// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
// Ported from kernels_hip/moe_down_fused_k_q8_0.hip — Track 4 mechanical port, no functional change intended.

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Fused MoE down-projection for T=1 decode — Q8_0 variant.
//
// Q8_0 = 34-byte blocks (fp16 scale + 32 int8 quants). Straight fp32
// accumulation is fine — per lane touches ~K/16 terms (~132 for
// ffPer=2112) which stays inside FP32's representable range with X
// as the dominant noise source.
//
//   accum[n] += sum_{k=0..K-1} kw[k] * sum_{l=0..ffPer-1}
//                                dequant_q8_0(W[expIdx[k]], n, l)
//                                * gateAct[k, l]
//
// This is the FIRST read-modify-write kernel in the HIP port: `accum`
// starts with existing content and gets summed INTO, not overwritten.
// Callers rely on this to fold multiple experts across dispatches or
// to combine with a residual stream in-place.
//
// Launch geometry matches matmul_q8_0_vec — WgSize=64, 4 sub-groups
// of 16 lanes mapped explicitly via tid/16 and tid%16.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifndef MOE_DOWN_LOCAL
#define MOE_DOWN_LOCAL 64
#endif

#ifndef MOE_DOWN_SG
#define MOE_DOWN_SG 16
#endif

#define MOE_DOWN_OUTPUTS_PER_GROUP (MOE_DOWN_LOCAL / MOE_DOWN_SG)

#define Q8_0_BLOCK_ELEMENTS 32
#define Q8_0_BLOCK_BYTES    34

// 1024 elements = 32 blocks = 4 KiB SLM per workgroup — matches
// matmul_q8_0_vec.hip.
#define X_TILE_ELEMENTS 1024

static __device__ __forceinline__ float warp16_reduce_sum(float v) {
    v += __shfl_xor(v, 8, 16);
    v += __shfl_xor(v, 4, 16);
    v += __shfl_xor(v, 2, 16);
    v += __shfl_xor(v, 1, 16);
    return v;
}

extern "C" __global__ __launch_bounds__(MOE_DOWN_LOCAL)
void moe_down_fused_k_q8_0(
    const float*         __restrict__ X,          // [K, ffPer]
    const unsigned char* __restrict__ W,          // Q8_0 expert bank base
    const int*           __restrict__ expIdx,     // [K]
    const float*         __restrict__ kw,         // [K] router × down-scale
          float*         __restrict__ accum,      // [dModel] read-modify-write
    const int                         ffPer,
    const int                         dModel,
    const int                         kActive,
    const int                         expertBytes)
{
    __shared__ float xTile[X_TILE_ELEMENTS];

    const int  wg       = blockIdx.x;
    const int  tid      = threadIdx.x;
    const int  lsize    = blockDim.x;
    const int  sgInWg   = tid / MOE_DOWN_SG;    // 0..3
    const int  sgLocal  = tid % MOE_DOWN_SG;    // 0..15
    const int  n        = wg * MOE_DOWN_OUTPUTS_PER_GROUP + sgInWg;
    const bool active   = (n < dModel);
    const int  nBlocks  = ffPer / Q8_0_BLOCK_ELEMENTS;
    const int  rowBytes = nBlocks * Q8_0_BLOCK_BYTES;

    // Outer accumulator across K active experts. Each lane sums its
    // quant-strided partials, scales by kw[k], folds into `accumSum`.
    // A single warp16_reduce_sum at the end sums across the 16 sub-
    // group lanes.
    float accumSum = 0.0f;

    for (int k = 0; k < kActive; ++k) {
        const int   e   = expIdx[k];
        const float ekw = kw[k];

        const unsigned char* __restrict__ Wexpert =
            W + static_cast<size_t>(e) * static_cast<size_t>(expertBytes);
        const float* __restrict__ Xk =
            X + static_cast<size_t>(k) * static_cast<size_t>(ffPer);

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

                const int blockStart   = tile / Q8_0_BLOCK_ELEMENTS;
                const int blocksInTile = X_TILE_ELEMENTS / Q8_0_BLOCK_ELEMENTS;
                const int blockEnd     = (blockStart + blocksInTile < nBlocks)
                                           ? (blockStart + blocksInTile)
                                           : nBlocks;

                for (int b = blockStart; b < blockEnd; ++b) {
                    const unsigned char* __restrict__ block =
                        row + b * Q8_0_BLOCK_BYTES;
                    const float d =
                        __half2float(*reinterpret_cast<const __half*>(block));
                    const signed char* qs =
                        reinterpret_cast<const signed char*>(block + 2);

                    const int xLocalBase = (b - blockStart) * Q8_0_BLOCK_ELEMENTS;

                    for (int l = sgLocal; l < Q8_0_BLOCK_ELEMENTS;
                         l += MOE_DOWN_SG) {
                        sum = __fmaf_rn(xTile[xLocalBase + l],
                                        d * static_cast<float>(qs[l]),
                                        sum);
                    }
                }
            }

            __syncthreads();
        }

        // Scale this expert's lane-partial by kw[k] and fold into the
        // outer accumulator.
        accumSum = __fmaf_rn(sum, ekw, accumSum);
    }

    accumSum = warp16_reduce_sum(accumSum);

    if (active && sgLocal == 0) {
        accum[n] += accumSum;   // read-modify-write
    }
}