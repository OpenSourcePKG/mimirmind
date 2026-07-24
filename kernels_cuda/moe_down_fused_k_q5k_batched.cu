// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Batched fused MoE down-projection (Q5_K) — M-Cuda.Batch batched variant
// of moe_down_fused_k_q5k. Processes nSeq decode tokens, each with its own
// gate activations, routed-expert list, router weights and output
// accumulator, in ONE launch. The Q5_K expert bank is shared. Math per
// (seq, n) is byte-identical to the single-token kernel (same K-loop,
// Kahan folds, warp16 reduce, read-modify-write); only a per-sequence
// offset (blockIdx.y) is added to X / expIdx / kw / accum. Cat B of the
// hybrid batch-dim audit 2026-07-24 (per-token MoE routing).
//
// Layout (per-sequence strides derive from ffPer,dModel,kActive):
//   X (gateAct) : [nSeq, kActive, ffPer]  seqStride = kActive*ffPer
//   expIdx      : [nSeq, kActive]         seqStride = kActive
//   kw          : [nSeq, kActive]         seqStride = kActive
//   accum       : [nSeq, dModel]          seqStride = dModel (read-modify-write)
//   W           : Q5_K expert bank, shared across sequences
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

#define Q5K_BLOCK_ELEMENTS 256
#define Q5K_BLOCK_BYTES    176

#define X_TILE_ELEMENTS 1024

static __device__ __forceinline__ float warp16_reduce_sum(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 8, 16);
    v += __shfl_xor_sync(0xffffffffu, v, 4, 16);
    v += __shfl_xor_sync(0xffffffffu, v, 2, 16);
    v += __shfl_xor_sync(0xffffffffu, v, 1, 16);
    return v;
}

static __device__ __forceinline__ void getScaleMinK4(
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

extern "C" __global__ __launch_bounds__(MOE_DOWN_LOCAL)
void moe_down_fused_k_q5k_batched(
    const float*         __restrict__ X,          // [nSeq, K, ffPer]
    const unsigned char* __restrict__ W,          // Q5_K expert bank (shared)
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
    const int  sgInWg   = tid / MOE_DOWN_SG;    // 0..3
    const int  sgLocal  = tid % MOE_DOWN_SG;    // 0..15
    const int  n        = wg * MOE_DOWN_OUTPUTS_PER_GROUP + sgInWg;
    const bool active   = (n < dModel);
    const int  nSuper   = ffPer / Q5K_BLOCK_ELEMENTS;
    const int  rowBytes = nSuper * Q5K_BLOCK_BYTES;

    float          accumSum = 0.0f;
    volatile float accumKc  = 0.0f;

    for (int k = 0; k < kActive; ++k) {
        const int   e   = expIdxB[k];
        const float ekw = kwB[k];

        const unsigned char* __restrict__ Wexpert =
            W + static_cast<size_t>(e) * static_cast<size_t>(expertBytes);
        const float* __restrict__ Xk =
            Xb + static_cast<size_t>(k) * static_cast<size_t>(ffPer);

        float          sum = 0.0f;
        volatile float kc  = 0.0f;

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

                const int sbStart  = tile / Q5K_BLOCK_ELEMENTS;
                const int sbInTile = X_TILE_ELEMENTS / Q5K_BLOCK_ELEMENTS;
                const int sbEnd    = (sbStart + sbInTile < nSuper)
                                       ? (sbStart + sbInTile)
                                       : nSuper;

                for (int sb = sbStart; sb < sbEnd; ++sb) {
                    const unsigned char* __restrict__ block =
                        row + sb * Q5K_BLOCK_BYTES;

                    const __half* d_ptr    = reinterpret_cast<const __half*>(block);
                    const __half* dmin_ptr = reinterpret_cast<const __half*>(block + 2);
                    const float d    = __half2float(d_ptr[0]);
                    const float dmin = __half2float(dmin_ptr[0]);

                    const unsigned char* scales = block + 4;
                    const unsigned char* qh     = block + 16;
                    const unsigned char* qs     = block + 48;

                    const int xLocalBase = (sb - sbStart) * Q5K_BLOCK_ELEMENTS;

                    for (int l = sgLocal; l < 32; l += MOE_DOWN_SG) {
                        const unsigned int qhByte = qh[l];

                        #pragma unroll
                        for (int pair = 0; pair < 4; ++pair) {
                            const int jLo = 2 * pair;
                            const int jHi = jLo + 1;
                            const unsigned int u1 = 1u << jLo;
                            const unsigned int u2 = 1u << jHi;

                            unsigned int sLo, mLo, sHi, mHi;
                            getScaleMinK4(jLo, scales, sLo, mLo);
                            getScaleMinK4(jHi, scales, sHi, mHi);

                            const float dLo  = d    * static_cast<float>(sLo);
                            const float mmLo = dmin * static_cast<float>(mLo);
                            const float dHi  = d    * static_cast<float>(sHi);
                            const float mmHi = dmin * static_cast<float>(mHi);

                            const unsigned char qb = qs[pair * 32 + l];
                            const float qLo = static_cast<float>(
                                (qb & 0x0Fu) + ((qhByte & u1) ? 16u : 0u));
                            const float qHi = static_cast<float>(
                                (qb >> 4)    + ((qhByte & u2) ? 16u : 0u));

                            const float wLo = dLo * qLo - mmLo;
                            const float wHi = dHi * qHi - mmHi;

                            #define KAHAN_ADD(dest, comp, term)                 \
                                do {                                             \
                                    const float _y = (term) - (comp);            \
                                    const float _t = (dest) + _y;                \
                                    (comp) = (_t - (dest)) - _y;                 \
                                    (dest) = _t;                                 \
                                } while (0)

                            KAHAN_ADD(sum, kc,
                                xTile[xLocalBase + jLo * 32 + l] * wLo);
                            KAHAN_ADD(sum, kc,
                                xTile[xLocalBase + jHi * 32 + l] * wHi);

                            #undef KAHAN_ADD
                        }
                    }
                }
            }

            __syncthreads();
        }

        const float lanePartial = (sum + kc) * ekw;
        const float _y = lanePartial - accumKc;
        const float _t = accumSum + _y;
        accumKc  = (_t - accumSum) - _y;
        accumSum = _t;
    }

    accumSum += accumKc;
    accumSum = warp16_reduce_sum(accumSum);

    if (active && sgLocal == 0) {
        accumB[n] += accumSum;   // read-modify-write
    }
}
