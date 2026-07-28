// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Batched FlashAttention merge kernel — decode mode (T_q == 1),
// M-Cuda.Batch variant of attention_flash_merge. Folds the per-tile
// partials of nSeq sequences into their final outputs in ONE launch.
// Math per (seq, head) is byte-identical to the single-sequence kernel;
// a per-sequence offset (blockIdx.y) is added to partialMlo / out, the
// position comes from curLenPtr[seq], and the per-head partial slab uses
// the same UNIFORM kTilesStride the batched partial kernel wrote.
//
// Provisional per-sequence strides (settled in Phase D):
//   partialMlo [nSeq, nHeads, kTilesStride, (2+headDim)]  partialSeqStride
//   out        [nSeq, nHeads, headDim]                    outSeqStride
//   curLenPtr  [nSeq]
// Launch: grid = dim3(nHeads, nSeq, 1), block = ATTN_FLASH_LOCAL.

#include <cuda_runtime.h>
#include <math.h>   // for INFINITY

#ifndef ATTN_FLASH_LOCAL
#define ATTN_FLASH_LOCAL 16
#endif

#ifndef ATTN_FLASH_MAX_KTILES
#define ATTN_FLASH_MAX_KTILES 512
#endif

#ifndef ATTN_FLASH_KTILE
#define ATTN_FLASH_KTILE 64
#endif

extern "C" __global__ __launch_bounds__(ATTN_FLASH_LOCAL)
void attention_flash_merge_batched(
    const float* __restrict__ partialMlo,
          float* __restrict__ out,
    const int                 nHeads,
    const int                 headDim,
    const int* __restrict__   curLenPtr,       // [nSeq]
    const int                 kTilesStride,     // batch-wide max K-tiles
    const int                 partialSeqStride,
    const int                 outSeqStride)
{
    const int hq  = blockIdx.x;
    const int seq = blockIdx.y;
    const int lid = threadIdx.x;

    const int positionOffset = curLenPtr[seq];
    const int kMax           = positionOffset + 1;
    const int nKTiles        = (kMax + ATTN_FLASH_KTILE - 1) / ATTN_FLASH_KTILE;

    const int stride = 2 + headDim;
    const float* __restrict__ baseMlo =
        partialMlo + (size_t)seq * (size_t)partialSeqStride
                   + (size_t)hq  * (size_t)kTilesStride * (size_t)stride;

    __shared__ float alphas[ATTN_FLASH_MAX_KTILES];
    __shared__ float mFinalSlm;
    __shared__ float lFinalSlm;

    if (lid == 0) {
        float mFinal = -INFINITY;
        for (int t = 0; t < nKTiles; ++t) {
            const float mt = baseMlo[(size_t)t * (size_t)stride + 0];
            if (mt > mFinal) mFinal = mt;
        }
        if (mFinal == -INFINITY) mFinal = 0.0f;

        float lFinal = 0.0f;
        for (int t = 0; t < nKTiles; ++t) {
            const float mt = baseMlo[(size_t)t * (size_t)stride + 0];
            const float lt = baseMlo[(size_t)t * (size_t)stride + 1];
            const float beta = (mt == -INFINITY) ? 0.0f : expf(mt - mFinal);
            alphas[t] = beta;
            lFinal  += beta * lt;
        }

        mFinalSlm = mFinal;
        lFinalSlm = lFinal;
    }
    __syncthreads();

    const float lFinal = lFinalSlm;
    const float invL   = (lFinal > 0.0f) ? (1.0f / lFinal) : 0.0f;

    float* __restrict__ oOut =
        out + (size_t)seq * (size_t)outSeqStride + (size_t)hq * (size_t)headDim;

    for (int d = lid; d < headDim; d += ATTN_FLASH_LOCAL) {
        float acc = 0.0f;
        for (int t = 0; t < nKTiles; ++t) {
            const float ot_d = baseMlo[(size_t)t * (size_t)stride + 2 + d];
            acc += alphas[t] * ot_d;
        }
        oOut[d] = acc * invL;
    }
}
