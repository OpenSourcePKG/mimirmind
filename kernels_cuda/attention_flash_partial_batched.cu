// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Batched FlashAttention partial-tile kernel — decode mode (T_q == 1),
// M-Cuda.Batch variant of attention_flash_partial. Processes nSeq
// sequences, each with its own query, its own KV cache and its own
// current length, in ONE launch. Math per (seq, head, K-tile) is
// byte-identical to the single-sequence kernel; a per-sequence offset
// (blockIdx.z) is added to q / k / v / partialMlo, the position comes
// from curLenPtr[seq], and the per-head partial slab uses a UNIFORM
// kTilesStride (= batch-wide max K-tiles) so the layout is regular
// across sequences with different lengths.
//
// Provisional per-sequence strides (KV layout settled in Phase D):
//   q          [nSeq, nHeads, headDim]     qSeqStride
//   k / v      [nSeq, T_k, nKvHeads, headDim]  kvSeqStride
//   partialMlo [nSeq, nHeads, kTilesStride, (2+headDim)]  partialSeqStride
//   curLenPtr  [nSeq]
// Launch: grid = dim3(nHeads, kTilesStride, nSeq), block = ATTN_FLASH_LOCAL.

#include <cuda_runtime.h>
#include <math.h>   // for INFINITY

#ifndef ATTN_FLASH_LOCAL
#define ATTN_FLASH_LOCAL 16
#endif

#ifndef ATTN_FLASH_KTILE
#define ATTN_FLASH_KTILE 64
#endif

static __device__ __forceinline__ float warp16_reduce_max(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 8, 16));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 4, 16));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2, 16));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1, 16));
    return v;
}

static __device__ __forceinline__ float warp16_reduce_sum(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 8, 16);
    v += __shfl_xor_sync(0xffffffffu, v, 4, 16);
    v += __shfl_xor_sync(0xffffffffu, v, 2, 16);
    v += __shfl_xor_sync(0xffffffffu, v, 1, 16);
    return v;
}

extern "C" __global__ __launch_bounds__(ATTN_FLASH_LOCAL)
void attention_flash_partial_batched(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
          float* __restrict__ partialMlo,
    const int                 nHeads,
    const int                 nKvHeads,
    const int                 headDim,
    const int* __restrict__   curLenPtr,      // [nSeq]
    const float               scale,
    const int                 slidingWindow,
    const int                 kTilesStride,    // batch-wide max K-tiles
    const int                 qSeqStride,
    const int                 kvSeqStride,
    const int                 partialSeqStride)
{
    const int hq  = blockIdx.x;
    const int kt  = blockIdx.y;
    const int seq = blockIdx.z;
    const int lid = threadIdx.x;

    const float* __restrict__ qSeq = q + (size_t)seq * qSeqStride;
    const float* __restrict__ kSeq = k + (size_t)seq * kvSeqStride;
    const float* __restrict__ vSeq = v + (size_t)seq * kvSeqStride;
    float*       __restrict__ partialSeq = partialMlo + (size_t)seq * partialSeqStride;

    const int kvStride       = nKvHeads * headDim;
    const int hkv            = (hq * nKvHeads) / nHeads;
    const int positionOffset = curLenPtr[seq];

    const int kMax    = positionOffset + 1;
    const int kMin    = (slidingWindow > 0 && kMax > slidingWindow)
                          ? (kMax - slidingWindow) : 0;
    const int kStartRaw = kt * ATTN_FLASH_KTILE;
    const int kStart  = (kStartRaw > kMin) ? kStartRaw : kMin;
    const int kEndRaw = kStartRaw + ATTN_FLASH_KTILE;
    const int kEnd    = (kEndRaw < kMax) ? kEndRaw : kMax;

    // Uniform per-head stride so sequences of different length share a
    // regular layout; tiles beyond a sequence's own nKTiles just emit a
    // neutral partial that the merge kernel never reads.
    float* __restrict__ mloPtr =
        partialSeq + ((size_t)hq * (size_t)kTilesStride + (size_t)kt)
                   * (size_t)(2 + headDim);

    if (kStart >= kEnd) {
        if (lid == 0) {
            mloPtr[0] = -INFINITY;
            mloPtr[1] = 0.0f;
        }
        for (int d = lid; d < headDim; d += ATTN_FLASH_LOCAL) {
            mloPtr[2 + d] = 0.0f;
        }
        return;
    }

    const float* __restrict__ qVec = qSeq + (size_t)hq * (size_t)headDim;

    __shared__ float scores[ATTN_FLASH_KTILE];

    const int tileLen = kEnd - kStart;

    for (int kk = lid; kk < tileLen; kk += ATTN_FLASH_LOCAL) {
        const int absKk = kStart + kk;
        const float* __restrict__ kVec =
            kSeq + (size_t)absKk * (size_t)kvStride + (size_t)hkv * (size_t)headDim;
        float acc = 0.0f;
        for (int d = 0; d < headDim; ++d) {
            acc += qVec[d] * kVec[d];
        }
        scores[kk] = acc * scale;
    }
    __syncthreads();

    float mPart = -INFINITY;
    for (int kk = lid; kk < tileLen; kk += ATTN_FLASH_LOCAL) {
        const float s = scores[kk];
        if (s > mPart) mPart = s;
    }
    const float mLocal = warp16_reduce_max(mPart);

    float lPart = 0.0f;
    for (int kk = lid; kk < tileLen; kk += ATTN_FLASH_LOCAL) {
        const float e = expf(scores[kk] - mLocal);
        scores[kk] = e;
        lPart += e;
    }
    const float lLocal = warp16_reduce_sum(lPart);
    __syncthreads();

    for (int d = lid; d < headDim; d += ATTN_FLASH_LOCAL) {
        float acc = 0.0f;
        for (int kk = 0; kk < tileLen; ++kk) {
            const float* __restrict__ vVec =
                vSeq + (size_t)(kStart + kk) * (size_t)kvStride
                     + (size_t)hkv * (size_t)headDim;
            acc += scores[kk] * vVec[d];
        }
        mloPtr[2 + d] = acc;
    }

    if (lid == 0) {
        mloPtr[0] = mLocal;
        mloPtr[1] = lLocal;
    }
}
