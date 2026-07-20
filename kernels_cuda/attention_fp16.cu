// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
// Ported from kernels_hip/attention_fp16.hip — Track 4 mechanical port, no functional change intended.

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// FP16-KV variant of attention.hip. K and V are __half; Q and OUT
// stay fp32. Loads promote fp16 → fp32 in registers via __half2float,
// all reductions and the softmax remain fp32-precision.
//
// Structurally byte-for-byte the f32 variant apart from load sites —
// the KV-dtype swap is intentionally isolated so the port pair stays
// easy to diff against.
//
// Invoked only when the KV cache is fp16-typed; f32 caches continue
// to use attention.hip.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <math.h>   // for INFINITY

#ifndef ATTN_LOCAL
#define ATTN_LOCAL 16
#endif

#ifndef ATTN_SG
#define ATTN_SG 16
#endif

#ifndef ATTN_MAX_TK
#define ATTN_MAX_TK 16384
#endif

static __device__ __forceinline__ float warp16_reduce_max(float v) {
    v = fmaxf(v, __shfl_xor(v, 8, 16));
    v = fmaxf(v, __shfl_xor(v, 4, 16));
    v = fmaxf(v, __shfl_xor(v, 2, 16));
    v = fmaxf(v, __shfl_xor(v, 1, 16));
    return v;
}

static __device__ __forceinline__ float warp16_reduce_sum(float v) {
    v += __shfl_xor(v, 8, 16);
    v += __shfl_xor(v, 4, 16);
    v += __shfl_xor(v, 2, 16);
    v += __shfl_xor(v, 1, 16);
    return v;
}

extern "C" __global__ __launch_bounds__(ATTN_LOCAL)
void attention_fp16(
    const float*  __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
          float*  __restrict__ out,
    const int                  T_q,
    const int                  nHeads,
    const int                  nKvHeads,
    const int                  headDim,
    const int*    __restrict__ curLenPtr,
    const float                scale,
    const int                  slidingWindow)
{
    (void)T_q;

    const int hq  = blockIdx.x;
    const int pq  = blockIdx.y;
    const int lid = threadIdx.x;

    const int qStride        = nHeads   * headDim;
    const int kvStride       = nKvHeads * headDim;
    const int hkv            = (hq * nKvHeads) / nHeads;
    const int positionOffset = curLenPtr[0];
    const int absPos         = positionOffset + pq;
    const int kMax           = absPos + 1;
    const int kMin           = (slidingWindow > 0 && kMax > slidingWindow)
                                 ? (kMax - slidingWindow) : 0;

    const float* __restrict__ qVec =
        q + static_cast<size_t>(pq) * static_cast<size_t>(qStride)
          + static_cast<size_t>(hq) * static_cast<size_t>(headDim);
          float* __restrict__ oVec =
        out + static_cast<size_t>(pq) * static_cast<size_t>(qStride)
            + static_cast<size_t>(hq) * static_cast<size_t>(headDim);

    __shared__ float scores[ATTN_MAX_TK];

    // -- Pass 1 — Q·K dot products, scaled. K promotes fp16 → fp32. ---
    for (int kk = kMin + lid; kk < kMax; kk += ATTN_LOCAL) {
        const __half* __restrict__ kVec =
            k + static_cast<size_t>(kk) * static_cast<size_t>(kvStride)
              + static_cast<size_t>(hkv) * static_cast<size_t>(headDim);
        float acc = 0.0f;
        for (int d = 0; d < headDim; ++d) {
            acc += qVec[d] * __half2float(kVec[d]);
        }
        scores[kk - kMin] = acc * scale;
    }
    __syncthreads();

    // -- Pass 2 — stable softmax over [kMin, kMax). -------------------
    float mPart = -INFINITY;
    for (int kk = kMin + lid; kk < kMax; kk += ATTN_LOCAL) {
        const float s = scores[kk - kMin];
        if (s > mPart) mPart = s;
    }
    const float maxScore = warp16_reduce_max(mPart);

    float lPart = 0.0f;
    for (int kk = kMin + lid; kk < kMax; kk += ATTN_LOCAL) {
        const float e = expf(scores[kk - kMin] - maxScore);
        scores[kk - kMin] = e;
        lPart += e;
    }
    const float sumExp = warp16_reduce_sum(lPart);
    const float invSum = 1.0f / sumExp;
    __syncthreads();

    // -- Pass 3 — out[d] = sum_kk softmax_unnorm[kk] * v[kk][d] / sum.
    // V promotes fp16 → fp32 per read.
    for (int d = lid; d < headDim; d += ATTN_LOCAL) {
        float acc = 0.0f;
        for (int kk = kMin; kk < kMax; ++kk) {
            const __half* __restrict__ vVec =
                v + static_cast<size_t>(kk) * static_cast<size_t>(kvStride)
                  + static_cast<size_t>(hkv) * static_cast<size_t>(headDim);
            acc += scores[kk - kMin] * __half2float(vVec[d]);
        }
        oVec[d] = acc * invSum;
    }
}