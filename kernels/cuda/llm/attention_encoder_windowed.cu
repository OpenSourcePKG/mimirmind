// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Non-causal LOCAL (sliding-window) multi-head self-attention for ModernBERT's
// sliding_attention layers (the Laya backbone). 1:1 copy of attention_encoder.cu
// but a query at position pq attends only to keys kk with |kk - pq| <= window
// (window = local_attention/2 = 64 for ModernBERT). window <= 0 would attend to
// all keys (== the full-window kernel); callers pass window > 0 for local layers.
//
// Layout (row-major fp32): q/out [T, nHeads, headDim], k/v [T, nKvHeads, headDim].
// Workgroup geometry: grid( nHeads, T, 1 ), block( ATTN_LOCAL, 1, 1 ).

#include <cuda_runtime.h>

#include <math.h>   // INFINITY

#ifndef ATTN_LOCAL
#define ATTN_LOCAL 16
#endif

#ifndef ATTN_MAX_TK
#define ATTN_MAX_TK 16384
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

extern "C" __global__ __launch_bounds__(ATTN_LOCAL)
void attention_encoder_windowed(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
          float* __restrict__ out,
    const int                 T_k,
    const int                 nHeads,
    const int                 nKvHeads,
    const int                 headDim,
    const float               scale,
    const int                 window)
{
    const int hq  = blockIdx.x;
    const int pq  = blockIdx.y;
    const int lid = threadIdx.x;

    const int qStride  = nHeads   * headDim;
    const int kvStride = nKvHeads * headDim;
    const int hkv      = (hq * nKvHeads) / nHeads;

    const float* __restrict__ qVec =
        q + static_cast<size_t>(pq) * static_cast<size_t>(qStride)
          + static_cast<size_t>(hq) * static_cast<size_t>(headDim);
          float* __restrict__ oVec =
        out + static_cast<size_t>(pq) * static_cast<size_t>(qStride)
            + static_cast<size_t>(hq) * static_cast<size_t>(headDim);

    __shared__ float scores[ATTN_MAX_TK];

    // Pass 1 — Q·K, scaled; keys outside the window get -inf (0 weight).
    for (int kk = lid; kk < T_k; kk += ATTN_LOCAL) {
        if (window > 0 && (kk - pq > window || pq - kk > window)) {
            scores[kk] = -INFINITY;
            continue;
        }
        const float* __restrict__ kVec =
            k + static_cast<size_t>(kk) * static_cast<size_t>(kvStride)
              + static_cast<size_t>(hkv) * static_cast<size_t>(headDim);
        float acc = 0.0f;
        for (int d = 0; d < headDim; ++d) {
            acc += qVec[d] * kVec[d];
        }
        scores[kk] = acc * scale;
    }
    __syncthreads();

    // Pass 2 — stable softmax (masked keys contribute exp(-inf)=0).
    float mPart = -INFINITY;
    for (int kk = lid; kk < T_k; kk += ATTN_LOCAL) {
        const float s = scores[kk];
        if (s > mPart) mPart = s;
    }
    const float maxScore = warp16_reduce_max(mPart);

    float lPart = 0.0f;
    for (int kk = lid; kk < T_k; kk += ATTN_LOCAL) {
        const float e = expf(scores[kk] - maxScore);
        scores[kk] = e;
        lPart += e;
    }
    const float sumExp = warp16_reduce_sum(lPart);
    const float invSum = 1.0f / sumExp;
    __syncthreads();

    // Pass 3 — weighted sum over V.
    for (int d = lid; d < headDim; d += ATTN_LOCAL) {
        float acc = 0.0f;
        for (int kk = 0; kk < T_k; ++kk) {
            const float* __restrict__ vVec =
                v + static_cast<size_t>(kk) * static_cast<size_t>(kvStride)
                  + static_cast<size_t>(hkv) * static_cast<size_t>(headDim);
            acc += scores[kk] * vVec[d];
        }
        oVec[d] = acc * invSum;
    }
}
