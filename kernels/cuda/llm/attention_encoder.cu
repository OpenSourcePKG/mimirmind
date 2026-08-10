// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Non-causal (bidirectional) multi-head self-attention for the BERT /
// RoBERTa / XLM-R encoder (EncoderRunner / cross-encoder reranker path).
//
// 1:1 mirror of attention.cu (the untiled variant-a decoder attention) but
// EVERY query attends to ALL keys [0, T_k) — no causal mask, no sliding
// window, no positionOffset / KV-cache. Single full-sequence forward.
//
// Layout (all row-major fp32):
//   q:   [T_q, nHeads,   headDim]
//   k:   [T_k, nKvHeads, headDim]
//   v:   [T_k, nKvHeads, headDim]
//   out: [T_q, nHeads,   headDim]
// GQA: query head hq reads KV head hkv = (hq * nKvHeads) / nHeads
//      (encoders use nHeads == nKvHeads; kept general for reuse).
//
// Workgroup geometry:  grid( nHeads, T_q, 1 ), block( ATTN_LOCAL, 1, 1 )
//
// The CPU golden reference is compute::multiHeadAttention with
// positionOffset == T_k (its min(positionOffset+p+1, T_k) clamp then makes
// every query attend to the full [0, T_k) range — bit-equivalent to this).

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
void attention_encoder(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
          float* __restrict__ out,
    const int                 T_k,
    const int                 nHeads,
    const int                 nKvHeads,
    const int                 headDim,
    const float               scale)
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

    // -- Pass 1 — Q·K over ALL keys [0, T_k), scaled. -----------------
    for (int kk = lid; kk < T_k; kk += ATTN_LOCAL) {
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

    // -- Pass 2 — stable softmax over [0, T_k). -----------------------
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

    // -- Pass 3 — out[d] = sum_kk softmax_unnorm[kk] * v[kk][d] / sum. -
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
