// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Batched non-causal multi-head self-attention for the BERT/RoBERTa/XLM-R
// cross-encoder reranker (EncoderRunner). Runs B independent sequences packed
// into one dispatch so a rerank of B (query, passage) pairs is a single GPU
// forward instead of B sequential ones.
//
// Each sequence b occupies rows [b*Tmax, b*Tmax + Tmax) but only its first
// seqLens[b] rows are real (left-aligned; the rest is padding). A query at
// (batch b, pos pq) attends ONLY to keys [0, seqLens[b]) of the SAME batch —
// padding is never attended, and padding queries (pq >= seqLens[b]) are
// skipped. 1:1 with attention_encoder.cu per sequence.
//
// Layout (all row-major fp32), row index r = b*Tmax + t:
//   q/k/v/out: [B*Tmax, nHeads(orKv), headDim]
// Grid( nHeads, Tmax, B ), block( ATTN_LOCAL ).

#include <cuda_runtime.h>

#include <math.h>

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
void attention_encoder_batched(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
          float* __restrict__ out,
    const int* __restrict__   seqLens,   // [B]
    const int                 B,
    const int                 Tmax,
    const int                 nHeads,
    const int                 nKvHeads,
    const int                 headDim,
    const float               scale)
{
    const int hq  = blockIdx.x;
    const int pq  = blockIdx.y;
    const int b   = blockIdx.z;
    const int lid = threadIdx.x;

    if (b >= B) {
        return;
    }
    const int len = seqLens[b];

    const int qStride  = nHeads   * headDim;
    const int kvStride = nKvHeads * headDim;
    const int hkv      = (hq * nKvHeads) / nHeads;
    const size_t rowBase = static_cast<size_t>(b) * static_cast<size_t>(Tmax);

    float* __restrict__ oVec =
        out + (rowBase + static_cast<size_t>(pq)) * static_cast<size_t>(qStride)
            + static_cast<size_t>(hq) * static_cast<size_t>(headDim);

    // Padding query row (or empty sequence): zero its output and stop.
    if (pq >= len) {
        for (int d = lid; d < headDim; d += ATTN_LOCAL) {
            oVec[d] = 0.0f;
        }
        return;
    }

    const float* __restrict__ qVec =
        q + (rowBase + static_cast<size_t>(pq)) * static_cast<size_t>(qStride)
          + static_cast<size_t>(hq) * static_cast<size_t>(headDim);

    __shared__ float scores[ATTN_MAX_TK];

    // Pass 1 — Q·K over the batch's real keys [0, len), scaled.
    for (int kk = lid; kk < len; kk += ATTN_LOCAL) {
        const float* __restrict__ kVec =
            k + (rowBase + static_cast<size_t>(kk)) * static_cast<size_t>(kvStride)
              + static_cast<size_t>(hkv) * static_cast<size_t>(headDim);
        float acc = 0.0f;
        for (int d = 0; d < headDim; ++d) {
            acc += qVec[d] * kVec[d];
        }
        scores[kk] = acc * scale;
    }
    __syncthreads();

    // Pass 2 — stable softmax over [0, len).
    float mPart = -INFINITY;
    for (int kk = lid; kk < len; kk += ATTN_LOCAL) {
        const float s = scores[kk];
        if (s > mPart) mPart = s;
    }
    const float maxScore = warp16_reduce_max(mPart);

    float lPart = 0.0f;
    for (int kk = lid; kk < len; kk += ATTN_LOCAL) {
        const float e = expf(scores[kk] - maxScore);
        scores[kk] = e;
        lPart += e;
    }
    const float sumExp = warp16_reduce_sum(lPart);
    const float invSum = 1.0f / sumExp;
    __syncthreads();

    // Pass 3 — out[d] = sum_kk exp_kk * v[kk][d] / sum.
    for (int d = lid; d < headDim; d += ATTN_LOCAL) {
        float acc = 0.0f;
        for (int kk = 0; kk < len; ++kk) {
            const float* __restrict__ vVec =
                v + (rowBase + static_cast<size_t>(kk)) * static_cast<size_t>(kvStride)
                  + static_cast<size_t>(hkv) * static_cast<size_t>(headDim);
            acc += scores[kk] * vVec[d];
        }
        oVec[d] = acc * invSum;
    }
}
