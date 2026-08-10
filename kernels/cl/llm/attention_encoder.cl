// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

// Non-causal (bidirectional) multi-head self-attention — the BERT/RoBERTa/
// XLM-R encoder attention for the cross-encoder reranker (EncoderRunner).
// A 1:1 mirror of attention.cl EXCEPT every query attends to ALL keys
// [0, T_k): no causal mask, no sliding window, no positionOffset/curLen,
// no KV cache. nHeads == nKvHeads for XLM-R (no GQA), but the GQA mapping
// is kept for generality.
//
// Layout (all row-major f32, USM):
//   q/k/v/out: [T, nHeads(orKv), headDim]   (T_q == T_k == T)
//
// scale is applied to Q·K before softmax (1/sqrt(headDim)).
//
// Geometry: global = (nHeads, T, 1), local = ATTN_ENC_LOCAL = 16, one
// 16-wide subgroup per (head, query-pos). SLM scores[ATTN_ENC_MAX_TK].
// CPU reference: compute::multiHeadAttention(..., positionOffset = T_k).

#pragma OPENCL EXTENSION cl_intel_subgroups : enable

#ifndef ATTN_ENC_LOCAL
#define ATTN_ENC_LOCAL 16
#endif

#ifndef ATTN_ENC_SG
#define ATTN_ENC_SG 16
#endif

#ifndef ATTN_ENC_MAX_TK
#define ATTN_ENC_MAX_TK 16384
#endif

__attribute__((reqd_work_group_size(ATTN_ENC_LOCAL, 1, 1)))
__attribute__((intel_reqd_sub_group_size(ATTN_ENC_SG)))
__kernel void attention_encoder(
    __global const float* q,
    __global const float* k,
    __global const float* v,
    __global       float* out,
    const int             T_k,
    const int             nHeads,
    const int             nKvHeads,
    const int             headDim,
    const float           scale)
{
    const int hq  = (int)get_group_id(0);
    const int pq  = (int)get_group_id(1);
    const int lid = (int)get_local_id(0);

    const int qStride  = nHeads   * headDim;
    const int kvStride = nKvHeads * headDim;
    const int hkv      = (hq * nKvHeads) / nHeads;

    __global const float* qVec = q   + pq * qStride + hq * headDim;
    __global       float* oVec = out + pq * qStride + hq * headDim;

    __local float scores[ATTN_ENC_MAX_TK];

    // Pass 1 — Q·K dot products, scaled, over the full key range.
    for (int kk = lid; kk < T_k; kk += ATTN_ENC_LOCAL) {
        __global const float* kVec = k + kk * kvStride + hkv * headDim;
        float acc = 0.0f;
        for (int d = 0; d < headDim; ++d) {
            acc += qVec[d] * kVec[d];
        }
        scores[kk] = acc * scale;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Pass 2 — stable softmax over [0, T_k).
    float mPart = -INFINITY;
    for (int kk = lid; kk < T_k; kk += ATTN_ENC_LOCAL) {
        const float s = scores[kk];
        if (s > mPart) mPart = s;
    }
    const float maxScore = sub_group_reduce_max(mPart);

    float lPart = 0.0f;
    for (int kk = lid; kk < T_k; kk += ATTN_ENC_LOCAL) {
        const float e = exp(scores[kk] - maxScore);
        scores[kk] = e;
        lPart += e;
    }
    const float sumExp = sub_group_reduce_add(lPart);
    const float invSum = 1.0f / sumExp;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Pass 3 — out[d] = (sum_kk exp_kk * v[kk][d]) / sum.
    for (int d = lid; d < headDim; d += ATTN_ENC_LOCAL) {
        float acc = 0.0f;
        for (int kk = 0; kk < T_k; ++kk) {
            __global const float* vVec = v + kk * kvStride + hkv * headDim;
            acc += scores[kk] * vVec[d];
        }
        oVec[d] = acc * invSum;
    }
}
