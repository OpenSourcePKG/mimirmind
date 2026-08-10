// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

// Batched non-causal encoder self-attention (XLM-R cross-encoder reranker,
// EncoderRunner batched path). B sequences packed as rows r = b*Tmax + t in
// q/k/v/out ([B*Tmax, nHeads*headDim]); each sequence b uses its first
// seqLens[b] rows (left-aligned, rest padding). A query (b, pq) attends ONLY
// to keys [0, seqLens[b]) of the same batch; padding queries get a zeroed
// output. Per sequence this is 1:1 with attention_encoder.cl.
//
// Geometry: global = (nHeads, Tmax, B), local = ATTN_ENC_LOCAL = 16,
// one 16-wide subgroup per (head, query-pos, batch). CPU reference:
// compute::multiHeadAttention per sequence with positionOffset = seqLen.

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
__kernel void attention_encoder_batched(
    __global const float* q,
    __global const float* k,
    __global const float* v,
    __global       float* out,
    __global const int*   seqLens,   // [B]
    const int             B,
    const int             Tmax,
    const int             nHeads,
    const int             nKvHeads,
    const int             headDim,
    const float           scale)
{
    const int hq  = (int)get_group_id(0);
    const int pq  = (int)get_group_id(1);
    const int b   = (int)get_group_id(2);
    const int lid = (int)get_local_id(0);

    if (b >= B) {
        return;
    }
    const int len = seqLens[b];

    const int qStride  = nHeads   * headDim;
    const int kvStride = nKvHeads * headDim;
    const int hkv      = (hq * nKvHeads) / nHeads;
    const size_t rowBase = (size_t)b * (size_t)Tmax;

    __global float* oVec =
        out + (rowBase + (size_t)pq) * (size_t)qStride + (size_t)hq * (size_t)headDim;

    // Padding query (or empty sequence): zero output and stop.
    if (pq >= len) {
        for (int d = lid; d < headDim; d += ATTN_ENC_LOCAL) {
            oVec[d] = 0.0f;
        }
        return;
    }

    __global const float* qVec =
        q + (rowBase + (size_t)pq) * (size_t)qStride + (size_t)hq * (size_t)headDim;

    __local float scores[ATTN_ENC_MAX_TK];

    // Pass 1 — Q·K over the batch's real keys [0, len), scaled.
    for (int kk = lid; kk < len; kk += ATTN_ENC_LOCAL) {
        __global const float* kVec =
            k + (rowBase + (size_t)kk) * (size_t)kvStride + (size_t)hkv * (size_t)headDim;
        float acc = 0.0f;
        for (int d = 0; d < headDim; ++d) {
            acc += qVec[d] * kVec[d];
        }
        scores[kk] = acc * scale;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Pass 2 — stable softmax over [0, len).
    float mPart = -INFINITY;
    for (int kk = lid; kk < len; kk += ATTN_ENC_LOCAL) {
        const float s = scores[kk];
        if (s > mPart) mPart = s;
    }
    const float maxScore = sub_group_reduce_max(mPart);

    float lPart = 0.0f;
    for (int kk = lid; kk < len; kk += ATTN_ENC_LOCAL) {
        const float e = exp(scores[kk] - maxScore);
        scores[kk] = e;
        lPart += e;
    }
    const float sumExp = sub_group_reduce_add(lPart);
    const float invSum = 1.0f / sumExp;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Pass 3 — out[d] = sum_kk exp_kk * v[kk][d] / sum.
    for (int d = lid; d < headDim; d += ATTN_ENC_LOCAL) {
        float acc = 0.0f;
        for (int kk = 0; kk < len; ++kk) {
            __global const float* vVec =
                v + (rowBase + (size_t)kk) * (size_t)kvStride + (size_t)hkv * (size_t)headDim;
            acc += scores[kk] * vVec[d];
        }
        oVec[d] = acc * invSum;
    }
}
