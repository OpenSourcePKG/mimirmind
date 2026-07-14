// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

// FP16-KV variant of attention.cl. K and V are stored as fp16 in USM
// (halved bandwidth + halved footprint), Q and OUT stay fp32. Loads
// promote fp16 → fp32 in registers via vload_half; all reductions and
// the softmax remain fp32-precision.
//
// Everything else (layout stride math, causal + SWA masking, Pass 1/2/3
// structure, subgroup reductions) is bit-for-bit identical to the f32
// variant so the KV-dtype swap stays isolated to load-sites.
//
// M10.2 Phase 0 Commit 3 — this kernel is only invoked when
// `KvCache::dtype() == FP16`; the f32 path stays on attention.cl,
// preserving bit parity against pre-M10.2 behaviour.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

#ifndef ATTN_LOCAL
#define ATTN_LOCAL 16
#endif

#ifndef ATTN_SG
#define ATTN_SG 16
#endif

#ifndef ATTN_MAX_TK
#define ATTN_MAX_TK 16384
#endif

__attribute__((reqd_work_group_size(ATTN_LOCAL, 1, 1)))
__attribute__((intel_reqd_sub_group_size(ATTN_SG)))
__kernel void attention_fp16(
    __global const float* q,
    __global const half*  k,
    __global const half*  v,
    __global       float* out,
    const int             T_q,
    const int             nHeads,
    const int             nKvHeads,
    const int             headDim,
    __global const int*   curLenPtr,
    const float           scale,
    const int             slidingWindow)
{
    (void)T_q;

    const int hq  = (int)get_group_id(0);
    const int pq  = (int)get_group_id(1);
    const int lid = (int)get_local_id(0);

    const int qStride        = nHeads   * headDim;
    const int kvStride       = nKvHeads * headDim;
    const int hkv            = (hq * nKvHeads) / nHeads;
    const int positionOffset = curLenPtr[0];
    const int absPos         = positionOffset + pq;
    const int kMax           = absPos + 1;
    const int kMin           = (slidingWindow > 0 && kMax > slidingWindow)
                                 ? (kMax - slidingWindow) : 0;

    __global const float* qVec = q   + pq * qStride + hq * headDim;
    __global       float* oVec = out + pq * qStride + hq * headDim;

    __local float scores[ATTN_MAX_TK];

    // -- Pass 1 — Q·K dot products, scaled. ---------------------------
    for (int kk = kMin + lid; kk < kMax; kk += ATTN_LOCAL) {
        __global const half* kVec = k + kk * kvStride + hkv * headDim;
        float acc = 0.0f;
        for (int d = 0; d < headDim; ++d) {
            acc += qVec[d] * vload_half(d, kVec);
        }
        scores[kk - kMin] = acc * scale;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // -- Pass 2 — stable softmax over [kMin, kMax). -------------------
    float mPart = -INFINITY;
    for (int kk = kMin + lid; kk < kMax; kk += ATTN_LOCAL) {
        const float s = scores[kk - kMin];
        if (s > mPart) mPart = s;
    }
    const float maxScore = sub_group_reduce_max(mPart);

    float lPart = 0.0f;
    for (int kk = kMin + lid; kk < kMax; kk += ATTN_LOCAL) {
        const float e = exp(scores[kk - kMin] - maxScore);
        scores[kk - kMin] = e;
        lPart += e;
    }
    const float sumExp = sub_group_reduce_add(lPart);
    const float invSum = 1.0f / sumExp;
    barrier(CLK_LOCAL_MEM_FENCE);

    // -- Pass 3 — out[d] = (sum_kk softmax_unnorm[kk] * v[kk][d]) / sum.
    for (int d = lid; d < headDim; d += ATTN_LOCAL) {
        float acc = 0.0f;
        for (int kk = kMin; kk < kMax; ++kk) {
            __global const half* vVec = v + kk * kvStride + hkv * headDim;
            acc += scores[kk - kMin] * vload_half(d, vVec);
        }
        oVec[d] = acc * invSum;
    }
}