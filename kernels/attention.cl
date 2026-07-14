// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

// Multi-head self-attention, GQA-aware, causal. Drop-in equivalent of
// compute::multiHeadAttention but resident on the iGPU.
//
// Layout (all row-major f32, USM):
//   q:   [T_q, nHeads,    headDim]
//   k:   [T_k, nKvHeads,  headDim]
//   v:   [T_k, nKvHeads,  headDim]
//   out: [T_q, nHeads,    headDim]
//
// Causal mask: query position pq attends to key positions
//   [kMin, min(positionOffset + pq + 1, T_k))
// where
//   kMax = positionOffset + pq + 1
//   kMin = (slidingWindow > 0 && kMax > slidingWindow)
//            ? (kMax - slidingWindow) : 0
// so `slidingWindow == 0` degenerates to pure causal (all past keys).
// `slidingWindow > 0` matches Gemma-family SWA layers: each query sees
// only the last `slidingWindow` keys of its causal prefix.
//
// GQA: query head hq reads from KV head hkv = (hq * nKvHeads) / nHeads.
//
// scale is applied to Q·K BEFORE softmax. Qwen-family passes
// 1/sqrt(headDim); Gemma 4 passes 1.0 (its f_attention_scale = 1).
//
// Workgroup geometry — M5f.3 variant (a):
//   global = (nHeads, T_q, 1) with ATTN_LOCAL=ATTN_SG=16 threads per WG.
//   One workgroup owns exactly one (hq, pq): computes the score row,
//   does the softmax, and writes the output row. No cross-WG sync.
//
// SLM use: scores[ATTN_MAX_TK] = 64 KiB at MAX_TK=16384. Exactly at
//   Arc Xe-LPG's 64 KiB per-work-group SLM budget — the plain path
//   cannot grow further without a real refactor (M9.8b: online-softmax
//   + tiled scores, i.e. FlashAttention semantics for T_q > 1). For
//   sizes below MAX_TK the fail-fast guard lives in
//   GpuOps::attentionAsync.
//
// Reference CPU implementation: src/compute/Attention.cpp.

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

// M-CLR.2: positionOffset moved to a __global int-slot (curLenPtr) that
// the host updates before every dispatch. T_k was previously passed by
// value; it was only used to cap kMax when positionOffset+pq+1 exceeded
// it, but every current caller satisfies T_k = positionOffset + T_q, so
// the cap is a no-op and T_k drops out of the signature. See M-CLR
// inventory (Synaipse research/2026-07-06-mclr1-kernel-inventory.md).
__attribute__((reqd_work_group_size(ATTN_LOCAL, 1, 1)))
__attribute__((intel_reqd_sub_group_size(ATTN_SG)))
__kernel void attention(
    __global const float* q,
    __global const float* k,
    __global const float* v,
    __global       float* out,
    const int             T_q,
    const int             nHeads,
    const int             nKvHeads,
    const int             headDim,
    __global const int*   curLenPtr,
    const float           scale,
    const int             slidingWindow)
{
    const int hq  = (int)get_group_id(0);
    const int pq  = (int)get_group_id(1);
    const int lid = (int)get_local_id(0);

    const int qStride       = nHeads   * headDim;
    const int kvStride      = nKvHeads * headDim;
    const int hkv           = (hq * nKvHeads) / nHeads;
    const int positionOffset = curLenPtr[0];
    const int absPos        = positionOffset + pq;
    const int kMax          = absPos + 1;
    // Sliding-window low bound. Zero means "no window" (Full-Attention
    // or non-SWA architectures).
    const int kMin          = (slidingWindow > 0 && kMax > slidingWindow)
                                ? (kMax - slidingWindow) : 0;
    // scores[] is indexed by (kk - kMin) so the SLM footprint stays at
    // slidingWindow floats — but we keep the [ATTN_MAX_TK] allocation
    // to avoid a variable-length local; indexing writes/reads to
    // scores[kk - kMin] instead of scores[kk] compared to the pre-SWA
    // version.

    __global const float* qVec = q   + pq * qStride + hq * headDim;
    __global       float* oVec = out + pq * qStride + hq * headDim;

    __local float scores[ATTN_MAX_TK];

    // -- Pass 1 — Q·K dot products, scaled. ---------------------------
    // Stripe k-positions across the 16-wide subgroup. Inner loop over
    // headDim stays in registers.
    for (int kk = kMin + lid; kk < kMax; kk += ATTN_LOCAL) {
        __global const float* kVec = k + kk * kvStride + hkv * headDim;
        float acc = 0.0f;
        for (int d = 0; d < headDim; ++d) {
            acc += qVec[d] * kVec[d];
        }
        scores[kk - kMin] = acc * scale;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // -- Pass 2 — stable softmax over [kMin, kMax). -------------------
    // 2a) max-reduction. Per-thread partial then subgroup_reduce_max.
    float mPart = -INFINITY;
    for (int kk = kMin + lid; kk < kMax; kk += ATTN_LOCAL) {
        const float s = scores[kk - kMin];
        if (s > mPart) mPart = s;
    }
    const float maxScore = sub_group_reduce_max(mPart);

    // 2b) replace scores with exp(score - max), accumulate sum.
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
    // Stripe d across threads. Each thread sweeps [kMin, kMax) to pick
    // up the V column.
    for (int d = lid; d < headDim; d += ATTN_LOCAL) {
        float acc = 0.0f;
        for (int kk = kMin; kk < kMax; ++kk) {
            __global const float* vVec = v + kk * kvStride + hkv * headDim;
            acc += scores[kk - kMin] * vVec[d];
        }
        oVec[d] = acc * invSum;
    }
}