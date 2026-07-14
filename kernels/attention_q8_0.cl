// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

// Q8_0-KV variant of attention.cl. K and V are stored as ggml-style
// Q8_0 blocks in USM (32 elements per block: fp16 scale + 32 int8 quants
// = 34 B per block). Q and OUT stay fp32. Loads dequantise on the fly in
// registers (block scale × int8 → fp32); the softmax state stays fully
// fp32-precision.
//
// Everything else (layout stride math via block indices, causal + SWA
// masking, Pass 1/2/3 structure, subgroup reductions) is bit-for-bit
// identical to the f32 / fp16 variants so the KV-dtype swap stays
// isolated to load-sites.
//
// Row layout: one K/V row is (kvDim / 32) Q8_0 blocks × 34 B. Since
// `hkv * headDim` is a multiple of 32 for every architecture we run
// (Qwen 2.5 headDim=64, Gemma 4 headDim=256/512) the per-head block
// origin lands on a block boundary, so `headBlockBase = (hkv*headDim)/32`
// and each of the `headDim/32` blocks per head is fully-owned by this
// head. The `headDim % 32 == 0` invariant is enforced by KvCache's ctor
// assertion (see KvCache.cpp).
//
// M10.2 Phase 1a Commit 4 — this kernel is only invoked when
// `KvCache::dtype() == Q8_0`; the f32 path stays on attention.cl and
// the fp16 path stays on attention_fp16.cl, preserving bit parity
// against those baselines.

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

#define Q8_0_BLOCK_ELEMENTS 32
#define Q8_0_BLOCK_BYTES    34

__attribute__((reqd_work_group_size(ATTN_LOCAL, 1, 1)))
__attribute__((intel_reqd_sub_group_size(ATTN_SG)))
__kernel void attention_q8_0(
    __global const float* q,
    __global const uchar* k,
    __global const uchar* v,
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
    const int kvDim          = nKvHeads * headDim;
    const int nBlocksPerRow  = kvDim   / Q8_0_BLOCK_ELEMENTS;
    const int nBlocksPerHead = headDim / Q8_0_BLOCK_ELEMENTS;
    const int hkv            = (hq * nKvHeads) / nHeads;
    const int headBlockBase  = (hkv * headDim) / Q8_0_BLOCK_ELEMENTS;
    const int positionOffset = curLenPtr[0];
    const int absPos         = positionOffset + pq;
    const int kMax           = absPos + 1;
    const int kMin           = (slidingWindow > 0 && kMax > slidingWindow)
                                 ? (kMax - slidingWindow) : 0;

    __global const float* qVec = q   + pq * qStride + hq * headDim;
    __global       float* oVec = out + pq * qStride + hq * headDim;

    __local float scores[ATTN_MAX_TK];

    // -- Pass 1 — Q·K dot products, scaled. ---------------------------
    // For each K row, iterate the `nBlocksPerHead` blocks that belong
    // to head `hkv`. The block scale loads once per block; the 32
    // int8 quants are then multiplied against the matching 32 fp32
    // qVec entries, and the block-local dot is folded into `acc` with
    // one scale multiply.
    for (int kk = kMin + lid; kk < kMax; kk += ATTN_LOCAL) {
        __global const uchar* kRow = k
            + (size_t)kk * (size_t)nBlocksPerRow * (size_t)Q8_0_BLOCK_BYTES;
        __global const uchar* kHead = kRow
            + (size_t)headBlockBase * (size_t)Q8_0_BLOCK_BYTES;
        float acc = 0.0f;
        for (int blk = 0; blk < nBlocksPerHead; ++blk) {
            __global const uchar* blkPtr = kHead
                + (size_t)blk * (size_t)Q8_0_BLOCK_BYTES;
            const float bscale = vload_half(0, (__global const half*)blkPtr);
            __global const char* qArr = (__global const char*)(blkPtr + 2);
            const int dBase = blk * Q8_0_BLOCK_ELEMENTS;
            float sub = 0.0f;
            for (int in = 0; in < Q8_0_BLOCK_ELEMENTS; ++in) {
                sub += qVec[dBase + in] * (float)qArr[in];
            }
            acc += bscale * sub;
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
    // Stripe d across threads. Per (kk, d) we load one Q8_0 block header
    // (scale) plus one int8 quant. Adjacent threads on the same WG hit
    // the same block (headDim/32 blocks per head, ATTN_LOCAL=16 threads,
    // headDim ∈ {64, 256, 512}) so the scale-load coalesces naturally
    // in the L1/SLM caches.
    for (int d = lid; d < headDim; d += ATTN_LOCAL) {
        const int blk = d / Q8_0_BLOCK_ELEMENTS;
        const int in  = d % Q8_0_BLOCK_ELEMENTS;
        float acc = 0.0f;
        for (int kk = kMin; kk < kMax; ++kk) {
            __global const uchar* vRow = v
                + (size_t)kk * (size_t)nBlocksPerRow * (size_t)Q8_0_BLOCK_BYTES;
            __global const uchar* blkPtr = vRow
                + (size_t)(headBlockBase + blk) * (size_t)Q8_0_BLOCK_BYTES;
            const float bscale = vload_half(0, (__global const half*)blkPtr);
            const char qi = ((__global const char*)blkPtr)[2 + in];
            acc += scores[kk - kMin] * bscale * (float)qi;
        }
        oVec[d] = acc * invSum;
    }
}