// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

// Q8_0-KV variant of attention_flash_partial.cl. Decode-mode
// FlashAttention pass 1 (T_q == 1), reading K/V as Q8_0 blocks (32
// elements per block: fp16 scale + 32 int8 quants = 34 B). Loads
// dequantise on the fly in registers; the per-tile (m, l, o_unnorm)
// partials written to `partialMlo` stay fp32 so the existing
// attention_flash_merge kernel (which reads and combines partials in
// fp32) is unchanged and works for all KV dtypes.
//
// Row / head layout matches attention_q8_0.cl: one K/V row is
// `(kvDim / 32)` blocks of 34 B; `hkv * headDim` is a multiple of 32
// (headDim ∈ {64, 256, 512} for the models we run), so the per-head
// block origin lands on a block boundary.
//
// M10.2 Phase 1a Commit 4 — this kernel is only invoked when
// `KvCache::dtype() == Q8_0`; the f32 path stays on
// attention_flash_partial.cl and the fp16 path stays on
// attention_flash_partial_fp16.cl, preserving bit parity against those
// baselines.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

#ifndef ATTN_FLASH_LOCAL
#define ATTN_FLASH_LOCAL 16
#endif

#ifndef ATTN_FLASH_SG
#define ATTN_FLASH_SG 16
#endif

#ifndef ATTN_FLASH_KTILE
#define ATTN_FLASH_KTILE 64
#endif

#define Q8_0_BLOCK_ELEMENTS 32
#define Q8_0_BLOCK_BYTES    34

__attribute__((reqd_work_group_size(ATTN_FLASH_LOCAL, 1, 1)))
__attribute__((intel_reqd_sub_group_size(ATTN_FLASH_SG)))
__kernel void attention_flash_partial_q8_0(
    __global const float* q,
    __global const uchar* k,
    __global const uchar* v,
    __global       float* partialMlo,
    const int             nHeads,
    const int             nKvHeads,
    const int             headDim,
    __global const int*   curLenPtr,
    const float           scale,
    const int             slidingWindow)
{
    const int hq  = (int)get_group_id(0);
    const int kt  = (int)get_group_id(1);
    const int lid = (int)get_local_id(0);

    const int kvDim          = nKvHeads * headDim;
    const int nBlocksPerRow  = kvDim   / Q8_0_BLOCK_ELEMENTS;
    const int nBlocksPerHead = headDim / Q8_0_BLOCK_ELEMENTS;
    const int hkv            = (hq * nKvHeads) / nHeads;
    const int headBlockBase  = (hkv * headDim) / Q8_0_BLOCK_ELEMENTS;
    const int positionOffset = curLenPtr[0];

    const int kMax    = positionOffset + 1;
    const int kMin    = (slidingWindow > 0 && kMax > slidingWindow)
                          ? (kMax - slidingWindow) : 0;
    const int nKTiles = (kMax + ATTN_FLASH_KTILE - 1) / ATTN_FLASH_KTILE;
    const int kStartRaw = kt * ATTN_FLASH_KTILE;
    const int kStart  = (kStartRaw > kMin) ? kStartRaw : kMin;
    const int kEndRaw = kStartRaw + ATTN_FLASH_KTILE;
    const int kEnd    = (kEndRaw < kMax) ? kEndRaw : kMax;

    __global float* mloPtr =
        partialMlo + ((size_t)hq * (size_t)nKTiles + (size_t)kt) *
                     (size_t)(2 + headDim);

    // Past the causal mask OR entirely below the sliding-window low
    // bound — emit a neutral partial so the merge sees no contribution.
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

    __global const float* qVec = q + (size_t)hq * (size_t)headDim;

    __local float scores[ATTN_FLASH_KTILE];

    const int tileLen = kEnd - kStart;

    // Pass 1 — Q·K scores, scaled. Dequant per block: read scale once,
    // dot 32 elements against qVec, fold into `acc` with one scale mul.
    for (int kk = lid; kk < tileLen; kk += ATTN_FLASH_LOCAL) {
        const int absKk = kStart + kk;
        __global const uchar* kRow = k
            + (size_t)absKk * (size_t)nBlocksPerRow * (size_t)Q8_0_BLOCK_BYTES;
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
        scores[kk] = acc * scale;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Pass 2 — stable softmax on the tile.
    float mPart = -INFINITY;
    for (int kk = lid; kk < tileLen; kk += ATTN_FLASH_LOCAL) {
        const float s = scores[kk];
        if (s > mPart) mPart = s;
    }
    const float mLocal = sub_group_reduce_max(mPart);

    float lPart = 0.0f;
    for (int kk = lid; kk < tileLen; kk += ATTN_FLASH_LOCAL) {
        const float e = exp(scores[kk] - mLocal);
        scores[kk] = e;
        lPart += e;
    }
    const float lLocal = sub_group_reduce_add(lPart);
    barrier(CLK_LOCAL_MEM_FENCE);

    // Pass 3 — unnormalized V-weighted sum. Stripe d across threads.
    // Per (kk, d) we load one Q8_0 block header (scale) plus one int8
    // quant — the compiler can hoist the scale load across the d-loop
    // because within the WG adjacent threads hit the same block.
    for (int d = lid; d < headDim; d += ATTN_FLASH_LOCAL) {
        const int blk = d / Q8_0_BLOCK_ELEMENTS;
        const int in  = d % Q8_0_BLOCK_ELEMENTS;
        float acc = 0.0f;
        for (int kk = 0; kk < tileLen; ++kk) {
            __global const uchar* vRow = v
                + (size_t)(kStart + kk) * (size_t)nBlocksPerRow *
                  (size_t)Q8_0_BLOCK_BYTES;
            __global const uchar* blkPtr = vRow
                + (size_t)(headBlockBase + blk) * (size_t)Q8_0_BLOCK_BYTES;
            const float bscale = vload_half(0, (__global const half*)blkPtr);
            const char qi = ((__global const char*)blkPtr)[2 + in];
            acc += scores[kk] * bscale * (float)qi;
        }
        mloPtr[2 + d] = acc;
    }

    if (lid == 0) {
        mloPtr[0] = mLocal;
        mloPtr[1] = lLocal;
    }
}