// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

// Q8_0-KV variant of attention_prefill_flash.cl. Single-workgroup
// streaming FlashAttention for T_q > 1 (M5i.J), reading K/V as Q8_0
// blocks (32 elements per block: fp16 scale + 32 int8 quants = 34 B).
// Loads dequantise on the fly in registers; the online-softmax state
// (m, l, o) stays fully fp32-precision, matching the F32 / FP16
// variants.
//
// Row / head layout matches the other Q8_0 attention kernels: one K/V
// row is `(kvDim / 32)` blocks of 34 B; `hkv * headDim` is a multiple
// of 32 for every model we run (headDim ∈ {64, 256, 512}), so the
// per-head block origin lands on a block boundary. The K-tile of 128
// keys iterates `headDim/32` Q8_0 blocks per K row.
//
// M10.2 Phase 1a Commit 4 — this kernel is only invoked when
// `KvCache::dtype() == Q8_0`; the f32 path stays on
// attention_prefill_flash.cl and the fp16 path stays on
// attention_prefill_flash_fp16.cl, preserving bit parity against those
// baselines.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

#ifndef ATTN_FLASH_PREFILL_LOCAL
#define ATTN_FLASH_PREFILL_LOCAL 16
#endif

#ifndef ATTN_FLASH_PREFILL_SG
#define ATTN_FLASH_PREFILL_SG 16
#endif

#ifndef ATTN_FLASH_PREFILL_KTILE
#define ATTN_FLASH_PREFILL_KTILE 128
#endif

#ifndef ATTN_FLASH_PREFILL_MAX_HEADDIM
#define ATTN_FLASH_PREFILL_MAX_HEADDIM 512
#endif

#define Q8_0_BLOCK_ELEMENTS 32
#define Q8_0_BLOCK_BYTES    34

__attribute__((reqd_work_group_size(ATTN_FLASH_PREFILL_LOCAL, 1, 1)))
__attribute__((intel_reqd_sub_group_size(ATTN_FLASH_PREFILL_SG)))
__kernel void attention_prefill_flash_q8_0(
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
    const int ktStart        = kMin / ATTN_FLASH_PREFILL_KTILE;
    const int nKTiles        = (kMax + ATTN_FLASH_PREFILL_KTILE - 1)
                               / ATTN_FLASH_PREFILL_KTILE;

    __global const float* qVec = q   + (size_t)pq * (size_t)qStride
                                     + (size_t)hq * (size_t)headDim;
    __global       float* oVec = out + (size_t)pq * (size_t)qStride
                                     + (size_t)hq * (size_t)headDim;

    __local float scores[ATTN_FLASH_PREFILL_KTILE];
    __local float oRun  [ATTN_FLASH_PREFILL_MAX_HEADDIM];

    float m = -INFINITY;
    float l = 0.0f;
    for (int d = lid; d < headDim; d += ATTN_FLASH_PREFILL_LOCAL) {
        oRun[d] = 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int kt = ktStart; kt < nKTiles; ++kt) {
        const int kStartRaw = kt * ATTN_FLASH_PREFILL_KTILE;
        const int kStart    = (kStartRaw > kMin) ? kStartRaw : kMin;
        const int kEndRaw   = kStartRaw + ATTN_FLASH_PREFILL_KTILE;
        const int kEnd      = (kEndRaw < kMax) ? kEndRaw : kMax;
        const int tileLen   = kEnd - kStart;

        // -- Pass A — Q·K scaled scores for this K-tile. --------------
        // Dequant per block: read scale once, dot 32 elements against
        // qVec, fold into `acc` with one scale mul.
        for (int kk = lid; kk < tileLen; kk += ATTN_FLASH_PREFILL_LOCAL) {
            __global const uchar* kRow = k
                + (size_t)(kStart + kk) * (size_t)nBlocksPerRow *
                  (size_t)Q8_0_BLOCK_BYTES;
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

        // -- Pass B — Online-softmax rescale. -------------------------
        float mTilePart = -INFINITY;
        for (int kk = lid; kk < tileLen; kk += ATTN_FLASH_PREFILL_LOCAL) {
            const float s = scores[kk];
            if (s > mTilePart) mTilePart = s;
        }
        const float mTile = sub_group_reduce_max(mTilePart);
        const float mNew  = (m > mTile) ? m : mTile;
        const float alpha = exp(m - mNew);

        float lTilePart = 0.0f;
        for (int kk = lid; kk < tileLen; kk += ATTN_FLASH_PREFILL_LOCAL) {
            const float e = exp(scores[kk] - mNew);
            scores[kk] = e;
            lTilePart += e;
        }
        const float lTile = sub_group_reduce_add(lTilePart);
        l = alpha * l + lTile;
        m = mNew;
        barrier(CLK_LOCAL_MEM_FENCE);

        // -- Pass C — scale-and-accumulate V into oRun. ---------------
        // Stripe d across threads. Per (kk, d) we load one Q8_0 block
        // header (scale) plus one int8 quant. Adjacent threads on the
        // same WG hit the same block so the scale-load coalesces.
        for (int d = lid; d < headDim; d += ATTN_FLASH_PREFILL_LOCAL) {
            const int blk = d / Q8_0_BLOCK_ELEMENTS;
            const int in  = d % Q8_0_BLOCK_ELEMENTS;
            float acc = alpha * oRun[d];
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
            oRun[d] = acc;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const float invL = (l > 0.0f) ? (1.0f / l) : 0.0f;
    for (int d = lid; d < headDim; d += ATTN_FLASH_PREFILL_LOCAL) {
        oVec[d] = oRun[d] * invL;
    }
}