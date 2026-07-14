// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

// R1 — GQA-Head-Packed variant of attention_prefill_flash_q8_0.cl.
//
// Colfax-style query-head packing: instead of one workgroup per
// (query-head, query-position), one workgroup handles ALL query heads
// of a KV-group at a given query position. The K and V blocks are
// dequantised ONCE from USM into registers per K-tile step and reused
// across every query head in the group — cuts KV-load bandwidth by
// the GQA ratio (`nHeads / nKvHeads`).
//
// For Gemma 4 26B-A4B: 16 heads / 8 KV-heads = 2 heads per group →
// 2× KV-bandwidth reduction. For pure MHA (ratio=1) this kernel is
// equivalent to the baseline; the dispatch layer picks the plain
// kernel there so the extra per-Q-head register array cost is not
// paid for nothing.
//
// Launch geometry (differs from the plain-flash prefill kernel):
//   get_group_id(0) = KV-head index    (0..nKvHeads-1)
//   get_group_id(1) = query position   (0..T_q-1)
//
// Compile-time constant `ATTN_FLASH_PREFILL_N_Q_PER_KV_MAX` bounds the
// register-array + SLM allocation for per-Q-head state. Runtime
// `nQPerKv = nHeads / nKvHeads` must be ≤ this cap; dispatch enforces
// the check host-side and falls back to the plain kernel if not.

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

// Max query heads per KV group. Sized for Qwen-style GQA (up to 8:1);
// Gemma 4 26B ratio=2 uses only the first 2 slots. Dispatch guards
// nQPerKv <= this value.
#ifndef ATTN_FLASH_PREFILL_N_Q_PER_KV_MAX
#define ATTN_FLASH_PREFILL_N_Q_PER_KV_MAX 8
#endif

#define Q8_0_BLOCK_ELEMENTS 32
#define Q8_0_BLOCK_BYTES    34

__attribute__((reqd_work_group_size(ATTN_FLASH_PREFILL_LOCAL, 1, 1)))
__attribute__((intel_reqd_sub_group_size(ATTN_FLASH_PREFILL_SG)))
__kernel void attention_prefill_flash_q8_0_gqa(
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

    const int hkv     = (int)get_group_id(0);   // KV-head index
    const int pq      = (int)get_group_id(1);   // query position
    const int lid     = (int)get_local_id(0);
    const int nQPerKv = nHeads / nKvHeads;      // >= 1, ≤ N_Q_PER_KV_MAX

    const int qStride        = nHeads   * headDim;
    const int kvDim          = nKvHeads * headDim;
    const int nBlocksPerRow  = kvDim   / Q8_0_BLOCK_ELEMENTS;
    const int nBlocksPerHead = headDim / Q8_0_BLOCK_ELEMENTS;
    const int headBlockBase  = (hkv * headDim) / Q8_0_BLOCK_ELEMENTS;
    const int positionOffset = curLenPtr[0];
    const int absPos         = positionOffset + pq;
    const int kMax           = absPos + 1;
    const int kMin           = (slidingWindow > 0 && kMax > slidingWindow)
                                 ? (kMax - slidingWindow) : 0;
    const int ktStart        = kMin / ATTN_FLASH_PREFILL_KTILE;
    const int nKTiles        = (kMax + ATTN_FLASH_PREFILL_KTILE - 1)
                               / ATTN_FLASH_PREFILL_KTILE;

    // Q-head vector bases: one entry per active Q-head in this group.
    // hq = hkv * nQPerKv + qh.
    __global const float* qVecs[ATTN_FLASH_PREFILL_N_Q_PER_KV_MAX];
    __global       float* oVecs[ATTN_FLASH_PREFILL_N_Q_PER_KV_MAX];

    #pragma unroll
    for (int qh = 0; qh < ATTN_FLASH_PREFILL_N_Q_PER_KV_MAX; ++qh) {
        const int hq = hkv * nQPerKv + qh;
        qVecs[qh] = q   + (size_t)pq * (size_t)qStride
                        + (size_t)hq * (size_t)headDim;
        oVecs[qh] = out + (size_t)pq * (size_t)qStride
                        + (size_t)hq * (size_t)headDim;
    }

    // Per-Q-head online-softmax state.
    __local float scores[ATTN_FLASH_PREFILL_N_Q_PER_KV_MAX]
                        [ATTN_FLASH_PREFILL_KTILE];
    __local float oRun  [ATTN_FLASH_PREFILL_N_Q_PER_KV_MAX]
                        [ATTN_FLASH_PREFILL_MAX_HEADDIM];

    float m_reg[ATTN_FLASH_PREFILL_N_Q_PER_KV_MAX];
    float l_reg[ATTN_FLASH_PREFILL_N_Q_PER_KV_MAX];

    #pragma unroll
    for (int qh = 0; qh < ATTN_FLASH_PREFILL_N_Q_PER_KV_MAX; ++qh) {
        m_reg[qh] = -INFINITY;
        l_reg[qh] = 0.0f;
    }
    #pragma unroll
    for (int qh = 0; qh < ATTN_FLASH_PREFILL_N_Q_PER_KV_MAX; ++qh) {
        if (qh >= nQPerKv) continue;
        for (int d = lid; d < headDim; d += ATTN_FLASH_PREFILL_LOCAL) {
            oRun[qh][d] = 0.0f;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int kt = ktStart; kt < nKTiles; ++kt) {
        const int kStartRaw = kt * ATTN_FLASH_PREFILL_KTILE;
        const int kStart    = (kStartRaw > kMin) ? kStartRaw : kMin;
        const int kEndRaw   = kStartRaw + ATTN_FLASH_PREFILL_KTILE;
        const int kEnd      = (kEndRaw < kMax) ? kEndRaw : kMax;
        const int tileLen   = kEnd - kStart;

        // -- Pass A — Q·K scaled scores for this K-tile. --------------
        // The K-blocks (scale + 32 int8) are loaded ONCE per (kk, blk)
        // pair and reused across every active Q-head. This is the R1
        // savings vs the plain kernel where each Q-head reads its own
        // copy of K from USM.
        for (int kk = lid; kk < tileLen; kk += ATTN_FLASH_PREFILL_LOCAL) {
            __global const uchar* kRow = k
                + (size_t)(kStart + kk) * (size_t)nBlocksPerRow *
                  (size_t)Q8_0_BLOCK_BYTES;
            __global const uchar* kHead = kRow
                + (size_t)headBlockBase * (size_t)Q8_0_BLOCK_BYTES;

            float acc[ATTN_FLASH_PREFILL_N_Q_PER_KV_MAX];
            #pragma unroll
            for (int qh = 0; qh < ATTN_FLASH_PREFILL_N_Q_PER_KV_MAX; ++qh) {
                acc[qh] = 0.0f;
            }

            for (int blk = 0; blk < nBlocksPerHead; ++blk) {
                __global const uchar* blkPtr = kHead
                    + (size_t)blk * (size_t)Q8_0_BLOCK_BYTES;
                const float bscale =
                    vload_half(0, (__global const half*)blkPtr);
                __global const char* qArr =
                    (__global const char*)(blkPtr + 2);
                const int dBase = blk * Q8_0_BLOCK_ELEMENTS;
                for (int in = 0; in < Q8_0_BLOCK_ELEMENTS; ++in) {
                    // Each K-quant scaled once; broadcast to every
                    // active Q-head's dot-product accumulator.
                    const float k_val = bscale * (float)qArr[in];
                    #pragma unroll
                    for (int qh = 0;
                         qh < ATTN_FLASH_PREFILL_N_Q_PER_KV_MAX;
                         ++qh) {
                        acc[qh] += qVecs[qh][dBase + in] * k_val;
                    }
                }
            }

            #pragma unroll
            for (int qh = 0; qh < ATTN_FLASH_PREFILL_N_Q_PER_KV_MAX;
                 ++qh) {
                if (qh >= nQPerKv) continue;
                scores[qh][kk] = acc[qh] * scale;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // -- Pass B — Online-softmax rescale, per Q-head. -------------
        float alpha[ATTN_FLASH_PREFILL_N_Q_PER_KV_MAX];
        #pragma unroll
        for (int qh = 0; qh < ATTN_FLASH_PREFILL_N_Q_PER_KV_MAX; ++qh) {
            if (qh >= nQPerKv) {
                alpha[qh] = 1.0f;
                continue;
            }
            float mTilePart = -INFINITY;
            for (int kk = lid; kk < tileLen;
                 kk += ATTN_FLASH_PREFILL_LOCAL) {
                const float s = scores[qh][kk];
                if (s > mTilePart) mTilePart = s;
            }
            const float mTile = sub_group_reduce_max(mTilePart);
            const float mNew  = (m_reg[qh] > mTile) ? m_reg[qh] : mTile;
            alpha[qh]         = exp(m_reg[qh] - mNew);

            float lTilePart = 0.0f;
            for (int kk = lid; kk < tileLen;
                 kk += ATTN_FLASH_PREFILL_LOCAL) {
                const float e = exp(scores[qh][kk] - mNew);
                scores[qh][kk] = e;
                lTilePart += e;
            }
            const float lTile = sub_group_reduce_add(lTilePart);
            l_reg[qh] = alpha[qh] * l_reg[qh] + lTile;
            m_reg[qh] = mNew;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // -- Pass C — V·softmax → oRun, shared V-load across Q-heads. --
        // Each thread strides over headDim. Per (d, kk) we load one V
        // Q8_0 block header (scale + int8 quant), broadcast the resulting
        // v_val across every active Q-head's accumulator.
        for (int d = lid; d < headDim; d += ATTN_FLASH_PREFILL_LOCAL) {
            const int blk = d / Q8_0_BLOCK_ELEMENTS;
            const int in  = d % Q8_0_BLOCK_ELEMENTS;

            float acc_v[ATTN_FLASH_PREFILL_N_Q_PER_KV_MAX];
            #pragma unroll
            for (int qh = 0; qh < ATTN_FLASH_PREFILL_N_Q_PER_KV_MAX;
                 ++qh) {
                acc_v[qh] = (qh < nQPerKv)
                                ? (alpha[qh] * oRun[qh][d])
                                : 0.0f;
            }

            for (int kk = 0; kk < tileLen; ++kk) {
                __global const uchar* vRow = v
                    + (size_t)(kStart + kk) * (size_t)nBlocksPerRow *
                      (size_t)Q8_0_BLOCK_BYTES;
                __global const uchar* blkPtr = vRow
                    + (size_t)(headBlockBase + blk) *
                      (size_t)Q8_0_BLOCK_BYTES;
                const float bscale =
                    vload_half(0, (__global const half*)blkPtr);
                const char qi = ((__global const char*)blkPtr)[2 + in];
                const float v_val = bscale * (float)qi;

                #pragma unroll
                for (int qh = 0;
                     qh < ATTN_FLASH_PREFILL_N_Q_PER_KV_MAX; ++qh) {
                    acc_v[qh] += scores[qh][kk] * v_val;
                }
            }

            #pragma unroll
            for (int qh = 0; qh < ATTN_FLASH_PREFILL_N_Q_PER_KV_MAX;
                 ++qh) {
                if (qh >= nQPerKv) continue;
                oRun[qh][d] = acc_v[qh];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    #pragma unroll
    for (int qh = 0; qh < ATTN_FLASH_PREFILL_N_Q_PER_KV_MAX; ++qh) {
        if (qh >= nQPerKv) continue;
        const float invL = (l_reg[qh] > 0.0f) ? (1.0f / l_reg[qh]) : 0.0f;
        for (int d = lid; d < headDim; d += ATTN_FLASH_PREFILL_LOCAL) {
            oVecs[qh][d] = oRun[qh][d] * invL;
        }
    }
}