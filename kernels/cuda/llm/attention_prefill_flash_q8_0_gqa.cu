// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
// Ported from kernels_hip/attention_prefill_flash_q8_0_gqa.hip — Track 4 mechanical port, no functional change intended.

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// R1 — GQA-Head-Packed variant of attention_prefill_flash_q8_0.hip.
//
// Colfax-style query-head packing: instead of one workgroup per
// (query-head, query-position), one workgroup handles ALL query heads
// of a KV-group at a given query position. The K and V blocks are
// dequantised ONCE per K-tile step and reused across every query
// head in the group — cuts KV-load bandwidth by the GQA ratio
// (nHeads / nKvHeads).
//
// For Gemma 4 26B-A4B: 16 heads / 8 KV-heads = 2 heads per group →
// 2× KV-bandwidth reduction. For pure MHA (ratio=1) this kernel is
// equivalent to the plain kernel; dispatch picks the plain kernel
// there so the extra per-Q-head register / SLM cost is not paid for
// nothing.
//
// Launch geometry (differs from the plain-flash prefill kernel):
//   dim3 grid ( nKvHeads, T_q, 1 )
//   dim3 block( ATTN_FLASH_PREFILL_LOCAL, 1, 1 )
//
// Compile-time constant ATTN_FLASH_PREFILL_N_Q_PER_KV_MAX bounds the
// register-array + SLM allocation for per-Q-head state. Runtime
// nQPerKv = nHeads / nKvHeads must be ≤ this cap; dispatch enforces
// the check host-side and falls back to the plain kernel if not.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <math.h>   // for INFINITY

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

extern "C" __global__ __launch_bounds__(ATTN_FLASH_PREFILL_LOCAL)
void attention_prefill_flash_q8_0_gqa(
    const float*         __restrict__ q,
    const unsigned char* __restrict__ k,
    const unsigned char* __restrict__ v,
          float*         __restrict__ out,
    const int                         T_q,
    const int                         nHeads,
    const int                         nKvHeads,
    const int                         headDim,
    const int*           __restrict__ curLenPtr,
    const float                       scale,
    const int                         slidingWindow)
{
    (void)T_q;

    const int hkv     = blockIdx.x;                // KV-head index
    const int pq      = blockIdx.y;                // query position
    const int lid     = threadIdx.x;
    const int nQPerKv = nHeads / nKvHeads;         // >= 1, ≤ N_Q_PER_KV_MAX

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

    // Q-head vector bases: one entry per potentially-active Q-head slot.
    // hq = hkv * nQPerKv + qh.
    const float* qVecs[ATTN_FLASH_PREFILL_N_Q_PER_KV_MAX];
          float* oVecs[ATTN_FLASH_PREFILL_N_Q_PER_KV_MAX];

    #pragma unroll
    for (int qh = 0; qh < ATTN_FLASH_PREFILL_N_Q_PER_KV_MAX; ++qh) {
        const int hq = hkv * nQPerKv + qh;
        qVecs[qh] = q   + static_cast<size_t>(pq) * static_cast<size_t>(qStride)
                        + static_cast<size_t>(hq) * static_cast<size_t>(headDim);
        oVecs[qh] = out + static_cast<size_t>(pq) * static_cast<size_t>(qStride)
                        + static_cast<size_t>(hq) * static_cast<size_t>(headDim);
    }

    // Per-Q-head online-softmax state.
    __shared__ float scores[ATTN_FLASH_PREFILL_N_Q_PER_KV_MAX]
                           [ATTN_FLASH_PREFILL_KTILE];
    __shared__ float oRun  [ATTN_FLASH_PREFILL_N_Q_PER_KV_MAX]
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
    __syncthreads();

    for (int kt = ktStart; kt < nKTiles; ++kt) {
        const int kStartRaw = kt * ATTN_FLASH_PREFILL_KTILE;
        const int kStart    = (kStartRaw > kMin) ? kStartRaw : kMin;
        const int kEndRaw   = kStartRaw + ATTN_FLASH_PREFILL_KTILE;
        const int kEnd      = (kEndRaw < kMax) ? kEndRaw : kMax;
        const int tileLen   = kEnd - kStart;

        // -- Pass A — Q·K scaled scores. K-blocks (scale + 32 int8)
        // loaded ONCE per (kk, blk) pair and reused across every
        // active Q-head. This is the R1 savings vs the plain kernel.
        for (int kk = lid; kk < tileLen; kk += ATTN_FLASH_PREFILL_LOCAL) {
            const unsigned char* __restrict__ kRow = k
                + static_cast<size_t>(kStart + kk)
                * static_cast<size_t>(nBlocksPerRow)
                * static_cast<size_t>(Q8_0_BLOCK_BYTES);
            const unsigned char* __restrict__ kHead = kRow
                + static_cast<size_t>(headBlockBase)
                * static_cast<size_t>(Q8_0_BLOCK_BYTES);

            float acc[ATTN_FLASH_PREFILL_N_Q_PER_KV_MAX];
            #pragma unroll
            for (int qh = 0; qh < ATTN_FLASH_PREFILL_N_Q_PER_KV_MAX; ++qh) {
                acc[qh] = 0.0f;
            }

            for (int blk = 0; blk < nBlocksPerHead; ++blk) {
                const unsigned char* __restrict__ blkPtr = kHead
                    + static_cast<size_t>(blk)
                    * static_cast<size_t>(Q8_0_BLOCK_BYTES);
                const float bscale =
                    __half2float(*reinterpret_cast<const __half*>(blkPtr));
                const signed char* qArr =
                    reinterpret_cast<const signed char*>(blkPtr + 2);
                const int dBase = blk * Q8_0_BLOCK_ELEMENTS;
                for (int in = 0; in < Q8_0_BLOCK_ELEMENTS; ++in) {
                    // Each K-quant scaled once; broadcast to every
                    // active Q-head's dot-product accumulator.
                    const float k_val = bscale * static_cast<float>(qArr[in]);
                    #pragma unroll
                    for (int qh = 0;
                         qh < ATTN_FLASH_PREFILL_N_Q_PER_KV_MAX;
                         ++qh) {
                        acc[qh] += qVecs[qh][dBase + in] * k_val;
                    }
                }
            }

            #pragma unroll
            for (int qh = 0; qh < ATTN_FLASH_PREFILL_N_Q_PER_KV_MAX; ++qh) {
                if (qh >= nQPerKv) continue;
                scores[qh][kk] = acc[qh] * scale;
            }
        }
        __syncthreads();

        // -- Pass B — Online-softmax rescale, per Q-head. --------------
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
            const float mTile = warp16_reduce_max(mTilePart);
            const float mNew  = (m_reg[qh] > mTile) ? m_reg[qh] : mTile;
            alpha[qh]         = expf(m_reg[qh] - mNew);

            float lTilePart = 0.0f;
            for (int kk = lid; kk < tileLen;
                 kk += ATTN_FLASH_PREFILL_LOCAL) {
                const float e = expf(scores[qh][kk] - mNew);
                scores[qh][kk] = e;
                lTilePart += e;
            }
            const float lTile = warp16_reduce_sum(lTilePart);
            l_reg[qh] = alpha[qh] * l_reg[qh] + lTile;
            m_reg[qh] = mNew;
        }
        __syncthreads();

        // -- Pass C — V·softmax → oRun, shared V-load across Q-heads. --
        // Each thread strides over headDim. Per (d, kk) we load one V
        // Q8_0 block header (scale + int8 quant), broadcast the
        // resulting v_val across every active Q-head's accumulator.
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
                const unsigned char* __restrict__ vRow = v
                    + static_cast<size_t>(kStart + kk)
                    * static_cast<size_t>(nBlocksPerRow)
                    * static_cast<size_t>(Q8_0_BLOCK_BYTES);
                const unsigned char* __restrict__ blkPtr = vRow
                    + static_cast<size_t>(headBlockBase + blk)
                    * static_cast<size_t>(Q8_0_BLOCK_BYTES);
                const float bscale =
                    __half2float(*reinterpret_cast<const __half*>(blkPtr));
                const signed char qi =
                    reinterpret_cast<const signed char*>(blkPtr)[2 + in];
                const float v_val = bscale * static_cast<float>(qi);

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
        __syncthreads();
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