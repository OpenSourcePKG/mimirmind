// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// P3.a — GQA-head-packed variant of attention_prefill_flash.cu for the
// F32 KV path (the only prefill-attention kernel Qwen3-Next 35B reaches,
// since its KV cache is F32-only).
//
// Colfax-style query-head packing: instead of one workgroup per
// (query-head, query-position), one workgroup handles ALL query heads
// of a KV-group at a given query position. The K and V rows are read
// ONCE per K-tile step and reused across every query head in the group
// — cuts KV-load bandwidth by the GQA ratio (nHeads / nKvHeads). For
// Qwen3-Next (16 heads / 2 KV-heads = 8 per group) that is an 8x KV
// bandwidth reduction.
//
// The per-(query-head, key) math is byte-identical to the plain kernel
// (same natural-order headDim dot, same online-softmax, same
// accumulation order), so the output is bit-exact with
// attention_prefill_flash — parity is by construction. The only change
// is that K/V rows are loaded once per group and broadcast to every
// active Q-head's accumulator.
//
// Layouts (all row-major fp32, identical to attention_prefill_flash):
//   q   [T_q, nHeads,    headDim]
//   k   [T_k, nKvHeads,  headDim]
//   v   [T_k, nKvHeads,  headDim]
//   out [T_q, nHeads,    headDim]
//
// Launch geometry (differs from the plain-flash prefill kernel):
//   dim3 grid ( nKvHeads, T_q, 1 )
//   dim3 block( ATTN_FLASH_PREFILL_LOCAL, 1, 1 )   // 16 == half-wave
//
// Compile-time ATTN_FLASH_PREFILL_N_Q_PER_KV_MAX bounds the register-
// array + SLM allocation for per-Q-head state. Runtime nQPerKv =
// nHeads / nKvHeads must be <= this cap; dispatch enforces the check
// host-side and falls back to the plain kernel otherwise.

#include <cuda_runtime.h>

#include <math.h>   // for INFINITY

#ifndef ATTN_FLASH_PREFILL_LOCAL
#define ATTN_FLASH_PREFILL_LOCAL 16
#endif

#ifndef ATTN_FLASH_PREFILL_KTILE
#define ATTN_FLASH_PREFILL_KTILE 128
#endif

#ifndef ATTN_FLASH_PREFILL_MAX_HEADDIM
#define ATTN_FLASH_PREFILL_MAX_HEADDIM 512
#endif

// Max query heads per KV group. Sized for Qwen-style GQA (up to 8:1);
// dispatch guards nQPerKv <= this value and falls back otherwise.
#ifndef ATTN_FLASH_PREFILL_N_Q_PER_KV_MAX
#define ATTN_FLASH_PREFILL_N_Q_PER_KV_MAX 8
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

extern "C" __global__ __launch_bounds__(ATTN_FLASH_PREFILL_LOCAL)
void attention_prefill_flash_f32_gqa(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
          float* __restrict__ out,
    const int                 T_q,
    const int                 nHeads,
    const int                 nKvHeads,
    const int                 headDim,
    const int* __restrict__   curLenPtr,
    const float               scale,
    const int                 slidingWindow)
{
    (void)T_q;

    const int hkv     = blockIdx.x;                // KV-head index
    const int pq      = blockIdx.y;                // query position
    const int lid     = threadIdx.x;
    const int nQPerKv = nHeads / nKvHeads;         // >= 1, <= N_Q_PER_KV_MAX

    const int qStride        = nHeads   * headDim;
    const int kvStride       = nKvHeads * headDim;
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

        // -- Pass A — Q·K scaled scores. The K row is read ONCE and its
        // elements broadcast to every active Q-head's dot accumulator
        // (the GQA savings vs the plain kernel). Natural d-order keeps
        // the per-(head,key) dot bit-identical to attention_prefill_flash.
        for (int kk = lid; kk < tileLen; kk += ATTN_FLASH_PREFILL_LOCAL) {
            const float* __restrict__ kVec =
                k + static_cast<size_t>(kStart + kk) * static_cast<size_t>(kvStride)
                  + static_cast<size_t>(hkv) * static_cast<size_t>(headDim);

            float acc[ATTN_FLASH_PREFILL_N_Q_PER_KV_MAX];
            #pragma unroll
            for (int qh = 0; qh < ATTN_FLASH_PREFILL_N_Q_PER_KV_MAX; ++qh) {
                acc[qh] = 0.0f;
            }

            for (int d = 0; d < headDim; ++d) {
                const float k_val = kVec[d];
                #pragma unroll
                for (int qh = 0; qh < ATTN_FLASH_PREFILL_N_Q_PER_KV_MAX; ++qh) {
                    acc[qh] += qVecs[qh][d] * k_val;
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
            for (int kk = lid; kk < tileLen; kk += ATTN_FLASH_PREFILL_LOCAL) {
                const float s = scores[qh][kk];
                if (s > mTilePart) mTilePart = s;
            }
            const float mTile = warp16_reduce_max(mTilePart);
            const float mNew  = (m_reg[qh] > mTile) ? m_reg[qh] : mTile;
            alpha[qh]         = expf(m_reg[qh] - mNew);

            float lTilePart = 0.0f;
            for (int kk = lid; kk < tileLen; kk += ATTN_FLASH_PREFILL_LOCAL) {
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
        // Each thread strides over headDim. The V element at d is read
        // ONCE per (d, kk) and broadcast to every active Q-head.
        for (int d = lid; d < headDim; d += ATTN_FLASH_PREFILL_LOCAL) {
            float acc_v[ATTN_FLASH_PREFILL_N_Q_PER_KV_MAX];
            #pragma unroll
            for (int qh = 0; qh < ATTN_FLASH_PREFILL_N_Q_PER_KV_MAX; ++qh) {
                acc_v[qh] = (qh < nQPerKv) ? (alpha[qh] * oRun[qh][d]) : 0.0f;
            }

            for (int kk = 0; kk < tileLen; ++kk) {
                const float* __restrict__ vVec =
                    v + static_cast<size_t>(kStart + kk) * static_cast<size_t>(kvStride)
                      + static_cast<size_t>(hkv) * static_cast<size_t>(headDim);
                const float v_val = vVec[d];
                #pragma unroll
                for (int qh = 0; qh < ATTN_FLASH_PREFILL_N_Q_PER_KV_MAX; ++qh) {
                    acc_v[qh] += scores[qh][kk] * v_val;
                }
            }

            #pragma unroll
            for (int qh = 0; qh < ATTN_FLASH_PREFILL_N_Q_PER_KV_MAX; ++qh) {
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
