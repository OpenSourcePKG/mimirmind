// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
// Ported from kernels_hip/attention_prefill_flash.hip — Track 4 mechanical port, no functional change intended.

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// FlashAttention prefill kernel — single-workgroup streaming online-
// softmax per Dao et al. 2022 (original formulation, not the 2-kernel
// partial/merge variant used for decode).
//
// One workgroup owns exactly one (hq, pq) and streams K tile-by-tile
// through local memory. (m, l, o) live intra-WG so there is NO global
// scratch buffer, which is what lets this scale to arbitrary T_k
// without the O(T_q * K_TILES * headDim) blow-up that blocks the
// decode-flash pattern from being reused for prefill.
//
// Layouts (all row-major fp32, identical to attention.hip):
//   q   [T_q, nHeads,    headDim]
//   k   [T_k, nKvHeads,  headDim]
//   v   [T_k, nKvHeads,  headDim]
//   out [T_q, nHeads,    headDim]
//
// Causal mask + sliding window: kMax = positionOffset + pq + 1,
// kMin = max(0, kMax - slidingWindow) if slidingWindow > 0.
//
// GQA: hq → hkv = (hq * nKvHeads) / nHeads.
//
// Launch geometry:
//   dim3 grid ( nHeads, T_q, 1 )
//   dim3 block( ATTN_FLASH_PREFILL_LOCAL, 1, 1 )   // 16 == half-wave
//
// SLM per workgroup:
//   scores [KTILE=128]         ~=  0.5 KiB
//   oRun   [MAX_HEADDIM=512]   ~=  2.0 KiB
//                              ~=  2.5 KiB total
// (Compare against attention.hip variant (a) with 64 KiB scores[16384]
// — the occupancy killer this kernel is here to fix.)

#include <cuda_runtime.h>

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

static __device__ __forceinline__ float warp16_reduce_max(float v) {
    v = fmaxf(v, __shfl_xor(v, 8, 16));
    v = fmaxf(v, __shfl_xor(v, 4, 16));
    v = fmaxf(v, __shfl_xor(v, 2, 16));
    v = fmaxf(v, __shfl_xor(v, 1, 16));
    return v;
}

static __device__ __forceinline__ float warp16_reduce_sum(float v) {
    v += __shfl_xor(v, 8, 16);
    v += __shfl_xor(v, 4, 16);
    v += __shfl_xor(v, 2, 16);
    v += __shfl_xor(v, 1, 16);
    return v;
}

extern "C" __global__ __launch_bounds__(ATTN_FLASH_PREFILL_LOCAL)
void attention_prefill_flash(
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

    const int hq  = blockIdx.x;
    const int pq  = blockIdx.y;
    const int lid = threadIdx.x;

    const int qStride        = nHeads   * headDim;
    const int kvStride       = nKvHeads * headDim;
    const int hkv            = (hq * nKvHeads) / nHeads;
    const int positionOffset = curLenPtr[0];
    const int absPos         = positionOffset + pq;
    const int kMax           = absPos + 1;
    const int kMin           = (slidingWindow > 0 && kMax > slidingWindow)
                                 ? (kMax - slidingWindow) : 0;
    const int ktStart        = kMin / ATTN_FLASH_PREFILL_KTILE;
    const int nKTiles        = (kMax + ATTN_FLASH_PREFILL_KTILE - 1)
                               / ATTN_FLASH_PREFILL_KTILE;

    const float* __restrict__ qVec =
        q + static_cast<size_t>(pq) * static_cast<size_t>(qStride)
          + static_cast<size_t>(hq) * static_cast<size_t>(headDim);
          float* __restrict__ oVec =
        out + static_cast<size_t>(pq) * static_cast<size_t>(qStride)
            + static_cast<size_t>(hq) * static_cast<size_t>(headDim);

    __shared__ float scores[ATTN_FLASH_PREFILL_KTILE];
    __shared__ float oRun  [ATTN_FLASH_PREFILL_MAX_HEADDIM];

    // Running (m, l) live in registers, replicated across the 16
    // subgroup threads via warp16_reduce_*. oRun lives in SLM
    // because each thread only touches its strided d's.
    float m = -INFINITY;
    float l = 0.0f;
    for (int d = lid; d < headDim; d += ATTN_FLASH_PREFILL_LOCAL) {
        oRun[d] = 0.0f;
    }
    __syncthreads();

    for (int kt = ktStart; kt < nKTiles; ++kt) {
        const int kStartRaw = kt * ATTN_FLASH_PREFILL_KTILE;
        const int kStart    = (kStartRaw > kMin) ? kStartRaw : kMin;
        const int kEndRaw   = kStartRaw + ATTN_FLASH_PREFILL_KTILE;
        const int kEnd      = (kEndRaw < kMax) ? kEndRaw : kMax;
        const int tileLen   = kEnd - kStart;

        // -- Pass A — Q·K scaled scores for this K-tile. --------------
        for (int kk = lid; kk < tileLen; kk += ATTN_FLASH_PREFILL_LOCAL) {
            const float* __restrict__ kVec =
                k + static_cast<size_t>(kStart + kk) * static_cast<size_t>(kvStride)
                  + static_cast<size_t>(hkv) * static_cast<size_t>(headDim);
            float acc = 0.0f;
            for (int d = 0; d < headDim; ++d) {
                acc += qVec[d] * kVec[d];
            }
            scores[kk] = acc * scale;
        }
        __syncthreads();

        // -- Pass B — Online-softmax rescale (Dao 2022 identity). -----
        //   m_new = max(m_prev, m_tile)
        //   alpha = exp(m_prev - m_new)
        //   l_new = alpha * l_prev + l_tile'
        //   o_new = alpha * o_prev + sum(beta_kk * v)   (Pass C below)
        float mTilePart = -INFINITY;
        for (int kk = lid; kk < tileLen; kk += ATTN_FLASH_PREFILL_LOCAL) {
            const float s = scores[kk];
            if (s > mTilePart) mTilePart = s;
        }
        const float mTile = warp16_reduce_max(mTilePart);
        const float mNew  = (m > mTile) ? m : mTile;
        // First iter: m == -INFINITY, so alpha = exp(-inf - mNew) = 0
        // — IEEE-correct, prior-l and prior-o contributions vanish
        // exactly as required.
        const float alpha = expf(m - mNew);

        float lTilePart = 0.0f;
        for (int kk = lid; kk < tileLen; kk += ATTN_FLASH_PREFILL_LOCAL) {
            const float e = expf(scores[kk] - mNew);
            scores[kk] = e;
            lTilePart += e;
        }
        const float lTile = warp16_reduce_sum(lTilePart);
        l = alpha * l + lTile;
        m = mNew;
        __syncthreads();

        // -- Pass C — scale-and-accumulate V into oRun. ---------------
        for (int d = lid; d < headDim; d += ATTN_FLASH_PREFILL_LOCAL) {
            float acc = alpha * oRun[d];
            for (int kk = 0; kk < tileLen; ++kk) {
                const float* __restrict__ vVec =
                    v + static_cast<size_t>(kStart + kk) * static_cast<size_t>(kvStride)
                      + static_cast<size_t>(hkv) * static_cast<size_t>(headDim);
                acc += scores[kk] * vVec[d];
            }
            oRun[d] = acc;
        }
        __syncthreads();
    }

    // Normalise. l == 0 only in the degenerate all-masked case which
    // cannot happen for pq >= 0 (kMax >= 1); guard emits 0 for hygiene.
    const float invL = (l > 0.0f) ? (1.0f / l) : 0.0f;
    for (int d = lid; d < headDim; d += ATTN_FLASH_PREFILL_LOCAL) {
        oVec[d] = oRun[d] * invL;
    }
}