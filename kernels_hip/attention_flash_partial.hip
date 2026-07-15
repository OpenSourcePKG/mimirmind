// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// FlashAttention partial-tile kernel — decode mode (T_q == 1).
//
// HIP port of kernels/attention_flash_partial.cl. Each workgroup
// processes ONE K-tile of ONE head and emits a partial (m, l, o)
// triple in the global `partialMlo` scratch. The merge kernel folds
// per-tile partials into the final output using the online-softmax
// identity (Dao et al, 2022):
//
//   m_new = max(m_a, m_b)
//   l_new = exp(m_a - m_new) * l_a + exp(m_b - m_new) * l_b
//   o_new = exp(m_a - m_new) * o_a + exp(m_b - m_new) * o_b
//
// Layouts:
//   q          [1, nHeads,    headDim]
//   k          [T_k, nKvHeads, headDim]
//   v          [T_k, nKvHeads, headDim]
//   partialMlo [nHeads, nKTiles, (2 + headDim)]   global scratch
//
// Per (hq, kt) slot:
//   partialMlo + (hq * nKTiles + kt) * (2 + headDim)
//   - [0]              : m (local max of scaled Q·K within tile)
//   - [1]              : l (sum of exp(score - m) within tile)
//   - [2 .. 2+headDim) : o (sum of exp(score - m) * v[k][d] within tile,
//                           UNNORMALIZED — merge kernel divides by l_total)
//
// Launch geometry:
//   dim3 grid ( nHeads, nKTiles, 1 )
//   dim3 block( ATTN_FLASH_LOCAL, 1, 1 )
//
// WAVE32 note (RDNA3 gfx1101): ATTN_FLASH_LOCAL=16 means the workgroup
// occupies HALF a wave. That is a bit-parity-preserving 1:1 port of
// the L0 kernel; a follow-up perf pass may scale the workgroup to
// 32 (full wave) and adjust the reduction width, but that is a
// numerics-changing optimization, not part of this port.

#include <hip/hip_runtime.h>

#include <math.h>   // for INFINITY

#ifndef ATTN_FLASH_LOCAL
#define ATTN_FLASH_LOCAL 16
#endif

// Kept for parity of interface with the CL variant; not consumed
// directly by HIP (sub-group size is fixed by the arch's warpSize).
#ifndef ATTN_FLASH_SG
#define ATTN_FLASH_SG 16
#endif

#ifndef ATTN_FLASH_KTILE
#define ATTN_FLASH_KTILE 64
#endif

// Butterfly reduction over a 16-lane sub-group. `width=16` limits the
// shuffle to the low-16-lane group of the wave — on RDNA3 that leaves
// the upper 16 lanes untouched (they are inactive at workgroup=16).
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

extern "C" __global__ __launch_bounds__(ATTN_FLASH_LOCAL)
void attention_flash_partial(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
          float* __restrict__ partialMlo,
    const int                 nHeads,
    const int                 nKvHeads,
    const int                 headDim,
    const int* __restrict__   curLenPtr,
    const float               scale,
    const int                 slidingWindow)
{
    const int hq  = blockIdx.x;
    const int kt  = blockIdx.y;
    const int lid = threadIdx.x;

    const int kvStride       = nKvHeads * headDim;
    const int hkv            = (hq * nKvHeads) / nHeads;
    const int positionOffset = curLenPtr[0];

    // Decode mode: T_q == 1. Query attends to keys [kMin, kMax) where
    // kMax = positionOffset + 1 and kMin = max(0, kMax - slidingWindow)
    // (0 for non-SWA layers).
    const int kMax    = positionOffset + 1;
    const int kMin    = (slidingWindow > 0 && kMax > slidingWindow)
                          ? (kMax - slidingWindow) : 0;
    const int nKTiles = (kMax + ATTN_FLASH_KTILE - 1) / ATTN_FLASH_KTILE;
    const int kStartRaw = kt * ATTN_FLASH_KTILE;
    const int kStart  = (kStartRaw > kMin) ? kStartRaw : kMin;
    const int kEndRaw = kStartRaw + ATTN_FLASH_KTILE;
    const int kEnd    = (kEndRaw < kMax) ? kEndRaw : kMax;

    float* __restrict__ mloPtr =
        partialMlo + (static_cast<size_t>(hq) * static_cast<size_t>(nKTiles)
                    + static_cast<size_t>(kt))
                   * static_cast<size_t>(2 + headDim);

    // Past the causal mask OR entirely below the sliding-window low
    // bound — emit a neutral partial so the merge sees no contribution.
    // exp(-inf - mFinal) = 0 → β_t * l_t and β_t * o_t both vanish.
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

    const float* __restrict__ qVec =
        q + static_cast<size_t>(hq) * static_cast<size_t>(headDim);

    __shared__ float scores[ATTN_FLASH_KTILE];

    const int tileLen = kEnd - kStart;

    // Pass 1 — Q·K scores, scaled.
    for (int kk = lid; kk < tileLen; kk += ATTN_FLASH_LOCAL) {
        const int absKk = kStart + kk;
        const float* __restrict__ kVec =
            k + static_cast<size_t>(absKk) * static_cast<size_t>(kvStride)
              + static_cast<size_t>(hkv)  * static_cast<size_t>(headDim);
        float acc = 0.0f;
        for (int d = 0; d < headDim; ++d) {
            acc += qVec[d] * kVec[d];
        }
        scores[kk] = acc * scale;
    }
    __syncthreads();

    // Pass 2 — stable softmax on the tile.
    float mPart = -INFINITY;
    for (int kk = lid; kk < tileLen; kk += ATTN_FLASH_LOCAL) {
        const float s = scores[kk];
        if (s > mPart) mPart = s;
    }
    const float mLocal = warp16_reduce_max(mPart);

    float lPart = 0.0f;
    for (int kk = lid; kk < tileLen; kk += ATTN_FLASH_LOCAL) {
        const float e = expf(scores[kk] - mLocal);
        scores[kk] = e;
        lPart += e;
    }
    const float lLocal = warp16_reduce_sum(lPart);
    __syncthreads();

    // Pass 3 — unnormalized V-weighted sum: o_partial[d] = sum_kk
    // exp(scores[kk] - mLocal) * v[kStart+kk][hkv][d]. Stripe d across
    // threads. The merge kernel divides by l_total at the end.
    for (int d = lid; d < headDim; d += ATTN_FLASH_LOCAL) {
        float acc = 0.0f;
        for (int kk = 0; kk < tileLen; ++kk) {
            const float* __restrict__ vVec =
                v + static_cast<size_t>(kStart + kk) * static_cast<size_t>(kvStride)
                  + static_cast<size_t>(hkv)         * static_cast<size_t>(headDim);
            acc += scores[kk] * vVec[d];
        }
        mloPtr[2 + d] = acc;
    }

    if (lid == 0) {
        mloPtr[0] = mLocal;
        mloPtr[1] = lLocal;
    }
}