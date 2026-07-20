// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
// Ported from kernels_hip/attention.hip — Track 4 mechanical port, no functional change intended.

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Multi-head self-attention, GQA-aware, causal. HIP port of
// kernels/attention.cl — the untiled M5f.3 variant (a). Drop-in
// equivalent of compute::multiHeadAttention resident on the GPU.
//
// Layout (all row-major fp32):
//   q:   [T_q, nHeads,    headDim]
//   k:   [T_k, nKvHeads,  headDim]
//   v:   [T_k, nKvHeads,  headDim]
//   out: [T_q, nHeads,    headDim]
//
// Causal + SWA masking:
//   kMax = positionOffset + pq + 1
//   kMin = (slidingWindow > 0 && kMax > slidingWindow)
//            ? (kMax - slidingWindow) : 0
// slidingWindow == 0 degenerates to pure causal.
//
// GQA: query head hq reads from KV head hkv = (hq * nKvHeads) / nHeads.
//
// Workgroup geometry:
//   dim3 grid ( nHeads, T_q, 1 )
//   dim3 block( ATTN_LOCAL, 1, 1 )  // 16 threads == half-wave on gfx1101
//
// LDS use: scores[ATTN_MAX_TK] = 64 KiB per WG at MAX_TK=16384. This
// matches the Arc Xe-LPG ceiling that the L0 kernel was designed for;
// on RDNA3 gfx1101 each CU has ~128 KiB LDS, so occupancy is capped
// at ≤2 blocks / CU — the same "under-saturation for small nHeads*T_q"
// motivation that led to the flash-partial + merge pair. For larger
// contexts (T_k > MAX_TK) callers must use the flash path.
//
// WAVE32 note: sub_group_reduce_max / _add on 16 Lanes become
// __shfl_xor(v, off, 16) butterfly steps, identical to the reduction
// helpers used by attention_flash_partial.hip.

#include <cuda_runtime.h>

#include <math.h>   // for INFINITY

#ifndef ATTN_LOCAL
#define ATTN_LOCAL 16
#endif

// Kept for interface parity with the CL variant; not consumed here —
// warpSize is fixed at 32 on RDNA3 and the shuffles use width=16.
#ifndef ATTN_SG
#define ATTN_SG 16
#endif

#ifndef ATTN_MAX_TK
#define ATTN_MAX_TK 16384
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

extern "C" __global__ __launch_bounds__(ATTN_LOCAL)
void attention(
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

    const float* __restrict__ qVec =
        q + static_cast<size_t>(pq) * static_cast<size_t>(qStride)
          + static_cast<size_t>(hq) * static_cast<size_t>(headDim);
          float* __restrict__ oVec =
        out + static_cast<size_t>(pq) * static_cast<size_t>(qStride)
            + static_cast<size_t>(hq) * static_cast<size_t>(headDim);

    __shared__ float scores[ATTN_MAX_TK];

    // -- Pass 1 — Q·K dot products, scaled. ---------------------------
    for (int kk = kMin + lid; kk < kMax; kk += ATTN_LOCAL) {
        const float* __restrict__ kVec =
            k + static_cast<size_t>(kk) * static_cast<size_t>(kvStride)
              + static_cast<size_t>(hkv) * static_cast<size_t>(headDim);
        float acc = 0.0f;
        for (int d = 0; d < headDim; ++d) {
            acc += qVec[d] * kVec[d];
        }
        scores[kk - kMin] = acc * scale;
    }
    __syncthreads();

    // -- Pass 2 — stable softmax over [kMin, kMax). -------------------
    float mPart = -INFINITY;
    for (int kk = kMin + lid; kk < kMax; kk += ATTN_LOCAL) {
        const float s = scores[kk - kMin];
        if (s > mPart) mPart = s;
    }
    const float maxScore = warp16_reduce_max(mPart);

    float lPart = 0.0f;
    for (int kk = kMin + lid; kk < kMax; kk += ATTN_LOCAL) {
        const float e = expf(scores[kk - kMin] - maxScore);
        scores[kk - kMin] = e;
        lPart += e;
    }
    const float sumExp = warp16_reduce_sum(lPart);
    const float invSum = 1.0f / sumExp;
    __syncthreads();

    // -- Pass 3 — out[d] = sum_kk softmax_unnorm[kk] * v[kk][d] / sum.
    for (int d = lid; d < headDim; d += ATTN_LOCAL) {
        float acc = 0.0f;
        for (int kk = kMin; kk < kMax; ++kk) {
            const float* __restrict__ vVec =
                v + static_cast<size_t>(kk) * static_cast<size_t>(kvStride)
                  + static_cast<size_t>(hkv) * static_cast<size_t>(headDim);
            acc += scores[kk - kMin] * vVec[d];
        }
        oVec[d] = acc * invSum;
    }
}