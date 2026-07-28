// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Q8_0-KV variant of attention.hip. K and V are stored as ggml-style
// Q8_0 blocks (32 elements per block: fp16 scale + 32 int8 quants =
// 34 B per block). Q and OUT stay fp32. Loads dequantise on the fly
// in registers (block scale × int8 → fp32); the softmax state stays
// fully fp32-precision.
//
// Row layout: one K/V row is (kvDim / 32) Q8_0 blocks × 34 B. Since
// `hkv * headDim` is a multiple of 32 for every architecture we run
// (Qwen 2.5 headDim=64, Gemma 4 headDim=256/512) the per-head block
// origin lands on a block boundary. The `headDim % 32 == 0` invariant
// is enforced by KvCache's ctor assertion.
//
// Invoked only when the KV cache is Q8_0-typed; the f32 path stays on
// attention.hip and the fp16 path stays on attention_fp16.hip.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <math.h>   // for INFINITY

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
void attention_q8_0(
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

    const int hq  = blockIdx.x;
    const int pq  = blockIdx.y;
    const int lid = threadIdx.x;

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

    const float* __restrict__ qVec =
        q + static_cast<size_t>(pq) * static_cast<size_t>(qStride)
          + static_cast<size_t>(hq) * static_cast<size_t>(headDim);
          float* __restrict__ oVec =
        out + static_cast<size_t>(pq) * static_cast<size_t>(qStride)
            + static_cast<size_t>(hq) * static_cast<size_t>(headDim);

    __shared__ float scores[ATTN_MAX_TK];

    // -- Pass 1 — Q·K dot products, scaled. --------------------------
    // For each K row, iterate the `nBlocksPerHead` blocks that belong
    // to head `hkv`. The block scale loads once per block; the 32
    // int8 quants are then multiplied against the matching 32 fp32
    // qVec entries, and the block-local dot is folded into `acc` with
    // one scale multiply.
    for (int kk = kMin + lid; kk < kMax; kk += ATTN_LOCAL) {
        const unsigned char* __restrict__ kRow = k
            + static_cast<size_t>(kk) * static_cast<size_t>(nBlocksPerRow)
            * static_cast<size_t>(Q8_0_BLOCK_BYTES);
        const unsigned char* __restrict__ kHead = kRow
            + static_cast<size_t>(headBlockBase)
            * static_cast<size_t>(Q8_0_BLOCK_BYTES);
        float acc = 0.0f;
        for (int blk = 0; blk < nBlocksPerHead; ++blk) {
            const unsigned char* __restrict__ blkPtr = kHead
                + static_cast<size_t>(blk)
                * static_cast<size_t>(Q8_0_BLOCK_BYTES);
            const float bscale =
                __half2float(*reinterpret_cast<const __half*>(blkPtr));
            const signed char* qArr =
                reinterpret_cast<const signed char*>(blkPtr + 2);
            const int dBase = blk * Q8_0_BLOCK_ELEMENTS;
            float sub = 0.0f;
            for (int in = 0; in < Q8_0_BLOCK_ELEMENTS; ++in) {
                sub += qVec[dBase + in] * static_cast<float>(qArr[in]);
            }
            acc += bscale * sub;
        }
        scores[kk - kMin] = acc * scale;
    }
    __syncthreads();

    // -- Pass 2 — stable softmax over [kMin, kMax). ------------------
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
    // Per (kk, d) we load one Q8_0 block header (scale) plus one int8
    // quant. Adjacent threads on the same WG hit the same block, so
    // the scale-load coalesces in the L1 cache.
    for (int d = lid; d < headDim; d += ATTN_LOCAL) {
        const int blk = d / Q8_0_BLOCK_ELEMENTS;
        const int in  = d % Q8_0_BLOCK_ELEMENTS;
        float acc = 0.0f;
        for (int kk = kMin; kk < kMax; ++kk) {
            const unsigned char* __restrict__ vRow = v
                + static_cast<size_t>(kk) * static_cast<size_t>(nBlocksPerRow)
                * static_cast<size_t>(Q8_0_BLOCK_BYTES);
            const unsigned char* __restrict__ blkPtr = vRow
                + static_cast<size_t>(headBlockBase + blk)
                * static_cast<size_t>(Q8_0_BLOCK_BYTES);
            const float bscale =
                __half2float(*reinterpret_cast<const __half*>(blkPtr));
            const signed char qi =
                reinterpret_cast<const signed char*>(blkPtr)[2 + in];
            acc += scores[kk - kMin] * bscale * static_cast<float>(qi);
        }
        oVec[d] = acc * invSum;
    }
}