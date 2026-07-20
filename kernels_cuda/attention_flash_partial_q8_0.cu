// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
// Ported from kernels_hip/attention_flash_partial_q8_0.hip — Track 4 mechanical port, no functional change intended.

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Q8_0-KV variant of attention_flash_partial.hip. Decode-mode
// FlashAttention pass 1 (T_q == 1), reading K/V as Q8_0 blocks (32
// elements per block: fp16 scale + 32 int8 quants = 34 B). Loads
// dequantise on the fly in registers; the per-tile
// (m, l, o_unnorm) partials written to `partialMlo` stay fp32 so
// the existing attention_flash_merge kernel is unchanged and works
// for all KV dtypes.
//
// Row / head layout matches attention_q8_0.hip: one K/V row is
// (kvDim / 32) blocks of 34 B; hkv * headDim is a multiple of 32 for
// the models we run.
//
// Invoked only when the KV cache is Q8_0-typed.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <math.h>   // for INFINITY

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
void attention_flash_partial_q8_0(
    const float*         __restrict__ q,
    const unsigned char* __restrict__ k,
    const unsigned char* __restrict__ v,
          float*         __restrict__ partialMlo,
    const int                         nHeads,
    const int                         nKvHeads,
    const int                         headDim,
    const int*           __restrict__ curLenPtr,
    const float                       scale,
    const int                         slidingWindow)
{
    const int hq  = blockIdx.x;
    const int kt  = blockIdx.y;
    const int lid = threadIdx.x;

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

    float* __restrict__ mloPtr =
        partialMlo + (static_cast<size_t>(hq) * static_cast<size_t>(nKTiles)
                    + static_cast<size_t>(kt))
                   * static_cast<size_t>(2 + headDim);

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

    // Pass 1 — Q·K scores, scaled. Dequant per block: read scale
    // once, dot 32 elements against qVec, fold into acc with one
    // scale mul.
    for (int kk = lid; kk < tileLen; kk += ATTN_FLASH_LOCAL) {
        const int absKk = kStart + kk;
        const unsigned char* __restrict__ kRow = k
            + static_cast<size_t>(absKk) * static_cast<size_t>(nBlocksPerRow)
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

    // Pass 3 — unnormalized V-weighted sum. Stripe d across threads.
    // Per (kk, d) we load one Q8_0 block header (scale) plus one int8
    // quant — adjacent threads in the WG hit the same block.
    for (int d = lid; d < headDim; d += ATTN_FLASH_LOCAL) {
        const int blk = d / Q8_0_BLOCK_ELEMENTS;
        const int in  = d % Q8_0_BLOCK_ELEMENTS;
        float acc = 0.0f;
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
            acc += scores[kk] * bscale * static_cast<float>(qi);
        }
        mloPtr[2 + d] = acc;
    }

    if (lid == 0) {
        mloPtr[0] = mLocal;
        mloPtr[1] = lLocal;
    }
}
