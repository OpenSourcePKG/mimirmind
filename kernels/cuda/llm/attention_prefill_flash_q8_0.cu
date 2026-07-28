// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
// Ported from kernels_hip/attention_prefill_flash_q8_0.hip — Track 4 mechanical port, no functional change intended.

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Q8_0-KV variant of attention_prefill_flash.hip. Single-workgroup
// streaming FlashAttention for T_q > 1, reading K/V as Q8_0 blocks
// (32 elements per block: fp16 scale + 32 int8 quants = 34 B). Loads
// dequantise on the fly in registers; the online-softmax state (m, l,
// o) stays fully fp32-precision, matching the F32 / FP16 variants.
//
// Row / head layout matches the other Q8_0 attention kernels: one K/V
// row is (kvDim / 32) blocks of 34 B; hkv * headDim is a multiple of
// 32 for every model we run. The K-tile of 128 keys iterates
// headDim/32 Q8_0 blocks per K row.
//
// Invoked only when the KV cache is Q8_0-typed.

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
void attention_prefill_flash_q8_0(
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

        // -- Pass A — Q·K scaled scores. Dequant per Q8_0 block. ------
        for (int kk = lid; kk < tileLen; kk += ATTN_FLASH_PREFILL_LOCAL) {
            const unsigned char* __restrict__ kRow = k
                + static_cast<size_t>(kStart + kk)
                * static_cast<size_t>(nBlocksPerRow)
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

        // -- Pass B — Online-softmax rescale. -------------------------
        float mTilePart = -INFINITY;
        for (int kk = lid; kk < tileLen; kk += ATTN_FLASH_PREFILL_LOCAL) {
            const float s = scores[kk];
            if (s > mTilePart) mTilePart = s;
        }
        const float mTile = warp16_reduce_max(mTilePart);
        const float mNew  = (m > mTile) ? m : mTile;
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

        // -- Pass C — scale-and-accumulate V into oRun (Q8_0-dequant). --
        for (int d = lid; d < headDim; d += ATTN_FLASH_PREFILL_LOCAL) {
            const int blk = d / Q8_0_BLOCK_ELEMENTS;
            const int in  = d % Q8_0_BLOCK_ELEMENTS;
            float acc = alpha * oRun[d];
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
            oRun[d] = acc;
        }
        __syncthreads();
    }

    const float invL = (l > 0.0f) ? (1.0f / l) : 0.0f;
    for (int d = lid; d < headDim; d += ATTN_FLASH_PREFILL_LOCAL) {
        oVec[d] = oRun[d] * invL;
    }
}