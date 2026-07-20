// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
// Ported from kernels_hip/attention_prefill_flash_fp16.hip — Track 4 mechanical port, no functional change intended.

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// FP16-KV variant of attention_prefill_flash.hip. Single-workgroup
// streaming FlashAttention for T_q > 1, reading K/V as __half instead
// of fp32. Loads promote to fp32 in registers via __half2float, so the
// online-softmax (m, l, o) state stays fully fp32-precision.
//
// Everything else — SLM layout, ktile geometry, Pass A/B/C structure,
// SWA masking, curLenPtr wiring — is bit-for-bit identical to the f32
// variant so the KV-dtype swap stays isolated to load-sites.
//
// Invoked only when the KV cache is fp16-typed.

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
void attention_prefill_flash_fp16(
    const float*  __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
          float*  __restrict__ out,
    const int                  T_q,
    const int                  nHeads,
    const int                  nKvHeads,
    const int                  headDim,
    const int*    __restrict__ curLenPtr,
    const float                scale,
    const int                  slidingWindow)
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

        // -- Pass A — Q·K scaled scores. K promotes fp16 → fp32. ------
        for (int kk = lid; kk < tileLen; kk += ATTN_FLASH_PREFILL_LOCAL) {
            const __half* __restrict__ kVec =
                k + static_cast<size_t>(kStart + kk) * static_cast<size_t>(kvStride)
                  + static_cast<size_t>(hkv) * static_cast<size_t>(headDim);
            float acc = 0.0f;
            for (int d = 0; d < headDim; ++d) {
                acc += qVec[d] * __half2float(kVec[d]);
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

        // -- Pass C — scale-and-accumulate V into oRun. ---------------
        for (int d = lid; d < headDim; d += ATTN_FLASH_PREFILL_LOCAL) {
            float acc = alpha * oRun[d];
            for (int kk = 0; kk < tileLen; ++kk) {
                const __half* __restrict__ vVec =
                    v + static_cast<size_t>(kStart + kk) * static_cast<size_t>(kvStride)
                      + static_cast<size_t>(hkv) * static_cast<size_t>(headDim);
                acc += scores[kk] * __half2float(vVec[d]);
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