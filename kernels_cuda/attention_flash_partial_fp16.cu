// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
// Ported from kernels_hip/attention_flash_partial_fp16.hip — Track 4 mechanical port, no functional change intended.

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// FP16-KV variant of attention_flash_partial.hip. Decode-mode
// FlashAttention pass 1 (T_q == 1), reading K/V as __half instead of
// float. Loads promote to fp32 in registers via __half2float; the
// per-tile (m, l, o_unnorm) partials written to `partialMlo` stay
// fp32 so the existing attention_flash_merge kernel works unchanged
// for both KV dtypes.
//
// Everything else — tile geometry, SLM layout, SWA masking, Pass 1/2/3
// structure, curLenPtr wiring, warp16 butterfly reduction — is bit-
// for-bit identical to the f32 variant so the KV-dtype swap stays
// isolated to load-sites.
//
// Invoked only when the KV cache is fp16-typed; the f32 dispatch
// stays on attention_flash_partial.hip.

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
void attention_flash_partial_fp16(
    const float*  __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
          float*  __restrict__ partialMlo,
    const int                  nHeads,
    const int                  nKvHeads,
    const int                  headDim,
    const int*    __restrict__ curLenPtr,
    const float                scale,
    const int                  slidingWindow)
{
    const int hq  = blockIdx.x;
    const int kt  = blockIdx.y;
    const int lid = threadIdx.x;

    const int kvStride       = nKvHeads * headDim;
    const int hkv            = (hq * nKvHeads) / nHeads;
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

    // Pass 1 — Q·K scores, scaled. K promotes fp16 → fp32 per read.
    for (int kk = lid; kk < tileLen; kk += ATTN_FLASH_LOCAL) {
        const int absKk = kStart + kk;
        const __half* __restrict__ kVec =
            k + static_cast<size_t>(absKk) * static_cast<size_t>(kvStride)
              + static_cast<size_t>(hkv)  * static_cast<size_t>(headDim);
        float acc = 0.0f;
        for (int d = 0; d < headDim; ++d) {
            acc += qVec[d] * __half2float(kVec[d]);
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

    // Pass 3 — unnormalized V-weighted sum. V promotes fp16 → fp32.
    for (int d = lid; d < headDim; d += ATTN_FLASH_LOCAL) {
        float acc = 0.0f;
        for (int kk = 0; kk < tileLen; ++kk) {
            const __half* __restrict__ vVec =
                v + static_cast<size_t>(kStart + kk) * static_cast<size_t>(kvStride)
                  + static_cast<size_t>(hkv)         * static_cast<size_t>(headDim);
            acc += scores[kk] * __half2float(vVec[d]);
        }
        mloPtr[2 + d] = acc;
    }

    if (lid == 0) {
        mloPtr[0] = mLocal;
        mloPtr[1] = lLocal;
    }
}