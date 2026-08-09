// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Per-head channel slice + optional GQA head repeat. CUDA port of
// kernels/gather_heads_from_channels.cl. CPU reference:
// compute::gatherHeadsFromChannels. Launch: grid ceil(T*dstHeads*S/L), block L.

#include <cuda_runtime.h>

#ifndef GATHER_HEADS_LOCAL
#define GATHER_HEADS_LOCAL 256
#endif

extern "C" __global__ __launch_bounds__(GATHER_HEADS_LOCAL)
void gather_heads_from_channels(
    const float* __restrict__ src,
    float*       __restrict__ dst,
    const int                 T,
    const int                 offset,
    const int                 srcHeads,
    const int                 dstHeads,
    const int                 S,
    const int                 convTotalWidth)
{
    const int gid   = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = T * dstHeads * S;
    if (gid >= total) {
        return;
    }
    const int s   = gid % S;
    const int rem = gid / S;
    const int hd  = rem % dstHeads;
    const int t   = rem / dstHeads;
    const int srcHead = hd % srcHeads;
    dst[gid] = src[(size_t)t * convTotalWidth + offset + (size_t)srcHead * S + s];
}

// GDN-Inc 2b (2026-08-09): fused post-conv prep (vLLM fused_post_conv_prep).
// Replaces the 3 gather_heads_from_channels launches (q/k/v split with GQA
// repeat) + 2 l2_norm launches (q, k) with ONE launch. One thread per (t, head)
// output row: gathers its S channels for q/k/v directly from the conv output,
// L2-normalises q and k over head_dim S, copies v raw. The gather index map and
// the L2 sumsq accumulation order (s = 0..S-1) are identical to the separate
// kernels, so the result is BIT-IDENTICAL. q/k use srcHeadsKV (GQA repeat); v
// has dstHeads == srcHeads so its source head is the row head directly.
// Layout of qkvMixed per token: [ q: hK*S | k: hK*S | v: hV*S ], keyDim = hK*S.
// Launch: grid ceil(T*dstHeads / L), block L.
extern "C" __global__ __launch_bounds__(GATHER_HEADS_LOCAL)
void fused_post_conv_prep(
    const float* __restrict__ qkvMixed,   // [T, convTotalWidth]
    float*       __restrict__ qOut,       // [T, dstHeads, S]  (L2-normed)
    float*       __restrict__ kOut,       // [T, dstHeads, S]  (L2-normed)
    float*       __restrict__ vOut,       // [T, dstHeads, S]  (raw)
    const int                 T,
    const int                 srcHeadsKV, // q/k source heads (GQA), = hK
    const int                 dstHeads,   // = hV
    const int                 S,
    const int                 convTotalWidth,
    const int                 keyDim,     // = hK * S
    const float               eps)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;   // (t, h)
    if (row >= T * dstHeads) {
        return;
    }
    const int h = row % dstHeads;
    const int t = row / dstHeads;
    const int srcHeadKV = h % srcHeadsKV;                    // GQA repeat for q/k

    const size_t base   = (size_t)t * convTotalWidth;
    const size_t qSrc   = base + (size_t)srcHeadKV * S;                 // offset 0
    const size_t kSrc   = base + (size_t)keyDim + (size_t)srcHeadKV * S;
    const size_t vSrc   = base + (size_t)2 * keyDim + (size_t)h * S;    // v: no repeat
    const size_t dstRow = (size_t)row * S;

    // q: gather + L2 norm over S.
    float sq = 0.0f;
    for (int s = 0; s < S; ++s) {
        const float x = qkvMixed[qSrc + s];
        sq += x * x;
    }
    const float qScale = 1.0f / fmaxf(sqrtf(sq), eps);
    for (int s = 0; s < S; ++s) {
        qOut[dstRow + s] = qkvMixed[qSrc + s] * qScale;
    }
    // k: gather + L2 norm over S.
    float sk = 0.0f;
    for (int s = 0; s < S; ++s) {
        const float x = qkvMixed[kSrc + s];
        sk += x * x;
    }
    const float kScale = 1.0f / fmaxf(sqrtf(sk), eps);
    for (int s = 0; s < S; ++s) {
        kOut[dstRow + s] = qkvMixed[kSrc + s] * kScale;
    }
    // v: gather only (no norm).
    for (int s = 0; s < S; ++s) {
        vOut[dstRow + s] = qkvMixed[vSrc + s];
    }
}
