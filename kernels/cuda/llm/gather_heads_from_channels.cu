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

// 5.21.12 — warp-coalesced post-conv prep. The scalar fused_post_conv_prep
// above runs ONE thread per (t, head) row with a serial S-loop: the 32 lanes
// of a warp read 32 different head blocks (stride S), so every load is fully
// uncoalesced — the 2026-09-05 sub-split profile put this "gdn.split" at 5.2%
// of the whole serving prefill (bigger than the conv itself). This variant
// runs ONE BLOCK per (t, head) row with `blockDim.x == S` threads: lane s
// touches element s, so q/k/v loads and the q/k writes are perfectly
// coalesced, and the two L2 sum-of-squares reductions run in parallel via a
// shared-memory tree instead of a 128-step serial loop. Bit-identical result
// (same fp32 sum order is NOT guaranteed — tree vs serial reassociates the
// reduction, so it is tolerance-equal; the norm is a scale, downstream is
// GEMM). Requires S == blockDim.x (host launches block = S; guarded for the
// prod S=128 shape, arbitrary S falls back to the scalar kernel host-side).
extern "C" __global__ __launch_bounds__(1024)
void fused_post_conv_prep_warp(
    const float* __restrict__ qkvMixed,
    float*       __restrict__ qOut,
    float*       __restrict__ kOut,
    float*       __restrict__ vOut,
    const int                 T,
    const int                 srcHeadsKV,
    const int                 dstHeads,
    const int                 S,
    const int                 convTotalWidth,
    const int                 keyDim,
    const float               eps)
{
    const int row = blockIdx.x;              // one (t, head) row per block
    if (row >= T * dstHeads) {
        return;
    }
    const int s = threadIdx.x;               // one element per thread (s < S)
    const int h = row % dstHeads;
    const int t = row / dstHeads;
    const int srcHeadKV = h % srcHeadsKV;

    const size_t base   = (size_t)t * convTotalWidth;
    const size_t qSrc   = base + (size_t)srcHeadKV * S;
    const size_t kSrc   = base + (size_t)keyDim + (size_t)srcHeadKV * S;
    const size_t vSrc   = base + (size_t)2 * keyDim + (size_t)h * S;
    const size_t dstRow = (size_t)row * S;

    const float xq = qkvMixed[qSrc + s];     // coalesced across the block
    const float xk = qkvMixed[kSrc + s];
    const float xv = qkvMixed[vSrc + s];

    extern __shared__ float red[];           // [2 * S]: q sq | k sq
    red[s]     = xq * xq;
    red[S + s] = xk * xk;
    __syncthreads();
    for (int off = S >> 1; off > 0; off >>= 1) {
        if (s < off) {
            red[s]     += red[s + off];
            red[S + s] += red[S + s + off];
        }
        __syncthreads();
    }
    const float qScale = 1.0f / fmaxf(sqrtf(red[0]), eps);
    const float kScale = 1.0f / fmaxf(sqrtf(red[S]), eps);

    qOut[dstRow + s] = xq * qScale;          // coalesced writes
    kOut[dstRow + s] = xk * kScale;
    vOut[dstRow + s] = xv;
}
