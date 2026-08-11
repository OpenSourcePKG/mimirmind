// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Device-side MoE top-K expert routing (M-Q3N.5).
//
// Replaces the host-side `compute::moeTopKRoute` + the host->USM copy loop
// that today forces a D2H/host/H2D round trip on every MoE layer. Moving
// top-K onto the device makes the whole MoE block (router matmul -> top-K ->
// gate/up fused-K -> down fused-K) an uninterrupted device stream.
//
// Math per token t (mirrors compute::moeTopKRoute exactly):
//   p     = softmax(logits[t])                 (max-subtract for stability)
//   keep  = indices of the top-K largest p     (descending; ties: low idx)
//   w[k]  = p[keep[k]] / sum_j p[keep[j]]  * wScale
//
// Softmax note: the global softmax denominator cancels in the top-K
// renormalisation (w[k] = exp[keep[k]] / sum_j exp[keep[j]]), so this kernel
// never computes the full sum — it works directly on exp(logit - max). This
// is algebraically identical to the CPU reference, up to float rounding.
//
// Layout: outIdx / outWeight are written in the exact [T, K] contiguous
// layout the fused-K consumers read from USM (expIdxSlot / kwSlot).
//
// v2 (2026-08-11) — WARP-PARALLEL. The v1 kernel let only thread 0 compute and
// declared two nExperts-sized stack arrays (ex[256], taken[256]) that spilled
// to local memory; at decode T=1 that single-thread local-memory selection was
// ~70 us/layer = ~10% of the whole decode step (the dominant "route" cost).
// v2 spreads the experts across the 32 lanes of one warp (PER experts/lane held
// in registers) and does the K-pass argmax with a warp butterfly reduction. It
// is bit-identical to v1 / the CPU reference: same global max, same
// exp(logit-max), same "highest exp, lowest index on ties" selection order,
// same keptSum accumulation order (k = 0..K), same renormalisation.

#include <cuda_runtime.h>

#ifndef MOE_TOPK_MAX_EXPERTS
#define MOE_TOPK_MAX_EXPERTS 256
#endif
#ifndef MOE_TOPK_MAX_K
#define MOE_TOPK_MAX_K 16
#endif
// Experts handled per lane: lane l owns experts l, l+32, l+64, ...
#define MOE_TOPK_PER ((MOE_TOPK_MAX_EXPERTS + 31) / 32)

// One block (one warp) per token. Grid.x = T, block.x = 32.
extern "C" __global__ void moe_topk(
    const float* __restrict__ logits,      // [T, nExperts]  router scores (F32)
          int*   __restrict__ outIdx,      // [T, K]         expert indices (desc)
          float* __restrict__ outWeight,   // [T, K]         renorm weights * wScale
    const int                 nExperts,
    const int                 K,
    const float               wScale)
{
    const unsigned FULL = 0xffffffffu;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int t    = static_cast<int>(blockIdx.x);
    const int kk   = (K < MOE_TOPK_MAX_K) ? K : MOE_TOPK_MAX_K;

    const float* __restrict__ row    = logits    + static_cast<size_t>(t) * nExperts;
    int*         __restrict__ idxOut = outIdx    + static_cast<size_t>(t) * K;
    float*       __restrict__ wOut   = outWeight + static_cast<size_t>(t) * K;

    const float NEG_BIG = -3.402823466e+38f;   // -FLT_MAX (invalid experts sink)

    // Each lane loads its experts' logits into registers (no local memory).
    float lg[MOE_TOPK_PER];
    int   eid[MOE_TOPK_PER];
#pragma unroll
    for (int j = 0; j < MOE_TOPK_PER; ++j) {
        const int e = lane + j * 32;
        eid[j] = e;
        lg[j]  = (e < nExperts) ? row[e] : NEG_BIG;
    }

    // 1. global max (warp butterfly) — matches the CPU ascending-scan max.
    float m = NEG_BIG;
#pragma unroll
    for (int j = 0; j < MOE_TOPK_PER; ++j) {
        m = fmaxf(m, lg[j]);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(FULL, m, o));
    }

    // 2. exp(logit - max); invalid experts -> 0 so they never win a pass.
    float ex[MOE_TOPK_PER];
#pragma unroll
    for (int j = 0; j < MOE_TOPK_PER; ++j) {
        ex[j] = (eid[j] < nExperts) ? __expf(lg[j] - m) : 0.0f;
    }
    bool taken[MOE_TOPK_PER];
#pragma unroll
    for (int j = 0; j < MOE_TOPK_PER; ++j) {
        taken[j] = false;
    }

    // 3. K-pass argmax, descending. Tie-break: lowest expert index wins (the
    //    CPU reference's fixed rule). Within a lane the ascending-j scan with a
    //    strict `>` keeps the lowest local index; the warp reduction breaks
    //    cross-lane ties to the lower index the same way.
    float keptSum = 0.0f;
    float selEx[MOE_TOPK_MAX_K];
    for (int k = 0; k < kk; ++k) {
        float lb  = -1.0f;              // exp(...) is always >= 0
        int   lbi = 0x7fffffff;
#pragma unroll
        for (int j = 0; j < MOE_TOPK_PER; ++j) {
            if (!taken[j] && eid[j] < nExperts && ex[j] > lb) {
                lb  = ex[j];
                lbi = eid[j];
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(FULL, lb,  o);
            const int   oi = __shfl_xor_sync(FULL, lbi, o);
            if (ov > lb || (ov == lb && oi < lbi)) {
                lb  = ov;
                lbi = oi;
            }
        }
        // All lanes now agree on the winner (lb, lbi); mark it taken on its owner.
#pragma unroll
        for (int j = 0; j < MOE_TOPK_PER; ++j) {
            if (eid[j] == lbi) {
                taken[j] = true;
            }
        }
        keptSum += lb;
        if (lane == 0) {
            idxOut[k] = lbi;
            selEx[k]  = lb;
        }
    }

    // 4. renormalise the kept weights to sum to 1, then apply wScale.
    if (lane == 0) {
        const float invKept = (keptSum > 0.0f) ? (1.0f / keptSum) : 1.0f;
        for (int k = 0; k < kk; ++k) {
            wOut[k] = selEx[k] * invKept * wScale;
        }
    }
}
