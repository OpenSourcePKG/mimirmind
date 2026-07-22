// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Device-side MoE top-K expert routing (M-Q3N.5).
//
// Replaces the host-side `compute::moeTopKRoute` + the host->USM copy loop
// that today forces a D2H/host/H2D round trip on every MoE layer
// (Qwen35MoeBackend.cpp:516,579). That per-layer host sync is the ~96 ms/tok
// wall that keeps decode launch-bound instead of bandwidth-bound; moving
// top-K onto the device makes the whole MoE block (router matmul -> top-K ->
// gate/up fused-K -> down fused-K) an uninterrupted device stream, which is
// the precondition for CUDA-graph / CLR / HipGraph capture.
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
// v1 is correctness-first: one block per token, a single thread does the
// selection (nExperts <= 256, K <= 16 => ~K*nExperts comparisons, negligible
// vs the router matmul at decode T=1). Block-parallel reduction is a perf
// follow-up and does not change results. Warp-size agnostic by construction
// (no shuffles) -> identical on Xe-LPG SIMD16 / gfx1101 WAVE32 / GB10 warp32.

#include <cuda_runtime.h>

#ifndef MOE_TOPK_MAX_EXPERTS
#define MOE_TOPK_MAX_EXPERTS 256
#endif
#ifndef MOE_TOPK_MAX_K
#define MOE_TOPK_MAX_K 16
#endif

// One block per token. Grid.x = T. Block.x is arbitrary (>=1); only thread 0
// computes in v1. Launch with 32 for warp-aligned occupancy accounting.
extern "C" __global__ void moe_topk(
    const float* __restrict__ logits,      // [T, nExperts]  router scores (F32)
          int*   __restrict__ outIdx,      // [T, K]         expert indices (desc)
          float* __restrict__ outWeight,   // [T, K]         renorm weights * wScale
    const int                 nExperts,
    const int                 K,
    const float               wScale)
{
    if (threadIdx.x != 0) {
        return;
    }
    const int t = static_cast<int>(blockIdx.x);

    // Clamp defensively: the host wrapper validates nExperts/K against these
    // ceilings before dispatch; a kernel cannot throw, so out-of-range would
    // otherwise corrupt the stack arrays below.
    const int nE = (nExperts < MOE_TOPK_MAX_EXPERTS) ? nExperts : MOE_TOPK_MAX_EXPERTS;
    const int kk = (K < MOE_TOPK_MAX_K) ? K : MOE_TOPK_MAX_K;

    const float* row     = logits    + static_cast<size_t>(t) * nExperts;
    int*         idxOut  = outIdx    + static_cast<size_t>(t) * K;
    float*       wOut    = outWeight + static_cast<size_t>(t) * K;

    // 1. max for numerical stability.
    float maxL = row[0];
    for (int e = 1; e < nE; ++e) {
        if (row[e] > maxL) {
            maxL = row[e];
        }
    }

    // 2. exp(logit - max). exp values are the unnormalised probabilities;
    //    the softmax denominator cancels in step 4's renormalisation.
    float ex[MOE_TOPK_MAX_EXPERTS];
    for (int e = 0; e < nE; ++e) {
        ex[e] = __expf(row[e] - maxL);
    }

    // 3. K-pass argmax, descending. Tie-break: lowest expert index wins.
    //    Iterating e ascending with a strict `>` keeps the first (lowest
    //    index) maximum on equal values — the fixed rule the CPU reference
    //    aligns to. `taken` masks already-selected experts.
    bool  taken[MOE_TOPK_MAX_EXPERTS];
    for (int e = 0; e < nE; ++e) {
        taken[e] = false;
    }

    float keptSum = 0.0f;
    float selEx[MOE_TOPK_MAX_K];
    for (int k = 0; k < kk; ++k) {
        int   best    = -1;
        float bestVal = -1.0f;              // exp(...) is always >= 0
        for (int e = 0; e < nE; ++e) {
            if (taken[e]) {
                continue;
            }
            if (best < 0 || ex[e] > bestVal) {
                bestVal = ex[e];
                best    = e;
            }
        }
        taken[best] = true;
        idxOut[k]   = best;
        selEx[k]    = bestVal;
        keptSum    += bestVal;
    }

    // 4. renormalise the kept weights to sum to 1, then apply wScale.
    const float invKept = (keptSum > 0.0f) ? (1.0f / keptSum) : 1.0f;
    for (int k = 0; k < kk; ++k) {
        wOut[k] = selEx[k] * invKept * wScale;
    }
}