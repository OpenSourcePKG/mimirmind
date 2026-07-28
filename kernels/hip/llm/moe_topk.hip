// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Device-side MoE top-K expert routing (M-Q3N.5) — HIP port of
// kernels_cuda/moe_topk.cu. Bit-for-bit the same algorithm; only the runtime
// header differs. Warp-size agnostic by construction (no shuffles, single
// thread per block does the selection) so gfx1101 WAVE32 needs no special
// casing — identical results to the CUDA (warp32) and L0 (SIMD16) builds.
//
// Replaces the host-side compute::moeTopKRoute + the host->USM copy loop
// (Qwen35MoeBackend.cpp:516,579) that forces a D2H/host/H2D round trip per
// MoE layer. Moving top-K onto the device makes the whole MoE block an
// uninterrupted device stream — the precondition for HipGraph capture
// (see M-HipGraph shelf, unshelved by this milestone).
//
// Math per token t (mirrors compute::moeTopKRoute exactly):
//   p     = softmax(logits[t])                 (max-subtract for stability)
//   keep  = indices of the top-K largest p     (descending; ties: low idx)
//   w[k]  = p[keep[k]] / sum_j p[keep[j]]  * wScale
//
// The global softmax denominator cancels in the top-K renormalisation, so
// this kernel works directly on exp(logit - max) and never forms the full
// sum — algebraically identical to the CPU reference, up to float rounding.
//
// outIdx / outWeight are written in the exact [T, K] layout the fused-K
// consumers read from USM (expIdxSlot / kwSlot).
//
// v1 is correctness-first: one block per token, a single thread does the
// selection (nExperts <= 256, K <= 16). Block-parallel reduction is a perf
// follow-up and does not change results.
//
// Launch:
//   dim3 grid ( T, 1, 1 )
//   dim3 block( 32, 1, 1 )   // warp-aligned; only thread 0 computes in v1

#include <hip/hip_runtime.h>

#ifndef MOE_TOPK_MAX_EXPERTS
#define MOE_TOPK_MAX_EXPERTS 256
#endif
#ifndef MOE_TOPK_MAX_K
#define MOE_TOPK_MAX_K 16
#endif

extern "C" __global__
void moe_topk(
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

    const float* row    = logits    + static_cast<size_t>(t) * nExperts;
    int*         idxOut = outIdx    + static_cast<size_t>(t) * K;
    float*       wOut   = outWeight + static_cast<size_t>(t) * K;

    // 1. max for numerical stability.
    float maxL = row[0];
    for (int e = 1; e < nE; ++e) {
        if (row[e] > maxL) {
            maxL = row[e];
        }
    }

    // 2. exp(logit - max). Unnormalised probabilities; the softmax
    //    denominator cancels in step 4's renormalisation.
    float ex[MOE_TOPK_MAX_EXPERTS];
    for (int e = 0; e < nE; ++e) {
        ex[e] = __expf(row[e] - maxL);
    }

    // 3. K-pass argmax, descending. Tie-break: lowest expert index wins
    //    (iterate e ascending with strict `>`). `taken` masks selected ones.
    bool taken[MOE_TOPK_MAX_EXPERTS];
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