// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

// Device-side MoE top-K expert routing (M-Q3N.5) — Level Zero / OpenCL C
// port of kernels_cuda/moe_topk.cu. Same algorithm; only the language
// bindings differ. This is the primary path for the 100 tok/s NUC target
// (Meteor Lake, Xe-LPG).
//
// Replaces the host-side compute::moeTopKRoute + the host->USM copy loop
// (Qwen35MoeBackend.cpp:516,579) that forces a D2H/host/H2D round trip per
// MoE layer — the ~96 ms/tok host_sync wall that keeps decode launch-bound.
// Keeping top-K on the device makes the whole MoE block an uninterrupted
// device stream, the precondition for Command-List-Replay (CLR) capture on
// Xe-LPG.
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
// v1 is correctness-first: one work-group per token, a single work-item does
// the selection (nExperts <= 256, K <= 16). Sub-group parallel reduction is a
// perf follow-up and does not change results.
//
// Launch: global = T * MOE_TOPK_LOCAL, local = MOE_TOPK_LOCAL
//         => group count = T, one group per token, only local id 0 computes.

#ifndef MOE_TOPK_LOCAL
#define MOE_TOPK_LOCAL 32
#endif
#ifndef MOE_TOPK_MAX_EXPERTS
#define MOE_TOPK_MAX_EXPERTS 256
#endif
#ifndef MOE_TOPK_MAX_K
#define MOE_TOPK_MAX_K 16
#endif

__attribute__((reqd_work_group_size(MOE_TOPK_LOCAL, 1, 1)))
__kernel void moe_topk(
    __global const float* logits,      // [T, nExperts]  router scores (F32)
    __global       int*   outIdx,      // [T, K]         expert indices (desc)
    __global       float* outWeight,   // [T, K]         renorm weights * wScale
    const int             nExperts,
    const int             K,
    const float           wScale)
{
    if (get_local_id(0) != 0) {
        return;
    }
    const int t = (int)get_group_id(0);

    // Clamp defensively: the host wrapper validates nExperts/K against these
    // ceilings before dispatch; a kernel cannot throw, so out-of-range would
    // otherwise overrun the private arrays below.
    const int nE = (nExperts < MOE_TOPK_MAX_EXPERTS) ? nExperts : MOE_TOPK_MAX_EXPERTS;
    const int kk = (K < MOE_TOPK_MAX_K) ? K : MOE_TOPK_MAX_K;

    __global const float* row    = logits    + (size_t)t * nExperts;
    __global       int*   idxOut = outIdx    + (size_t)t * K;
    __global       float* wOut   = outWeight + (size_t)t * K;

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
        ex[e] = exp(row[e] - maxL);
    }

    // 3. K-pass argmax, descending. Tie-break: lowest expert index wins
    //    (iterate e ascending with strict `>`). `taken` masks selected ones
    //    (char, since OpenCL C has no bool scalar).
    char taken[MOE_TOPK_MAX_EXPERTS];
    for (int e = 0; e < nE; ++e) {
        taken[e] = 0;
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
        taken[best] = 1;
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