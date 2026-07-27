// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

// Device-side MoE top-K expert routing (M-Q3N.5) — Level Zero / OpenCL C
// port of kernels_cuda/moe_topk.cu. Same algorithm; only the language
// bindings differ. This is the primary path for the 100 tok/s NUC target
// (Meteor Lake, Xe-LPG).
//
// Replaces the host-side compute::moeTopKRoute + the host->USM copy loop
// that forces a D2H/host/H2D round trip per MoE layer — the per-layer host
// round trip that keeps decode launch-bound and the precondition for
// Command-List-Replay (CLR) capture on Xe-LPG.
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
// v2 (perf): the whole work-group (MOE_TOPK_LOCAL lanes) cooperates. The
// v1 kernel let a single work-item do the max + exp + K-pass argmax
// serially (~nExperts*(K+2) ops); v2 splits every pass across the lanes
// with an SLM tree-reduce. Results are BIT-IDENTICAL to v1: max() and the
// lowest-index-tie-break argmax are exact, and the kept-mass sum stays a
// serial accumulate in selection order (lane 0), so the renormalised
// weights round exactly as before.
//
// Launch: one work-group per token (group count = T), local = MOE_TOPK_LOCAL.

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
    __local float exSh[MOE_TOPK_MAX_EXPERTS];   // exp(logit - max)
    __local char  taken[MOE_TOPK_MAX_EXPERTS];  // selected mask
    __local float redVal[MOE_TOPK_LOCAL];       // per-lane reduction scratch
    __local int   redIdx[MOE_TOPK_LOCAL];
    __local float keptSumSh;

    const int t     = (int)get_group_id(0);
    const int lid   = (int)get_local_id(0);
    const int lsize = (int)get_local_size(0);

    // Clamp defensively (host wrapper validates against these ceilings).
    const int nE = (nExperts < MOE_TOPK_MAX_EXPERTS) ? nExperts : MOE_TOPK_MAX_EXPERTS;
    const int kk = (K < MOE_TOPK_MAX_K) ? K : MOE_TOPK_MAX_K;

    __global const float* row    = logits    + (size_t)t * nExperts;
    __global       int*   idxOut = outIdx    + (size_t)t * K;
    __global       float* wOut   = outWeight + (size_t)t * K;

    // 1. max(row) — parallel, then SLM tree-reduce. max() is exact.
    float lmax = -INFINITY;
    for (int e = lid; e < nE; e += lsize) {
        lmax = fmax(lmax, row[e]);
    }
    redVal[lid] = lmax;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = lsize >> 1; s > 0; s >>= 1) {
        if (lid < s) {
            redVal[lid] = fmax(redVal[lid], redVal[lid + s]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float maxL = redVal[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // 2. exp(logit - max) into SLM + clear the taken mask.
    for (int e = lid; e < nE; e += lsize) {
        exSh[e]  = exp(row[e] - maxL);
        taken[e] = 0;
    }
    if (lid == 0) {
        keptSumSh = 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // 3. K-pass argmax. Each pass: every lane scans its strided slice for
    //    the local (max value, lowest index) among untaken experts, then an
    //    SLM tree-reduce picks the global one (ties -> lower index). Lane 0
    //    records the pick, marks it taken, and accumulates keptSum in
    //    selection order (bit-identical to the v1 serial sum).
    for (int k = 0; k < kk; ++k) {
        float bestVal = -1.0f;   // exp(...) >= 0
        int   bestIdx = -1;
        for (int e = lid; e < nE; e += lsize) {
            if (taken[e]) {
                continue;
            }
            const float v = exSh[e];
            if (bestIdx < 0 || v > bestVal) {   // strict '>' keeps lowest idx
                bestVal = v;
                bestIdx = e;
            }
        }
        redVal[lid] = bestVal;
        redIdx[lid] = bestIdx;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int s = lsize >> 1; s > 0; s >>= 1) {
            if (lid < s) {
                const float v2 = redVal[lid + s];
                const int   i2 = redIdx[lid + s];
                const float v1 = redVal[lid];
                const int   i1 = redIdx[lid];
                // Take the other half when it has a valid, strictly larger
                // value, or an equal value with a lower index (tie-break).
                const bool take = (i1 < 0) ||
                    (i2 >= 0 && (v2 > v1 || (v2 == v1 && i2 < i1)));
                if (take) {
                    redVal[lid] = v2;
                    redIdx[lid] = i2;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (lid == 0) {
            const int best = redIdx[0];
            taken[best] = 1;
            idxOut[k]   = best;
            keptSumSh  += redVal[0];    // == exSh[best], selection order
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // 4. renormalise the kept weights to sum to 1, then apply wScale.
    if (lid == 0) {
        const float invKept = (keptSumSh > 0.0f) ? (1.0f / keptSumSh) : 1.0f;
        for (int k = 0; k < kk; ++k) {
            wOut[k] = exSh[idxOut[k]] * invKept * wScale;
        }
    }
}
