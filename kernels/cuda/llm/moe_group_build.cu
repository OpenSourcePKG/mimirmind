// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Device-side MoE token-grouping build (M-Cuda.MoeGroup, Sub-Step A).
//
// Turns the flat per-assignment expert routing produced by `moe_topk`
// (expIdx[R], kw[R] with R = T*K assignments, one per (token, kept-expert))
// into the offset table + permutation a true grouped-by-expert GEMM needs:
// every expert's assignments become a contiguous row range, so each expert
// weight is read ONCE and GEMM'd over all its tokens (M = count[e]) instead
// of the fused-K path's re-read per (token, its-K-experts).
//
// This is the classic sorted-MoE / counting-sort layout (vLLM / Megablocks):
//   histogram -> exclusive prefix sum -> stable scatter into row ranges.
//
// Outputs (all device-resident, no host round trip):
//   expOffset[e]   = first compacted row of expert e   (e in [0, nExperts])
//                    expOffset[nExperts] == R           (exclusive-scan tail)
//   rowSrcTok[r]   = source token t of compacted row r  (= assignment/K)
//                    -> gather x[rowSrcTok[r]] into the compact activation buf
//   rowKw[r]       = router weight of compacted row r    (post-wScale, from kw)
//                    -> applied on the scatter back into the accumulator
//   asnToRow[i]    = compacted row of assignment i (= t*K+k); the INVERSE
//                    permutation. -1 for a dropped (malformed) assignment.
//                    -> lets the scatter run token-major (for k: asnToRow[t*K+k])
//                       so the K contributions of a token accumulate in a
//                       fixed order — deterministic, no atomics there.
// Expert e owns rows [expOffset[e], expOffset[e+1]); the weight to use for
// that range is expert e itself (groups are dense and index-ordered).
//
// v3 (5.21.11): block-parallel histogram + scatter via shared-memory atomics
// — the within-expert row order is the atomic claim order, NOT the stable
// ascending-index order, which every consumer is agnostic to (rows run the
// grouped GEMM independently; the accumulator scatter is asnToRow-driven in
// fixed k order) — the model output stays bit-identical. expOffset is exactly
// deterministic. See the in-body v3 note for the history.
//
// Launch: grid (1,1,1), block (256,1,1). Shared: (MAX_EXPERTS+1)+MAX_EXPERTS
// ints. Warp-size agnostic (no shuffles / no cross-lane work).

#include <cuda_runtime.h>

#ifndef MOE_GROUP_MAX_EXPERTS
#define MOE_GROUP_MAX_EXPERTS 256
#endif

extern "C" __global__ void moe_group_build(
    const int*   __restrict__ expIdx,     // [R]           expert per assignment
    const float* __restrict__ kw,         // [R]           router weight per assignment
          int*   __restrict__ expOffset,  // [nExperts+1]  out: exclusive prefix sum
          int*   __restrict__ rowSrcTok,  // [R]           out: source token per compacted row
          float* __restrict__ rowKw,      // [R]           out: router weight per compacted row
          int*   __restrict__ asnToRow,   // [R]           out: inverse perm (assignment -> row)
    const int                 R,          //               total assignments = T*K
    const int                 nExperts,
    const int                 K)
{
    // Defensive clamp: the host wrapper validates nExperts against this
    // ceiling before dispatch; a kernel cannot throw, so an out-of-range
    // nExperts would otherwise overrun the shared scratch below.
    const int nE = (nExperts < MOE_GROUP_MAX_EXPERTS) ? nExperts
                                                      : MOE_GROUP_MAX_EXPERTS;
    const int tid      = static_cast<int>(threadIdx.x);
    const int nthreads = static_cast<int>(blockDim.x);

    // v3 (2026-09-04, 5.21.11): the v2 histogram + stable scatter still ran
    // two serial O(R) global-read loops on thread 0 — at prefill R (chunk
    // tokens * K, up to ~4096 assignments * 48 MoE layers) that single thread
    // made rt.build ~6% of the whole serving prefill (~400 us/layer). v3
    // parallelises both phases across the block with shared-memory atomics.
    // The within-expert row ORDER is no longer the stable ascending-index
    // order — but every consumer is order-agnostic: each compacted row runs
    // the grouped GEMM independently and the accumulator scatter walks a
    // token's K contributions via asnToRow in fixed k order, so the MODEL
    // OUTPUT is bit-identical to the stable build (verified via byte-equal
    // greedy responses). expOffset stays exactly deterministic. The parity
    // test checks the permutation PROPERTIES (bijection, expert-range
    // membership, payload consistency) instead of the stable order.
    __shared__ int cnt[MOE_GROUP_MAX_EXPERTS + 1];
    __shared__ int cursor[MOE_GROUP_MAX_EXPERTS];

    for (int e = tid; e <= nE; e += nthreads) {
        cnt[e] = 0;
    }
    __syncthreads();

    // 1. Histogram (into cnt[e+1]) — parallel, shared-memory atomics.
    for (int i = tid; i < R; i += nthreads) {
        const int e = expIdx[i];
        if (e >= 0 && e < nE) {
            atomicAdd(&cnt[e + 1], 1);      // drop malformed routing defensively
        }
    }
    __syncthreads();

    // 2. Exclusive prefix sum — thread 0 over <= MOE_GROUP_MAX_EXPERTS shared
    //    ints (negligible). cnt[e] ends as expert e's row-range start.
    if (tid == 0) {
        for (int e = 1; e <= nE; ++e) {
            cnt[e] += cnt[e - 1];
        }
    }
    __syncthreads();

    // Publish the offsets, seed the scatter cursors, default the inverse perm —
    // all parallel across the block.
    for (int e = tid; e <= nE; e += nthreads) {
        expOffset[e] = cnt[e];
    }
    for (int e = tid; e < nE; e += nthreads) {
        cursor[e] = cnt[e];
    }
    for (int i = tid; i < R; i += nthreads) {
        asnToRow[i] = -1;                            // default: dropped
    }
    __syncthreads();

    // 3. Scatter — parallel; each assignment claims its row via an atomic
    //    cursor bump (within-expert order = claim order, see the v3 note).
    for (int i = tid; i < R; i += nthreads) {
        const int e = expIdx[i];
        if (e < 0 || e >= nE) {
            continue;
        }
        const int pos  = atomicAdd(&cursor[e], 1);
        rowSrcTok[pos] = (K > 0) ? (i / K) : 0;      // assignment i -> token t
        rowKw[pos]     = kw[i];
        asnToRow[i]    = pos;                         // inverse permutation
    }
}
