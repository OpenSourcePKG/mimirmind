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
//                       fixed order — deterministic, no atomics.
// Expert e owns rows [expOffset[e], expOffset[e+1]); the weight to use for
// that range is expert e itself (groups are dense and index-ordered).
//
// v1 is correctness-first, mirroring the moe_topk v1 rationale: ONE thread
// does the whole build. R = T*K is <= prefillChunk*K (<= 512*8 = 4096) and
// the work is O(R + nExperts) integer ops — negligible next to the expert
// GEMMs that read hundreds of MB of weights. A block-parallel histogram +
// scan is a perf follow-up and does not change the result. The stable scatter
// (assignments visited in ascending index order, `cursor` advanced per
// expert) makes the permutation deterministic, so the CPU counting-sort
// reference matches it exactly — the correctness gate.
//
// Launch: grid (1,1,1), block (1,1,1) (thread 0 only). No shared memory.
// Warp-size agnostic by construction (no shuffles / no cross-lane work).

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

    // v2 (2026-08-11): v1 was single-thread — the O(nExperts) zero + in-place
    // global prefix-sum + a cursor[nExperts] LOCAL array (register spill) cost
    // ~13 us/layer. v2 keeps the histogram / scan / stable-scatter on thread 0
    // (so outputs are bit-identical to the CPU counting-sort golden), but runs
    // them over SHARED counts (no global RMW, no local-mem cursor) and does the
    // O(nExperts) zero + expOffset write across all threads of one block.
    __shared__ int cnt[MOE_GROUP_MAX_EXPERTS + 1];
    __shared__ int cursor[MOE_GROUP_MAX_EXPERTS];

    for (int e = tid; e <= nE; e += nthreads) {
        cnt[e] = 0;
    }
    __syncthreads();

    // 1+2. Histogram (into cnt[e+1]) then exclusive prefix sum — thread 0, in
    //       shared memory. cnt[e] ends as expert e's row-range start; cnt[nE]=R'.
    if (tid == 0) {
        for (int i = 0; i < R; ++i) {
            const int e = expIdx[i];
            if (e >= 0 && e < nE) {
                cnt[e + 1] += 1;            // drop malformed routing defensively
            }
        }
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

    // 3. Stable scatter — thread 0, ascending `i`, so the within-expert order
    //    matches the CPU counting-sort golden bit-for-bit.
    if (tid == 0) {
        for (int i = 0; i < R; ++i) {
            const int e = expIdx[i];
            if (e < 0 || e >= nE) {
                continue;
            }
            const int pos  = cursor[e]++;
            rowSrcTok[pos] = (K > 0) ? (i / K) : 0;  // assignment i -> token t
            rowKw[pos]     = kw[i];
            asnToRow[i]    = pos;                     // inverse permutation
        }
    }
}
