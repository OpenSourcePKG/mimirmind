// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Device-side MoE tile-schedule build (M-Cuda.MoeGroup, Sub-Step E-a).
//
// Turns the per-expert row ranges produced by `moe_group_build`
// (expOffset[nExperts+1], an exclusive prefix sum) into a COMPACT per-tile
// schedule that a single device-driven grouped GEMM consumes in one launch —
// removing the host round trip (the `expOffset` D2H) and the per-expert host
// launch loop that made the host-driven grouped path (Option 1) lose to the
// fused-K batched path on GB10.
//
// Expert e owns compacted rows [expOffset[e], expOffset[e+1]); its count is
// split into ceil(count / tileM) tiles of <= tileM rows (tileM matches the
// grouped GEMM's per-block M cap, GEMM_MAX_M = 16). Each emitted tile is one
// grouped-GEMM work item: an expert id + an absolute row range into the
// compacted activation / output buffers.
//
// Outputs (all device-resident, no host round trip):
//   tileExpert[maxTiles]  = expert id of tile t, or -1 sentinel for the unused
//                           tail (the host over-provisions the grid to the
//                           static upper bound maxTiles, so unused blocks
//                           early-exit on the sentinel — no D2H of the real
//                           tile count needed to size the launch).
//   tileRow0[maxTiles]    = absolute start row (into xCompact / the grouped
//                           output) of tile t.
//   tileRows[maxTiles]    = row count of tile t (1..tileM).
//   nTiles[1]             = real tile count (device-side; for asserts / a
//                           future exact-grid launch, NOT needed for the
//                           over-provisioned launch above).
//
// Static upper bound (host-computable from R and nExperts, NO D2H):
//   maxTiles = ceil(R / tileM) + nExperts
// because sum_e ceil(count_e / tileM) <= sum_e (count_e/tileM + 1)
//        = R/tileM + nExperts. The host sizes both the scratch and grid.x
// from this, so the grid dimension never depends on a device value.
//
// v2 (2026-08-11): the whole schedule (maxTiles = ceil(R/tileM)+nExperts, ~258
// at decode) was written by ONE thread — the sentinel-fill of the ~250-entry
// tail alone is ~750 sequential global writes/layer, ~23 us/layer = the biggest
// remaining "route" cost after the warp-topk fix. v2 does the sentinel-fill
// across all threads of one block, then thread 0 runs the SAME sequential
// expert walk to overwrite the real tiles [0,t). The walk stays single-thread
// so the schedule is bit-identical to the CPU golden (the per-expert tile split
// order is unchanged); only the bulk tail-fill is parallelised.
//
// Launch: grid (1,1,1), block (256,1,1). No shared memory. Warp-size agnostic.

#include <cuda_runtime.h>

#ifndef MOE_TILES_MAX_EXPERTS
#define MOE_TILES_MAX_EXPERTS 256
#endif

extern "C" __global__ void moe_group_tiles(
    const int* __restrict__ expOffset,  // [nExperts+1]  exclusive prefix sum
          int* __restrict__ tileExpert,  // [maxTiles]   out: expert id / -1
          int* __restrict__ tileRow0,    // [maxTiles]   out: start row
          int* __restrict__ tileRows,    // [maxTiles]   out: row count (1..tileM)
          int* __restrict__ nTiles,      // [1]          out: real tile count
    const int                nExperts,
    const int                maxTiles,
    const int                tileM)
{
    // sc[e] = tile count of expert e, then (after the scan) its start tile.
    __shared__ int sc[MOE_TILES_MAX_EXPERTS];
    const int tid      = static_cast<int>(threadIdx.x);
    const int nthreads = static_cast<int>(blockDim.x);
    const int nE = (nExperts < MOE_TILES_MAX_EXPERTS) ? nExperts
                                                      : MOE_TILES_MAX_EXPERTS;
    const int tm = (tileM > 0) ? tileM : 1;

    // Parallel sentinel-fill of the whole schedule; the emit below overwrites
    // the real prefix. __syncthreads orders fill-before-emit.
    for (int i = tid; i < maxTiles; i += nthreads) {
        tileExpert[i] = -1;
        tileRow0[i]   = 0;
        tileRows[i]   = 0;
    }

    // Tiles per expert = ceil(count / tileM), 0 for an empty expert.
    for (int e = tid; e < nE; e += nthreads) {
        const int off = expOffset[e];
        const int end = expOffset[e + 1];
        sc[e] = (end > off) ? ((end - off + tm - 1) / tm) : 0;
    }
    __syncthreads();

    // Exclusive prefix sum (thread 0, shared, deterministic): sc[e] = start tile
    // of expert e. `acc` ends as the real tile count. Same expert-ascending tile
    // layout the single-thread v1 walk produced -> bit-identical schedule.
    if (tid == 0) {
        int acc = 0;
        for (int e = 0; e < nE; ++e) {
            const int c = sc[e];
            sc[e] = acc;
            acc  += c;
        }
        nTiles[0] = (acc <= maxTiles) ? acc : maxTiles;
    }
    __syncthreads();

    // Parallel emit: expert e writes its tiles at [sc[e], sc[e] + count).
    for (int e = tid; e < nE; e += nthreads) {
        const int off   = expOffset[e];
        const int end   = expOffset[e + 1];
        int       t     = sc[e];
        for (int row = off; row < end; row += tm, ++t) {
            if (t >= maxTiles) {
                break;                          // defensive; host over-provisions
            }
            int rows = end - row;
            if (rows > tm) {
                rows = tm;
            }
            tileExpert[t] = e;
            tileRow0[t]   = row;
            tileRows[t]   = rows;
        }
    }
}
