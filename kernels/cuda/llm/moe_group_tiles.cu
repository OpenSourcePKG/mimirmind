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
// v1 is correctness-first, mirroring the moe_group_build v1 rationale: ONE
// thread walks the experts and emits tiles sequentially. nExperts <= 256 and
// maxTiles <= ~R/16 + nExperts (a few hundred) — O(nExperts + tiles) integer
// ops, negligible next to the expert GEMMs. The sequential walk also makes
// the schedule deterministic so the CPU golden matches it exactly.
//
// Launch: grid (1,1,1), block (1,1,1) (thread 0 only). No shared memory.
// Warp-size agnostic by construction (no shuffles / no cross-lane work).

#include <cuda_runtime.h>

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
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }

    const int tm = (tileM > 0) ? tileM : 1;

    int t = 0;
    for (int e = 0; e < nExperts; ++e) {
        const int off = expOffset[e];
        const int end = expOffset[e + 1];
        int row = off;
        while (row < end) {
            if (t >= maxTiles) {
                // Defensive: the host sizes the grid/scratch to the static
                // upper bound so this never trips; a kernel cannot throw, so
                // clamp rather than overrun the output buffers.
                nTiles[0] = t;
                return;
            }
            int rows = end - row;
            if (rows > tm) {
                rows = tm;
            }
            tileExpert[t] = e;
            tileRow0[t]   = row;
            tileRows[t]   = rows;
            row += rows;
            ++t;
        }
    }

    // Sentinel-fill the unused tail so over-provisioned grid blocks early-exit.
    for (int i = t; i < maxTiles; ++i) {
        tileExpert[i] = -1;
        tileRow0[i]   = 0;
        tileRows[i]   = 0;
    }
    nTiles[0] = t;
}
