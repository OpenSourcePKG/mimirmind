// M8.K.1 experimental Q8_0 GEMM — v1 with a bigger M-tile and a
// sub-group broadcast for the per-block scale.
//
// Live-Autotune data (2026-07-02): v1 GEMM lost 2.2× to matvec-loop at
// every M-bucket (16 / 64 / 256). Per-op diagnosis showed matmul is
// 48 % of decode-time so a GEMM that actually wins would unlock real
// prefill headroom AND make M9.11 speculative decoding viable. Root
// cause hypothesis: v1's per-workgroup output volume amortises W
// dequant work over only M_TILE=8 rows, and every lane redundantly
// loads the same fp16 block scale from global memory.
//
// Changes vs matmul_q8_0_gemm.cl:
//   1. M_TILE = 16 (was 8). Doubles W-reuse per workgroup at the cost
//      of doubling the xTile SLM (32 KiB → 64 KiB per WG) and doubling
//      the per-lane accumulator array (sum[8] → sum[16] float).
//      Xe-LPG's per-WG SLM budget is 64 KiB so this sits right at the
//      edge; if the driver refuses the layout we roll back to M_TILE=8
//      on this kernel and pick a different tuning axis.
//   2. Sub-group broadcast for `d`. v1 has every lane in a 16-lane SG
//      re-read the same 2-byte fp16 scale from global; v2 has lane 0
//      read it and `sub_group_broadcast()` distribute. Removes 15/16
//      of the block-scale global reads.
//
// Everything else (WG=64, SG=16, 4 outputs per WG, X_TILE=1024) is
// deliberately unchanged so any perf delta is attributable to the two
// axes above.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

#ifndef MATMUL_Q8_0_V2_LOCAL
#define MATMUL_Q8_0_V2_LOCAL 64
#endif

#ifndef MATMUL_Q8_0_V2_SG
#define MATMUL_Q8_0_V2_SG 16
#endif

#ifndef MATMUL_Q8_0_GEMM_V2_M_TILE
#define MATMUL_Q8_0_GEMM_V2_M_TILE 16
#endif

#define MATMUL_Q8_0_V2_OUTPUTS_PER_GROUP \
    (MATMUL_Q8_0_V2_LOCAL / MATMUL_Q8_0_V2_SG)

#define Q8_0_BLOCK_ELEMENTS 32
#define Q8_0_BLOCK_BYTES    34

// 1024 elements = 32 blocks per K-tile.
// SLM: X_TILE_ELEMENTS * M_TILE * 4 B = 1024 * 16 * 4 = 64 KiB / WG.
#define X_TILE_ELEMENTS 1024

__attribute__((reqd_work_group_size(MATMUL_Q8_0_V2_LOCAL, 1, 1)))
__attribute__((intel_reqd_sub_group_size(MATMUL_Q8_0_V2_SG)))
__kernel void matmul_q8_0_gemm_v2(
    __global const float* X,
    __global const uchar* W,
    __global       float* Y,
    const int             K,
    const int             N,
    const int             M)
{
    __local float xTile[X_TILE_ELEMENTS][MATMUL_Q8_0_GEMM_V2_M_TILE];

    const int  wgN     = (int)get_group_id(0);
    const int  wgM     = (int)get_group_id(1);
    const int  sgInWg  = (int)get_sub_group_id();           // 0..3
    const int  sgLocal = (int)get_sub_group_local_id();     // 0..15
    const int  tid     = (int)get_local_id(0);
    const int  lsize   = (int)get_local_size(0);
    const int  n       = wgN * MATMUL_Q8_0_V2_OUTPUTS_PER_GROUP + sgInWg;
    const int  mBase   = wgM * MATMUL_Q8_0_GEMM_V2_M_TILE;
    const bool nActive = (n < N);
    const int  nBlocks = K / Q8_0_BLOCK_ELEMENTS;

    float sum[MATMUL_Q8_0_GEMM_V2_M_TILE];

    #pragma unroll
    for (int mm = 0; mm < MATMUL_Q8_0_GEMM_V2_M_TILE; ++mm) {
        sum[mm] = 0.0f;
    }

    for (int tile = 0; tile < K; tile += X_TILE_ELEMENTS) {
        const int tileK = min(X_TILE_ELEMENTS, K - tile);

        // Cooperative X-tile load — one thread per (m-slot, k-slot) pair
        // over the workgroup. Matches v1's stride pattern so any perf
        // change here is purely from the doubled tile size.
        const int loadTotal = MATMUL_Q8_0_GEMM_V2_M_TILE * X_TILE_ELEMENTS;
        for (int idx = tid; idx < loadTotal; idx += lsize) {
            const int  mSlot = idx / X_TILE_ELEMENTS;
            const int  iSlot = idx - mSlot * X_TILE_ELEMENTS;
            const int  mAct  = mBase + mSlot;
            const bool valid = (mAct < M) && (iSlot < tileK);
            xTile[iSlot][mSlot] =
                valid ? X[(size_t)mAct * (size_t)K + tile + iSlot] : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (nActive) {
            __global const uchar* row =
                W + (size_t)n * (size_t)nBlocks * Q8_0_BLOCK_BYTES;

            const int blockStart   = tile / Q8_0_BLOCK_ELEMENTS;
            const int blocksInTile = X_TILE_ELEMENTS / Q8_0_BLOCK_ELEMENTS;
            const int blockEnd     = min(blockStart + blocksInTile, nBlocks);

            for (int b = blockStart; b < blockEnd; ++b) {
                __global const uchar* block = row + b * Q8_0_BLOCK_BYTES;

                // Sub-group broadcast for the fp16 scale. Only lane 0
                // reads global; the other 15 lanes get the value via
                // register-file broadcast. Removes 15/16 of the block-
                // scale global reads.
                float d;
                if (sgLocal == 0) {
                    d = vload_half(0, (__global const half*)(block));
                }
                d = sub_group_broadcast(d, 0);

                __global const char* qs =
                    (__global const char*)(block + 2);

                const int xLocalBase = (b - blockStart) * Q8_0_BLOCK_ELEMENTS;

                // Same 2-quants-per-lane stride as v1. Each mad fans
                // out to M_TILE=16 accumulators (double v1's 8-fanout).
                for (int l = sgLocal; l < Q8_0_BLOCK_ELEMENTS;
                     l += MATMUL_Q8_0_V2_SG) {
                    const float w = d * (float)qs[l];

                    #pragma unroll
                    for (int mm = 0; mm < MATMUL_Q8_0_GEMM_V2_M_TILE; ++mm) {
                        sum[mm] = mad(xTile[xLocalBase + l][mm], w, sum[mm]);
                    }
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    #pragma unroll
    for (int mm = 0; mm < MATMUL_Q8_0_GEMM_V2_M_TILE; ++mm) {
        float s = sub_group_reduce_add(sum[mm]);
        if (nActive && sgLocal == 0) {
            const int mAct = mBase + mm;
            if (mAct < M) {
                Y[(size_t)mAct * (size_t)N + n] = s;
            }
        }
    }
}