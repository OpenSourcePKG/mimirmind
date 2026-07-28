// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

// M8.K.1 revised — Q8_0 GEMM with reduced SLM footprint.
//
// v2-first-attempt (2026-07-02 L0_TARGET_HOST) doubled M_TILE to 16
// and lost 2.35× to v1 across every M-bucket. Post-hoc root cause:
// 64 KiB SLM per workgroup halved the resident-WG count on Xe-LPG's
// per-Vector-Engine budget, killing occupancy.
//
// This revision goes the OTHER way. Everything is identical to v1
// (matmul_q8_0_gemm.cl) except X_TILE_ELEMENTS shrinks from 1024 to
// 256. That drops SLM from 32 KiB to 8 KiB per WG — 4× more workgroups
// resident on the Xe-LPG scheduler than v1, and 16× more than the
// v2-first-attempt.
//
// Everything else (M_TILE=8, WG=64, SG=16, 4 outputs/WG) is deliberately
// unchanged so any perf delta is attributable to the SLM axis alone.
// If v2-revised wins → matvec's small SLM footprint is confirmed as
// the real Xe-LPG scaling axis, and we chase further along that
// direction. If it loses → GEMM is architecturally the wrong shape on
// Xe-LPG at Governor-throttled clocks and M8.K should be shelved for
// speculative decoding / model swap.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

#ifndef MATMUL_Q8_0_V2_LOCAL
#define MATMUL_Q8_0_V2_LOCAL 64
#endif

#ifndef MATMUL_Q8_0_V2_SG
#define MATMUL_Q8_0_V2_SG 16
#endif

#ifndef MATMUL_Q8_0_GEMM_V2_M_TILE
#define MATMUL_Q8_0_GEMM_V2_M_TILE 8
#endif

#define MATMUL_Q8_0_V2_OUTPUTS_PER_GROUP \
    (MATMUL_Q8_0_V2_LOCAL / MATMUL_Q8_0_V2_SG)

#define Q8_0_BLOCK_ELEMENTS 32
#define Q8_0_BLOCK_BYTES    34

// 256 elements = 8 blocks per K-tile.
// SLM: X_TILE_ELEMENTS * M_TILE * 4 B = 256 * 8 * 4 = 8 KiB / WG.
// v1 had 32 KiB; this v2 revision has 4× smaller footprint so 4× more
// WGs can sit resident on each Xe Vector Engine.
#define X_TILE_ELEMENTS 256

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
                const float d =
                    vload_half(0, (__global const half*)(block));
                __global const char* qs =
                    (__global const char*)(block + 2);

                const int xLocalBase = (b - blockStart) * Q8_0_BLOCK_ELEMENTS;

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