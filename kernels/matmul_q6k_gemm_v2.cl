// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

// M8.K.1b — Q6_K GEMM with shrunk SLM (Xe-LPG occupancy fix).
//
// Same v2-idea as matmul_q8_0_gemm_v2 (see that kernel for the SLM-
// vs-occupancy analysis). Q6_K super-block is 256 elements → X_TILE=256
// = exactly one super-block per K-tile.
//
// SLM: 256 * 8 * 4 = 8 KiB / WG. v1 was 32 KiB / WG (X_TILE=1024).
// 4× more resident WGs on the Xe Vector Engine scheduler.
//
// Everything else (WG=64, SG=16, 4 outputs/WG, Kahan compensation,
// per-half dequant loop) is deliberately identical to v1 so any perf
// delta is attributable to the SLM axis alone.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

#ifndef MATMUL_Q6K_V2_LOCAL
#define MATMUL_Q6K_V2_LOCAL 64
#endif

#ifndef MATMUL_Q6K_V2_SG
#define MATMUL_Q6K_V2_SG 16
#endif

#ifndef MATMUL_Q6K_GEMM_V2_M_TILE
#define MATMUL_Q6K_GEMM_V2_M_TILE 8
#endif

#define MATMUL_Q6K_V2_OUTPUTS_PER_GROUP \
    (MATMUL_Q6K_V2_LOCAL / MATMUL_Q6K_V2_SG)

#define Q6K_BLOCK_ELEMENTS 256
#define Q6K_BLOCK_BYTES    210

// One super-block per K-tile. SLM: 256 * 8 * 4 = 8 KiB / WG.
#define X_TILE_ELEMENTS 256

__attribute__((reqd_work_group_size(MATMUL_Q6K_V2_LOCAL, 1, 1)))
__attribute__((intel_reqd_sub_group_size(MATMUL_Q6K_V2_SG)))
__kernel void matmul_q6k_gemm_v2(
    __global const float* X,
    __global const uchar* W,
    __global       float* Y,
    const int             K,
    const int             N,
    const int             M)
{
    __local float xTile[X_TILE_ELEMENTS][MATMUL_Q6K_GEMM_V2_M_TILE];

    const int  wgN     = (int)get_group_id(0);
    const int  wgM     = (int)get_group_id(1);
    const int  sgInWg  = (int)get_sub_group_id();
    const int  sgLocal = (int)get_sub_group_local_id();
    const int  tid     = (int)get_local_id(0);
    const int  lsize   = (int)get_local_size(0);
    const int  n       = wgN * MATMUL_Q6K_V2_OUTPUTS_PER_GROUP + sgInWg;
    const int  mBase   = wgM * MATMUL_Q6K_GEMM_V2_M_TILE;
    const bool nActive = (n < N);
    const int  nSuper  = K / Q6K_BLOCK_ELEMENTS;

    float          sum[MATMUL_Q6K_GEMM_V2_M_TILE];
    volatile float kc [MATMUL_Q6K_GEMM_V2_M_TILE];

    #pragma unroll
    for (int mm = 0; mm < MATMUL_Q6K_GEMM_V2_M_TILE; ++mm) {
        sum[mm] = 0.0f;
        kc [mm] = 0.0f;
    }

    for (int tile = 0; tile < K; tile += X_TILE_ELEMENTS) {
        const int tileK = min(X_TILE_ELEMENTS, K - tile);

        const int loadTotal = MATMUL_Q6K_GEMM_V2_M_TILE * X_TILE_ELEMENTS;
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
                W + (size_t)n * (size_t)nSuper * Q6K_BLOCK_BYTES;

            const int sbStart  = tile / Q6K_BLOCK_ELEMENTS;
            const int sbInTile = X_TILE_ELEMENTS / Q6K_BLOCK_ELEMENTS;
            const int sbEnd    = min(sbStart + sbInTile, nSuper);

            for (int sb = sbStart; sb < sbEnd; ++sb) {
                __global const uchar* block = row + sb * Q6K_BLOCK_BYTES;

                __global const uchar* ql = block;
                __global const uchar* qh = block + 128;
                __global const char*  sc =
                    (__global const char*)(block + 192);
                const float d =
                    vload_half(0, (__global const half*)(block + 208));

                const int xLocalBase = (sb - sbStart) * Q6K_BLOCK_ELEMENTS;

                for (int hIdx = 0; hIdx < 2; ++hIdx) {
                    const int xHalfBase = xLocalBase + hIdx * 128;
                    __global const uchar* qlp = ql + hIdx * 64;
                    __global const uchar* qhp = qh + hIdx * 32;
                    __global const char*  scp = sc + hIdx * 8;

                    for (int l = sgLocal; l < 32; l += MATMUL_Q6K_V2_SG) {
                        const int is = l / 16;

                        const char q1 = (char)((qlp[l +  0] & 0x0F) |
                                               (((qhp[l] >> 0) & 0x03) << 4)) - 32;
                        const char q2 = (char)((qlp[l + 32] & 0x0F) |
                                               (((qhp[l] >> 2) & 0x03) << 4)) - 32;
                        const char q3 = (char)((qlp[l +  0] >> 4) |
                                               (((qhp[l] >> 4) & 0x03) << 4)) - 32;
                        const char q4 = (char)((qlp[l + 32] >> 4) |
                                               (((qhp[l] >> 6) & 0x03) << 4)) - 32;

                        const float w0 = d * (float)scp[is + 0] * (float)q1;
                        const float w1 = d * (float)scp[is + 2] * (float)q2;
                        const float w2 = d * (float)scp[is + 4] * (float)q3;
                        const float w3 = d * (float)scp[is + 6] * (float)q4;

                        #define KAHAN_ADD(mm, term)                              \
                            do {                                                  \
                                const float _y = (term) - kc[mm];                 \
                                const float _t = sum[mm] + _y;                    \
                                kc[mm]  = (_t - sum[mm]) - _y;                    \
                                sum[mm] = _t;                                     \
                            } while (0)

                        #pragma unroll
                        for (int mm = 0; mm < MATMUL_Q6K_GEMM_V2_M_TILE; ++mm) {
                            KAHAN_ADD(mm, xTile[xHalfBase + l +  0][mm] * w0);
                            KAHAN_ADD(mm, xTile[xHalfBase + l + 32][mm] * w1);
                            KAHAN_ADD(mm, xTile[xHalfBase + l + 64][mm] * w2);
                            KAHAN_ADD(mm, xTile[xHalfBase + l + 96][mm] * w3);
                        }

                        #undef KAHAN_ADD
                    }
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    #pragma unroll
    for (int mm = 0; mm < MATMUL_Q6K_GEMM_V2_M_TILE; ++mm) {
        float s = sum[mm] + kc[mm];
        s = sub_group_reduce_add(s);
        if (nActive && sgLocal == 0) {
            const int mAct = mBase + mm;
            if (mAct < M) {
                Y[(size_t)mAct * (size_t)N + n] = s;
            }
        }
    }
}