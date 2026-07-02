// M8.K.1b — Q4_K GEMM with shrunk SLM (Xe-LPG occupancy fix).
//
// Same v2-idea as matmul_q8_0_gemm_v2 (see that kernel for the SLM-
// vs-occupancy analysis). Q4_K super-block is 256 elements → X_TILE=256
// = exactly one super-block per K-tile.
//
// SLM: 256 * 8 * 4 = 8 KiB / WG. v1 was 32 KiB / WG (X_TILE=1024).
// 4× more resident WGs on the Xe Vector Engine scheduler.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

#ifndef MATMUL_Q4K_V2_LOCAL
#define MATMUL_Q4K_V2_LOCAL 64
#endif

#ifndef MATMUL_Q4K_V2_SG
#define MATMUL_Q4K_V2_SG 16
#endif

#ifndef MATMUL_Q4K_GEMM_V2_M_TILE
#define MATMUL_Q4K_GEMM_V2_M_TILE 8
#endif

#define MATMUL_Q4K_V2_OUTPUTS_PER_GROUP \
    (MATMUL_Q4K_V2_LOCAL / MATMUL_Q4K_V2_SG)

#define Q4K_BLOCK_ELEMENTS 256
#define Q4K_BLOCK_BYTES    144

// One super-block per K-tile. SLM: 256 * 8 * 4 = 8 KiB / WG.
#define X_TILE_ELEMENTS 256

inline uchar2 q4k_scale_min_v2(int j, __global const uchar* sc) {
    uchar s, m;
    if (j < 4) {
        s = sc[j]     & 0x3F;
        m = sc[j + 4] & 0x3F;
    } else {
        s = (sc[j + 4] & 0x0F) | ((sc[j - 4] >> 6) << 4);
        m = (sc[j + 4] >> 4)   | ((sc[j]     >> 6) << 4);
    }
    return (uchar2)(s, m);
}

__attribute__((reqd_work_group_size(MATMUL_Q4K_V2_LOCAL, 1, 1)))
__attribute__((intel_reqd_sub_group_size(MATMUL_Q4K_V2_SG)))
__kernel void matmul_q4k_gemm_v2(
    __global const float* X,
    __global const uchar* W,
    __global       float* Y,
    const int             K,
    const int             N,
    const int             M)
{
    __local float xTile[X_TILE_ELEMENTS][MATMUL_Q4K_GEMM_V2_M_TILE];

    const int  wgN     = (int)get_group_id(0);
    const int  wgM     = (int)get_group_id(1);
    const int  sgInWg  = (int)get_sub_group_id();
    const int  sgLocal = (int)get_sub_group_local_id();
    const int  tid     = (int)get_local_id(0);
    const int  lsize   = (int)get_local_size(0);
    const int  n       = wgN * MATMUL_Q4K_V2_OUTPUTS_PER_GROUP + sgInWg;
    const int  mBase   = wgM * MATMUL_Q4K_GEMM_V2_M_TILE;
    const bool nActive = (n < N);
    const int  nSuper  = K / Q4K_BLOCK_ELEMENTS;

    float sum[MATMUL_Q4K_GEMM_V2_M_TILE];

    #pragma unroll
    for (int mm = 0; mm < MATMUL_Q4K_GEMM_V2_M_TILE; ++mm) {
        sum[mm] = 0.0f;
    }

    for (int tile = 0; tile < K; tile += X_TILE_ELEMENTS) {
        const int tileK = min(X_TILE_ELEMENTS, K - tile);

        const int loadTotal = MATMUL_Q4K_GEMM_V2_M_TILE * X_TILE_ELEMENTS;
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
                W + (size_t)n * (size_t)nSuper * Q4K_BLOCK_BYTES;

            const int sbStart  = tile / Q4K_BLOCK_ELEMENTS;
            const int sbInTile = X_TILE_ELEMENTS / Q4K_BLOCK_ELEMENTS;
            const int sbEnd    = min(sbStart + sbInTile, nSuper);

            for (int sb = sbStart; sb < sbEnd; ++sb) {
                __global const uchar* block = row + sb * Q4K_BLOCK_BYTES;

                const float d    = vload_half(0, (__global const half*)block);
                const float dmin = vload_half(1, (__global const half*)block);

                __global const uchar* scales = block + 4;
                __global const uchar* qs     = block + 16;

                const int xLocalBase = (sb - sbStart) * Q4K_BLOCK_ELEMENTS;

                for (int j = 0; j < 8; j += 2) {
                    const uchar2 sm0 = q4k_scale_min_v2(j,     scales);
                    const uchar2 sm1 = q4k_scale_min_v2(j + 1, scales);
                    const float  d1  = d    * (float)sm0.x;
                    const float  m1  = dmin * (float)sm0.y;
                    const float  d2  = d    * (float)sm1.x;
                    const float  m2  = dmin * (float)sm1.y;

                    const int qsOffset = (j / 2) * 32;
                    const int xLoBase  = xLocalBase + j * 32;
                    const int xHiBase  = xLocalBase + (j + 1) * 32;

                    for (int l = sgLocal; l < 32; l += MATMUL_Q4K_V2_SG) {
                        const uchar q   = qs[qsOffset + l];
                        const float wLo = d1 * (float)(q & 0x0F) - m1;
                        const float wHi = d2 * (float)(q >> 4)   - m2;

                        #pragma unroll
                        for (int mm = 0; mm < MATMUL_Q4K_GEMM_V2_M_TILE; ++mm) {
                            sum[mm] = mad(xTile[xLoBase + l][mm], wLo, sum[mm]);
                            sum[mm] = mad(xTile[xHiBase + l][mm], wHi, sum[mm]);
                        }
                    }
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    #pragma unroll
    for (int mm = 0; mm < MATMUL_Q4K_GEMM_V2_M_TILE; ++mm) {
        float s = sub_group_reduce_add(sum[mm]);
        if (nActive && sgLocal == 0) {
            const int mAct = mBase + mm;
            if (mAct < M) {
                Y[(size_t)mAct * (size_t)N + n] = s;
            }
        }
    }
}