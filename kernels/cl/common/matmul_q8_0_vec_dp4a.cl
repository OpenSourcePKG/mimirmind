// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

// DP4A matvec for Q8_0 weights with pre-quantised int8 activation.
//
//   Y[n] = xScale × sum_{b=0..nBlocks-1}
//              d[b] × sum_{i=0..31}( Xq[b*32+i] × Wq[n, b, i] )
//
//   Xq:     [K]     int8  (produced by x_quant_i8.cl)
//   Xscale: scalar F32    (pointer to a single float)
//   W:      [N, K] Q8_0   (K/32 blocks of 34 bytes, {fp16 d, int8 qs[32]})
//   Y:      [N]     F32
//
// Compared to matmul_q8_0_vec (M8.G), the inner accumulator stays in
// int32 across each 32-element block via DP4A (dot_4x8packed_ss_int),
// with only one fp32 multiply per block for d × xScale. That's 4×
// fewer FP32 muls in the hot loop and the char4 dot maps to Xe-LPG's
// IDPAS instruction.
//
// Launch geometry (M8.H.3 revision): matches matmul_q8_0_vec exactly
// so the two kernels are apples-to-apples in autotune. The SG=8
// variant we shipped in M8.H.1 was 30 % slower than plain matvec on
// Xe-LPG at 800 MHz (autotune bench, 2026-07-02) — the "no idle
// lanes" argument turned out to matter less than matching the
// hardware's preferred SG width.
//
//   local_size_x          = MATMUL_Q8_0_DP4A_LOCAL  (64)
//   sub_group_size        = MATMUL_Q8_0_DP4A_SG    (16) via intel_reqd_sub_group_size
//   outputs per workgroup = LOCAL / SG              (= 4)
//   global_size_x         = ceil(N / 4) * 64
//
// With 16 lanes but only 8 char4 chunks per 32-element block, each
// outer iteration processes TWO consecutive Q8_0 blocks: lanes 0..7
// cover block `b`, lanes 8..15 cover block `b+1`. Each lane owns one
// char4 dot and multiplies by ITS block's d × xScale before
// accumulating. sub_group_reduce_add at the end sums all 16 lane
// contributions (64 element products = 2 blocks worth).

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable

#ifndef MATMUL_Q8_0_DP4A_LOCAL
#define MATMUL_Q8_0_DP4A_LOCAL 64
#endif

#ifndef MATMUL_Q8_0_DP4A_SG
#define MATMUL_Q8_0_DP4A_SG 16
#endif

#define MATMUL_Q8_0_DP4A_OUTPUTS_PER_GROUP \
    (MATMUL_Q8_0_DP4A_LOCAL / MATMUL_Q8_0_DP4A_SG)

#define Q8_0_BLOCK_ELEMENTS 32
#define Q8_0_BLOCK_BYTES    34
#define Q8_0_BLOCK_CHAR4S   (Q8_0_BLOCK_ELEMENTS / 4)  // 8

// 1024 x elements = 32 blocks = 1 KiB SLM per workgroup (int8, so 4×
// smaller than the fp32 tile in matmul_q8_0_vec).
#define X_TILE_ELEMENTS 1024

__attribute__((reqd_work_group_size(MATMUL_Q8_0_DP4A_LOCAL, 1, 1)))
__attribute__((intel_reqd_sub_group_size(MATMUL_Q8_0_DP4A_SG)))
__kernel void matmul_q8_0_vec_dp4a(
    __global const char*  Xq,
    __global const float* Xscale,
    __global const uchar* W,
    __global       float* Y,
    const int             K,
    const int             N)
{
    __local char xTile[X_TILE_ELEMENTS];

    const int  wg      = (int)get_group_id(0);
    const int  sgInWg  = (int)get_sub_group_id();           // 0..3
    const int  sgLocal = (int)get_sub_group_local_id();     // 0..15
    const int  tid     = (int)get_local_id(0);
    const int  lsize   = (int)get_local_size(0);
    const int  n       = wg * MATMUL_Q8_0_DP4A_OUTPUTS_PER_GROUP + sgInWg;
    const bool active  = (n < N);
    const int  nBlocks = K / Q8_0_BLOCK_ELEMENTS;

    const float xScale = *Xscale;

    // Lane assignment for the 2-blocks-per-iteration pattern:
    //   lanes 0..7  → block b,   char4 index sgLocal
    //   lanes 8..15 → block b+1, char4 index sgLocal - 8
    const int laneBlockOff = sgLocal >> 3;               // 0 or 1
    const int laneChar4Idx = sgLocal & (Q8_0_BLOCK_CHAR4S - 1);  // 0..7

    float sum = 0.0f;

    for (int tile = 0; tile < K; tile += X_TILE_ELEMENTS) {
        const int tileK = min(X_TILE_ELEMENTS, K - tile);
        for (int i = tid; i < tileK; i += lsize) {
            xTile[i] = Xq[tile + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (active) {
            __global const uchar* row =
                W + (size_t)n * (size_t)nBlocks * Q8_0_BLOCK_BYTES;

            const int blockStart   = tile / Q8_0_BLOCK_ELEMENTS;
            const int blocksInTile = X_TILE_ELEMENTS / Q8_0_BLOCK_ELEMENTS;
            const int blockEnd     = min(blockStart + blocksInTile, nBlocks);

            for (int b = blockStart; b < blockEnd; b += 2) {
                const int bMy = b + laneBlockOff;
                if (bMy < blockEnd) {
                    __global const uchar* block =
                        row + bMy * Q8_0_BLOCK_BYTES;
                    const float d =
                        vload_half(0, (__global const half*)(block));
                    __global const char* wq_ptr =
                        (__global const char*)(block + 2);

                    const int xLocalBase =
                        (bMy - blockStart) * Q8_0_BLOCK_ELEMENTS;
                    __local const char* xq_ptr = xTile + xLocalBase;

                    const char4 wq = vload4(laneChar4Idx, wq_ptr);
                    const char4 xq = vload4(laneChar4Idx, xq_ptr);
                    const int   dp =
                        dot_4x8packed_ss_int(as_uint(wq), as_uint(xq));

                    sum = mad((float)dp, d * xScale, sum);
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    sum = sub_group_reduce_add(sum);

    if (active && sgLocal == 0) {
        Y[n] = sum;
    }
}