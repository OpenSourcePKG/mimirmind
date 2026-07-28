// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

// DP4A matvec for Q4_K weights with pre-quantised int8 activation.
//
//   Y[n] = xScale × sum_{sb=0..nSuper-1} sum_{j=0..7}
//              ( d * scale_j × sum_l q_j[l] × Xq[sb*256 + j*32 + l]
//              − dmin * min_j × sum_l Xq[sb*256 + j*32 + l] )
//
//   Xq:     [K]     int8  (produced by x_quant_i8.cl)
//   Xscale: scalar F32    (single float per row)
//   W:      [N, K] Q4_K   (K/256 super-blocks of 144 bytes)
//   Y:      [N]     F32
//
// Q4_K super-block layout (144 bytes):
//   fp16 d        (2)  super-block scale
//   fp16 dmin     (2)  super-block min-scale
//   uchar sc[12]      packed 6-bit sub-scales + 6-bit sub-mins for 8 sub-blocks
//   uchar qs[128]     256 nibbles (128 bytes, low nibble = sub-block j, high = j+1)
//
// Sub-block iteration mirrors matmul_q4k_vec / compute::dequantQ4K:
//   for j in 0, 2, 4, 6:
//     qsOffset = (j/2) * 32   // 32 bytes cover paired sub-blocks {j, j+1}
//     lo nibble → sub-block j; hi nibble → sub-block j+1
//
// DP4A packing:
//   4 int8 per uint32. Each 32-element sub-block = 8 char4 chunks.
//   Lanes 0..7 handle sub-block j (lo nibbles); lanes 8..15 handle
//   sub-block j+1 (hi nibbles). Each lane owns exactly one char4 chunk
//   (4 consecutive elements) per {j, j+1} pair.
//
// Two dot products per lane per pair:
//   dp    = dot_4x8packed_ss_int(wq, xq)         → sum q*xq
//   xsum  = dot_4x8packed_ss_int(0x01010101, xq) → sum xq
//   partial += xScale * (d * scale_j * dp − dmin * min_j * xsum)
//
// Launch geometry (matches matmul_q4k_vec so autotune is apples-to-apples):
//   local_size_x          = MATMUL_Q4K_DP4A_LOCAL (64)
//   sub_group_size        = MATMUL_Q4K_DP4A_SG    (16)
//   outputs per workgroup = LOCAL / SG            (= 4)

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable

#ifndef MATMUL_Q4K_DP4A_LOCAL
#define MATMUL_Q4K_DP4A_LOCAL 64
#endif

#ifndef MATMUL_Q4K_DP4A_SG
#define MATMUL_Q4K_DP4A_SG 16
#endif

#define MATMUL_Q4K_DP4A_OUTPUTS_PER_GROUP \
    (MATMUL_Q4K_DP4A_LOCAL / MATMUL_Q4K_DP4A_SG)

#define Q4K_BLOCK_ELEMENTS 256
#define Q4K_BLOCK_BYTES    144

// 1024 x elements = 4 super-blocks = 1 KiB SLM per WG (int8, 4× smaller
// than the fp32 tile in matmul_q4k_vec.cl).
#define X_TILE_ELEMENTS 1024

inline uchar2 q4k_dp4a_scale_min(int j, __global const uchar* sc) {
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

__attribute__((reqd_work_group_size(MATMUL_Q4K_DP4A_LOCAL, 1, 1)))
__attribute__((intel_reqd_sub_group_size(MATMUL_Q4K_DP4A_SG)))
__kernel void matmul_q4k_vec_dp4a(
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
    const int  n       = wg * MATMUL_Q4K_DP4A_OUTPUTS_PER_GROUP + sgInWg;
    const bool active  = (n < N);
    const int  nSuper  = K / Q4K_BLOCK_ELEMENTS;

    const float xScale = *Xscale;

    // Lane split: 0..7 → sub-block j (lo nibbles), 8..15 → j+1 (hi nibbles).
    const int laneBlockOff = sgLocal >> 3;                 // 0 or 1
    const int laneChar4Idx = sgLocal & 7;                  // 0..7

    // For unpacking hi nibbles we only need a shift-and-mask on each
    // byte, so pre-compute the shift once.
    const int  nibbleShift = laneBlockOff * 4;             // 0 or 4

    float sum = 0.0f;

    for (int tile = 0; tile < K; tile += X_TILE_ELEMENTS) {
        const int tileK = min(X_TILE_ELEMENTS, K - tile);
        for (int i = tid; i < tileK; i += lsize) {
            xTile[i] = Xq[tile + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (active) {
            __global const uchar* row =
                W + (size_t)n * (size_t)nSuper * Q4K_BLOCK_BYTES;

            const int sbStart  = tile / Q4K_BLOCK_ELEMENTS;
            const int sbInTile = X_TILE_ELEMENTS / Q4K_BLOCK_ELEMENTS;
            const int sbEnd    = min(sbStart + sbInTile, nSuper);

            for (int sb = sbStart; sb < sbEnd; ++sb) {
                __global const uchar* block = row + sb * Q4K_BLOCK_BYTES;

                const float d    = vload_half(0, (__global const half*)block);
                const float dmin = vload_half(1, (__global const half*)block);

                __global const uchar* scales = block + 4;    // 12 bytes
                __global const uchar* qs     = block + 16;   // 128 bytes

                const int xLocalBase = (sb - sbStart) * Q4K_BLOCK_ELEMENTS;

                for (int j = 0; j < 8; j += 2) {
                    // Each lane picks either sub-block j or j+1 based on
                    // laneBlockOff. The 32-byte qs range shared by the
                    // paired sub-blocks starts at qsOffset.
                    const uchar2 sm = q4k_dp4a_scale_min(j + laneBlockOff,
                                                        scales);
                    const float dsub = d    * (float)sm.x;
                    const float msub = dmin * (float)sm.y;

                    const int qsOffset =
                        (j / 2) * 32 + laneChar4Idx * 4;
                    const int xLocalOff =
                        xLocalBase + (j + laneBlockOff) * 32
                                   + laneChar4Idx * 4;

                    // Read 4 packed bytes; extract the correct nibble
                    // side into an int8 char4.
                    const uchar4 raw = vload4(0, qs + qsOffset);
                    const char4  wq  = (char4)(
                        (raw.x >> nibbleShift) & 0x0F,
                        (raw.y >> nibbleShift) & 0x0F,
                        (raw.z >> nibbleShift) & 0x0F,
                        (raw.w >> nibbleShift) & 0x0F);
                    const char4  xq  = vload4(0, xTile + xLocalOff);

                    const int dp   = dot_4x8packed_ss_int(
                        as_uint(wq), as_uint(xq));
                    const int xsum = dot_4x8packed_ss_int(
                        0x01010101u, as_uint(xq));

                    sum = mad((float)dp,   dsub * xScale, sum);
                    sum = mad((float)xsum, -msub * xScale, sum);
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