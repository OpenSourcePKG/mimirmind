// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// SLM-free Q5_K matvec with widened loads and split accumulators — the
// same transformation applied to matmul_q4k_vec, extended for Q5_K's 5th
// (high) bit. For a matvec X is shared by every workgroup and cache-hot,
// so the local-memory X-tile and its two per-tile barriers are dropped;
// each of the 16 lanes owns the contiguous pair {2l, 2l+1} of every
// 32-element half and issues uchar2 (qs, qh) + float2 (X) loads. Two
// independent FMA chains (sumLo/sumHi) break the serial accumulation.
// The qh mask no longer shifts sequentially — it is derived per j-pair as
// (1,2) << (2*jp), so the j-loop carries no cross-iteration dependency.
//
// Q5_K super-block (176 B): fp16 d, fp16 dmin, uchar scales[12],
// uchar qh[32] (one high bit per quant), uchar qs[128] (low nibbles).
//   q = (qs_nibble) | (qh_bit << 4) in [0..31];  value = d*scale*q - dmin*min

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

#ifndef MATMUL_Q5K_LOCAL
#define MATMUL_Q5K_LOCAL 64
#endif

#ifndef MATMUL_Q5K_SG
#define MATMUL_Q5K_SG 16
#endif

#define MATMUL_Q5K_OUTPUTS_PER_GROUP (MATMUL_Q5K_LOCAL / MATMUL_Q5K_SG)

#define Q5K_BLOCK_ELEMENTS 256
#define Q5K_BLOCK_BYTES    176

inline uchar2 q5k_scale_min(int j, __global const uchar* sc) {
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

__attribute__((reqd_work_group_size(MATMUL_Q5K_LOCAL, 1, 1)))
__attribute__((intel_reqd_sub_group_size(MATMUL_Q5K_SG)))
__kernel void matmul_q5k_vec(
    __global const float* X,
    __global const uchar* W,
    __global       float* Y,
    const int             K,
    const int             N)
{
    const int  wg       = (int)get_group_id(0);
    const int  sgInWg   = (int)get_sub_group_id();
    const int  sgLocal  = (int)get_sub_group_local_id();
    const int  n        = wg * MATMUL_Q5K_OUTPUTS_PER_GROUP + sgInWg;
    if (n >= N) return;

    const int  nSuper   = K / Q5K_BLOCK_ELEMENTS;
    __global const uchar* row =
        W + (size_t)n * (size_t)nSuper * Q5K_BLOCK_BYTES;

    const int e2 = 2 * sgLocal;

    float sumLo = 0.0f;
    float sumHi = 0.0f;

    for (int sb = 0; sb < nSuper; ++sb) {
        __global const uchar* block = row + sb * Q5K_BLOCK_BYTES;

        const float d    = vload_half(0, (__global const half*)block);
        const float dmin = vload_half(1, (__global const half*)block);

        __global const uchar* scales = block + 4;    // 12 bytes
        __global const uchar* qh     = block + 16;   // 32 bytes
        __global const uchar* qs     = block + 48;   // 128 bytes

        const int xbase = sb * Q5K_BLOCK_ELEMENTS;

        const uchar2 qhh = vload2(0, qh + e2);        // qh[2l], qh[2l+1]

        for (int j = 0; j < 8; j += 2) {
            const int jp = j / 2;
            const uchar2 sm0 = q5k_scale_min(j,     scales);
            const uchar2 sm1 = q5k_scale_min(j + 1, scales);
            const float  d1  = d    * (float)sm0.x;
            const float  m1  = dmin * (float)sm0.y;
            const float  d2  = d    * (float)sm1.x;
            const float  m2  = dmin * (float)sm1.y;

            const uchar u1 = (uchar)(1u << (2 * jp));
            const uchar u2 = (uchar)(2u << (2 * jp));

            const int qsOffset = jp * 32;
            const int xLoBase  = xbase + j * 32;
            const int xHiBase  = xbase + (j + 1) * 32;

            const uchar2 qq  = vload2(0, qs + qsOffset + e2);
            const float2 xlo = vload2(0, X + xLoBase + e2);
            const float2 xhi = vload2(0, X + xHiBase + e2);

            const float q0lo = (float)((qq.s0 & 0x0F) + ((qhh.s0 & u1) ? 16 : 0));
            const float q1lo = (float)((qq.s1 & 0x0F) + ((qhh.s1 & u1) ? 16 : 0));
            const float q0hi = (float)((qq.s0 >> 4)   + ((qhh.s0 & u2) ? 16 : 0));
            const float q1hi = (float)((qq.s1 >> 4)   + ((qhh.s1 & u2) ? 16 : 0));

            sumLo = mad(xlo.s0, d1 * q0lo - m1, sumLo);
            sumLo = mad(xlo.s1, d1 * q1lo - m1, sumLo);
            sumHi = mad(xhi.s0, d2 * q0hi - m2, sumHi);
            sumHi = mad(xhi.s1, d2 * q1hi - m2, sumHi);
        }
    }

    float sum = sub_group_reduce_add(sumLo + sumHi);

    if (sgLocal == 0) {
        Y[n] = sum;
    }
}
