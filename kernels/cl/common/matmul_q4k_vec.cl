// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// SLM-free Q4_K matvec with split accumulators and widened loads. Each
// of the 16 subgroup lanes owns a contiguous 2-element span of every
// 32-element half (lane l -> elements 2l, 2l+1), so it issues one uchar2
// weight load plus two float2 X loads per half instead of two scalar
// passes of single-byte / single-float loads. The wider memory
// transactions keep more loads in flight, which is the lever on this
// memory-latency-bound path; two independent FMA chains (sumLo/sumHi)
// break the otherwise-serial accumulation. Measured +16.7% end-to-end
// decode over the earlier SLM-staged single-accumulator kernel on
// Qwen2.5-1.5B-Q4_K (Xe-LPG, RP0), parity unchanged.
//
// Alignment: qs = block+16, block = row + sb*144 (144 % 16 == 0), so qs
// is 16-byte aligned; qsOffset in {0,32,64,96} is even -> the uchar2 /
// float2 loads are well-formed.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

#ifndef MATMUL_Q4K_LOCAL
#define MATMUL_Q4K_LOCAL 64
#endif

#ifndef MATMUL_Q4K_SG
#define MATMUL_Q4K_SG 16
#endif

#define MATMUL_Q4K_OUTPUTS_PER_GROUP (MATMUL_Q4K_LOCAL / MATMUL_Q4K_SG)

#define Q4K_BLOCK_ELEMENTS 256
#define Q4K_BLOCK_BYTES    144

inline uchar2 q4k_scale_min(int j, __global const uchar* sc) {
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

__attribute__((reqd_work_group_size(MATMUL_Q4K_LOCAL, 1, 1)))
__attribute__((intel_reqd_sub_group_size(MATMUL_Q4K_SG)))
__kernel void matmul_q4k_vec(
    __global const float* X,
    __global const uchar* W,
    __global       float* Y,
    const int             K,
    const int             N)
{
    const int  wg       = (int)get_group_id(0);
    const int  sgInWg   = (int)get_sub_group_id();          // 0..3
    const int  sgLocal  = (int)get_sub_group_local_id();    // 0..15
    const int  n        = wg * MATMUL_Q4K_OUTPUTS_PER_GROUP + sgInWg;
    if (n >= N) return;                                     // subgroup-uniform

    const int  nSuper   = K / Q4K_BLOCK_ELEMENTS;
    __global const uchar* row =
        W + (size_t)n * (size_t)nSuper * Q4K_BLOCK_BYTES;

    const int e2 = 2 * sgLocal;                             // 0,2,..,30

    float sumLo = 0.0f;
    float sumHi = 0.0f;

    for (int sb = 0; sb < nSuper; ++sb) {
        __global const uchar* block = row + sb * Q4K_BLOCK_BYTES;

        const float d    = vload_half(0, (__global const half*)block);
        const float dmin = vload_half(1, (__global const half*)block);

        __global const uchar* scales = block + 4;    // 12 bytes
        __global const uchar* qs     = block + 16;   // 128 bytes

        const int xbase = sb * Q4K_BLOCK_ELEMENTS;

        for (int j = 0; j < 8; j += 2) {
            const uchar2 sm0 = q4k_scale_min(j,     scales);
            const uchar2 sm1 = q4k_scale_min(j + 1, scales);
            const float  d1  = d    * (float)sm0.x;
            const float  m1  = dmin * (float)sm0.y;
            const float  d2  = d    * (float)sm1.x;
            const float  m2  = dmin * (float)sm1.y;

            const int qsOffset = (j / 2) * 32;
            const int xLoBase  = xbase + j * 32;
            const int xHiBase  = xbase + (j + 1) * 32;

            // Lane owns the contiguous pair {e2, e2+1} of this 32-half.
            const uchar2 qq  = vload2(0, qs + qsOffset + e2);
            const float2 xlo = vload2(0, X + xLoBase + e2);
            const float2 xhi = vload2(0, X + xHiBase + e2);

            const float q0lo = (float)(qq.s0 & 0x0F);
            const float q1lo = (float)(qq.s1 & 0x0F);
            const float q0hi = (float)(qq.s0 >> 4);
            const float q1hi = (float)(qq.s1 >> 4);

            sumLo = mad(xlo.s0, d1 * q0lo - m1, sumLo);
            sumLo = mad(xlo.s1, d1 * q1lo - m1, sumLo);
            sumHi = mad(xhi.s0, d2 * q0hi - m2, sumHi);
            sumHi = mad(xhi.s1, d2 * q1hi - m2, sumHi);
        }
    }

    // Combine the two independent chains, then collapse across the subgroup.
    float sum = sub_group_reduce_add(sumLo + sumHi);

    if (sgLocal == 0) {
        Y[n] = sum;
    }
}
