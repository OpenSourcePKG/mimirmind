// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

// Matrix-vector multiply with Q4_K weights, on-the-fly dequant.
//
//   Y[n] = sum_{k=0..K-1} X[k] * dequant_q4k(W, n, k)
//
//   X:  [K]     F32 dense vector (single token / M=1)
//   W:  [N, K]  Q4_K — each row is (K/256) super-blocks of 144 bytes
//   Y:  [N]     F32 dense vector
//
// Launch geometry (M5h subgroup-reduce, SLM-free):
//   local_size_x          = MATMUL_Q4K_LOCAL   (64)
//   sub_group_size        = MATMUL_Q4K_SG      (16) via intel_reqd_sub_group_size
//   outputs per workgroup = MATMUL_Q4K_LOCAL / MATMUL_Q4K_SG  (= 4)
//   global_size_x         = ceil(N / 4) * 64
//
// Each subgroup of 16 threads cooperatively computes ONE output by
// splitting the K-reduction across its members and collapsing with
// sub_group_reduce_add. Compared to M5e (1 thread per output), per-
// thread compute drops ~16×, and small-N matmuls (Q/K/V projections)
// saturate the iGPU much better.
//
// X is read straight from global — NO local-memory X-tile. For a matvec
// every workgroup consumes the SAME X vector, so X stays L2/L3-resident
// and cache-hot; the earlier SLM staging bought nothing while its two
// per-tile workgroup barriers throttled occupancy. Dropping it measured
// +7.7% end-to-end decode on Qwen2.5-1.5B-Q4_K (Xe-LPG, RP0), parity
// unchanged. `n` is subgroup-uniform, so the early return is
// subgroup-safe for sub_group_reduce_add.
//
// Reference: ggml-quants.c dequantize_row_q4_K. Sub-block iteration
// matches compute/Dequant.cpp dequantQ4K.

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

    float sum = 0.0f;

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

            // 16 threads share the 32-element half — each does 2.
            for (int l = sgLocal; l < 32; l += MATMUL_Q4K_SG) {
                const uchar q   = qs[qsOffset + l];
                const float qLo = (float)(q & 0x0F);
                const float qHi = (float)(q >> 4);
                sum = mad(X[xLoBase + l], d1 * qLo - m1, sum);
                sum = mad(X[xHiBase + l], d2 * qHi - m2, sum);
            }
        }
    }

    // Collapse the 16 partial sums in this subgroup into one.
    sum = sub_group_reduce_add(sum);

    if (sgLocal == 0) {
        Y[n] = sum;
    }
}