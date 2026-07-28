// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// SLM-free Q6_K matvec with plain-fma split accumulators. Drops (1) the
// local-memory X-tile and its two per-tile barriers — for a matvec X is
// shared by every workgroup and stays cache-hot — and (2) the
// volatile-Kahan accumulation, which forced every term through memory (the
// same latency killer the Q6_K GEMM fix c82a349 removed). Four independent
// accumulators (one per q1..q4 stream) break the dependency chain, which
// also recovers most of Kahan's precision without the volatile traffic.
// Measured +11.9% end-to-end decode on Qwen2.5-1.5B-Q4_K_M (its
// down_proj/output are Q6_K) on Xe-LPG (RP0); startup vec-parity maxDiff
// 1.9e-4, well inside the gate. Also speeds the 26B expert-down decode.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

#ifndef MATMUL_Q6K_LOCAL
#define MATMUL_Q6K_LOCAL 64
#endif

#ifndef MATMUL_Q6K_SG
#define MATMUL_Q6K_SG 16
#endif

#define MATMUL_Q6K_OUTPUTS_PER_GROUP (MATMUL_Q6K_LOCAL / MATMUL_Q6K_SG)

#define Q6K_BLOCK_ELEMENTS 256
#define Q6K_BLOCK_BYTES    210

__attribute__((reqd_work_group_size(MATMUL_Q6K_LOCAL, 1, 1)))
__attribute__((intel_reqd_sub_group_size(MATMUL_Q6K_SG)))
__kernel void matmul_q6k_vec(
    __global const float* X,
    __global const uchar* W,
    __global       float* Y,
    const int             K,
    const int             N)
{
    const int  wg      = (int)get_group_id(0);
    const int  sgInWg  = (int)get_sub_group_id();
    const int  sgLocal = (int)get_sub_group_local_id();
    const int  n       = wg * MATMUL_Q6K_OUTPUTS_PER_GROUP + sgInWg;
    if (n >= N) return;

    const int  nSuper  = K / Q6K_BLOCK_ELEMENTS;
    __global const uchar* row =
        W + (size_t)n * (size_t)nSuper * Q6K_BLOCK_BYTES;

    float acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f, acc4 = 0.0f;

    for (int sb = 0; sb < nSuper; ++sb) {
        __global const uchar* block = row + sb * Q6K_BLOCK_BYTES;

        __global const uchar* ql = block;              // 128 bytes
        __global const uchar* qh = block + 128;        // 64 bytes
        __global const char*  sc = (__global const char*)(block + 192);
        const float d = vload_half(0, (__global const half*)(block + 208));

        const int xBlockBase = sb * Q6K_BLOCK_ELEMENTS;

        // Two 128-element halves per super-block.
        for (int hIdx = 0; hIdx < 2; ++hIdx) {
            const int xHalfBase = xBlockBase + hIdx * 128;
            __global const uchar* qlp = ql + hIdx * 64;
            __global const uchar* qhp = qh + hIdx * 32;
            __global const char*  scp = sc + hIdx * 8;

            // 16 threads share the 32-element inner span (4 mads each).
            for (int l = sgLocal; l < 32; l += MATMUL_Q6K_SG) {
                const int is = l / 16;

                const char q1 = (char)((qlp[l +  0] & 0x0F) |
                                       (((qhp[l] >> 0) & 0x03) << 4)) - 32;
                const char q2 = (char)((qlp[l + 32] & 0x0F) |
                                       (((qhp[l] >> 2) & 0x03) << 4)) - 32;
                const char q3 = (char)((qlp[l +  0] >> 4) |
                                       (((qhp[l] >> 4) & 0x03) << 4)) - 32;
                const char q4 = (char)((qlp[l + 32] >> 4) |
                                       (((qhp[l] >> 6) & 0x03) << 4)) - 32;

                const float s0 = d * (float)scp[is + 0];
                const float s2 = d * (float)scp[is + 2];
                const float s4 = d * (float)scp[is + 4];
                const float s6 = d * (float)scp[is + 6];

                acc1 = mad(X[xHalfBase + l +  0], s0 * (float)q1, acc1);
                acc2 = mad(X[xHalfBase + l + 32], s2 * (float)q2, acc2);
                acc3 = mad(X[xHalfBase + l + 64], s4 * (float)q3, acc3);
                acc4 = mad(X[xHalfBase + l + 96], s6 * (float)q4, acc4);
            }
        }
    }

    float sum = sub_group_reduce_add((acc1 + acc2) + (acc3 + acc4));

    if (sgLocal == 0) {
        Y[n] = sum;
    }
}
