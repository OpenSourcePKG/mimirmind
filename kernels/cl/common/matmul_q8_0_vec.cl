// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// SLM-free Q8_0 matvec with widened loads and split accumulators. Same
// transformation as the Q4_K/Q6_K vec kernels: for a matvec X is shared
// by every workgroup and stays cache-hot, so the local-memory X-tile and
// its two per-tile barriers are pure overhead. Each of the 16 subgroup
// lanes owns the contiguous pair {2l, 2l+1} of the 32-element block and
// issues one char2 weight load + one float2 X load; two accumulators
// break the serial FMA chain.
//
//   X:  [K]     F32 dense vector (single token / M=1)
//   W:  [N, K]  Q8_0 — each row is (K/32) blocks of 34 bytes.
//   Y:  [N]     F32 dense vector
//
// Q8_0 block (ggml block_q8_0): fp16 d (2 B) + int8 qs[32] (32 B).
// value[i] = d * qs[i].

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

#ifndef MATMUL_Q8_0_LOCAL
#define MATMUL_Q8_0_LOCAL 64
#endif

#ifndef MATMUL_Q8_0_SG
#define MATMUL_Q8_0_SG 16
#endif

#define MATMUL_Q8_0_OUTPUTS_PER_GROUP (MATMUL_Q8_0_LOCAL / MATMUL_Q8_0_SG)

#define Q8_0_BLOCK_ELEMENTS 32
#define Q8_0_BLOCK_BYTES    34

__attribute__((reqd_work_group_size(MATMUL_Q8_0_LOCAL, 1, 1)))
__attribute__((intel_reqd_sub_group_size(MATMUL_Q8_0_SG)))
__kernel void matmul_q8_0_vec(
    __global const float* X,
    __global const uchar* W,
    __global       float* Y,
    const int             K,
    const int             N)
{
    const int  wg      = (int)get_group_id(0);
    const int  sgInWg  = (int)get_sub_group_id();
    const int  sgLocal = (int)get_sub_group_local_id();
    const int  n       = wg * MATMUL_Q8_0_OUTPUTS_PER_GROUP + sgInWg;
    if (n >= N) return;

    const int  nBlocks = K / Q8_0_BLOCK_ELEMENTS;
    __global const uchar* row =
        W + (size_t)n * (size_t)nBlocks * Q8_0_BLOCK_BYTES;

    const int e2 = 2 * sgLocal;

    float acc0 = 0.0f, acc1 = 0.0f;

    for (int b = 0; b < nBlocks; ++b) {
        __global const uchar* block = row + b * Q8_0_BLOCK_BYTES;
        const float d = vload_half(0, (__global const half*)block);
        __global const char* qs = (__global const char*)(block + 2);

        const int xbase = b * Q8_0_BLOCK_ELEMENTS;

        const char2  qq = vload2(0, qs + e2);
        const float2 xx = vload2(0, X + xbase + e2);

        acc0 = mad(xx.s0, d * (float)qq.s0, acc0);
        acc1 = mad(xx.s1, d * (float)qq.s1, acc1);
    }

    float sum = sub_group_reduce_add(acc0 + acc1);

    if (sgLocal == 0) {
        Y[n] = sum;
    }
}
