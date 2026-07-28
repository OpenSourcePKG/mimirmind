// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

// RMSNorm without a per-element weight: y = x * invRms.
//
// Used for Gemma 4 V-norm — V projection passes through a bare
// `ggml_rms_norm` (no learned weight) before going into the KV cache.
// Same tree-reduction body as rmsnorm.cl / rmsnorm_gemma.cl; the
// final write loop just skips the multiplicative weight.

#ifndef RMSNORM_NW_LOCAL_SIZE
#define RMSNORM_NW_LOCAL_SIZE 128
#endif

__attribute__((reqd_work_group_size(RMSNORM_NW_LOCAL_SIZE, 1, 1)))
__kernel void rmsnorm_no_weight(
    __global const float* x,       // [M, K]
    __global       float* y,       // [M, K]
    const float           eps,
    const int             K)
{
    __local float scratch[RMSNORM_NW_LOCAL_SIZE];

    const int m         = (int)get_group_id(0);
    const int tid       = (int)get_local_id(0);
    const int lsize     = (int)get_local_size(0);

    __global const float* xr = x + (size_t)m * (size_t)K;
    __global       float* yr = y + (size_t)m * (size_t)K;

    float acc = 0.0f;
    for (int k = tid; k < K; k += lsize) {
        const float v = xr[k];
        acc = mad(v, v, acc);
    }
    scratch[tid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const float mean    = scratch[0] / (float)K;
    const float invRms  = rsqrt(mean + eps);

    for (int k = tid; k < K; k += lsize) {
        yr[k] = xr[k] * invRms;
    }
}