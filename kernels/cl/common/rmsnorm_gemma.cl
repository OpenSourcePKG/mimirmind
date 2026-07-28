// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

// Gemma-style RMSNorm: y = x * (1 + weight) * invRms
//
// The Gemma family of models initialises norm weights at 0 and uses
// `(1 + weight)` rather than `weight` so the norm starts as identity.
// Algorithmically identical to rmsnorm.cl otherwise — same tree
// reduction, same mean / invRms / write loop.
//
// Used for all proper norm weights in gemma4 (attn_norm,
// post_attention_norm, ffn_norm, pre/post_ffw_norm_1/2, post_ffw_norm,
// attn_q_norm, attn_k_norm, output_norm). The router input still uses
// the regular rmsnorm.cl because there the per-element multiplier is
// `ffn_gate_inp.scale` applied without the +1 shift.

#ifndef RMSNORM_GEMMA_LOCAL_SIZE
#define RMSNORM_GEMMA_LOCAL_SIZE 128
#endif

__attribute__((reqd_work_group_size(RMSNORM_GEMMA_LOCAL_SIZE, 1, 1)))
__kernel void rmsnorm_gemma(
    __global const float* x,       // [M, K]
    __global const float* weight,  // [K]
    __global       float* y,       // [M, K]
    const float           eps,
    const int             K)
{
    __local float scratch[RMSNORM_GEMMA_LOCAL_SIZE];

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

    // Apply (1 + weight) per element — the Gemma convention.
    for (int k = tid; k < K; k += lsize) {
        yr[k] = xr[k] * (1.0f + weight[k]) * invRms;
    }
}