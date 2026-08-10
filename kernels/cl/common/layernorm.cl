// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

// LayerNorm (BERT/RoBERTa/XLM-R — the cross-encoder reranker / EncoderRunner):
//   mean   = mean(x[m, :])
//   var    = mean((x[m, :] - mean)^2)              (biased, two-pass)
//   y[m,k] = (x[m,k] - mean) / sqrt(var + eps) * weight[k] + bias[k]
//
// Launch: one workgroup per row m (global = M * LOCAL_SIZE, local = LOCAL_SIZE).
// Two tree reductions (sum, then sum of squared deviations) to match
// PyTorch/HF numerics for the parity gate. CPU reference: compute::layerNorm.

#ifndef LAYERNORM_LOCAL_SIZE
#define LAYERNORM_LOCAL_SIZE 128
#endif

__attribute__((reqd_work_group_size(LAYERNORM_LOCAL_SIZE, 1, 1)))
__kernel void layernorm(
    __global const float* x,       // [M, K]
    __global const float* weight,  // [K] gamma
    __global const float* bias,    // [K] beta
    __global       float* y,       // [M, K]
    const float           eps,
    const int             K)
{
    __local float scratch[LAYERNORM_LOCAL_SIZE];

    const int m     = (int)get_group_id(0);
    const int tid   = (int)get_local_id(0);
    const int lsize = (int)get_local_size(0);

    __global const float* xr = x + (size_t)m * (size_t)K;
    __global       float* yr = y + (size_t)m * (size_t)K;

    // Pass 1: mean.
    float acc = 0.0f;
    for (int k = tid; k < K; k += lsize) {
        acc += xr[k];
    }
    scratch[tid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float mean = scratch[0] / (float)K;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Pass 2: variance (mean of squared deviations).
    float vacc = 0.0f;
    for (int k = tid; k < K; k += lsize) {
        const float d = xr[k] - mean;
        vacc = mad(d, d, vacc);
    }
    scratch[tid] = vacc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float var    = scratch[0] / (float)K;
    const float invStd = rsqrt(var + eps);

    for (int k = tid; k < K; k += lsize) {
        yr[k] = (xr[k] - mean) * invStd * weight[k] + bias[k];
    }
}
