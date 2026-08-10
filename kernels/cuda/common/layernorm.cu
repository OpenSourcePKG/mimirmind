// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// LayerNorm (BERT / RoBERTa / XLM-R — the EncoderRunner / cross-encoder
// reranker path):
//
//   mean   = mean(x[m, :])
//   var    = mean( (x[m, :] - mean)^2 )                  (biased, two-pass)
//   y[m,k] = (x[m,k] - mean) / sqrt(var + eps) * weight[k] + bias[k]
//
// Mirrors rmsnorm.cu's launch shape (one workgroup per row m) and its
// power-of-two shared-memory tree reduction. Two passes so the variance is
// computed from (x-mean)^2 — matches PyTorch/HF LayerNorm numerics exactly
// (the cross-encoder parity gate needs that, unlike the E[x^2]-E[x]^2 form).
//
// Launch:
//   dim3 grid (M, 1, 1), dim3 block(LAYERNORM_LOCAL_SIZE, 1, 1)
// Overridable via -D LAYERNORM_LOCAL_SIZE=<n>; power of two, multiple of 32.

#include <cuda_runtime.h>

#ifndef LAYERNORM_LOCAL_SIZE
#define LAYERNORM_LOCAL_SIZE 128
#endif

extern "C" __global__ __launch_bounds__(LAYERNORM_LOCAL_SIZE)
void layernorm(
    const float* __restrict__ x,       // [M, K]
    const float* __restrict__ weight,  // [K]  (gamma)
    const float* __restrict__ bias,    // [K]  (beta)
          float* __restrict__ y,       // [M, K]
    const float               eps,
    const int                 K)
{
    __shared__ float scratch[LAYERNORM_LOCAL_SIZE];

    const int m     = blockIdx.x;
    const int tid   = threadIdx.x;
    const int lsize = blockDim.x;

    const float* __restrict__ xr = x + static_cast<size_t>(m) * static_cast<size_t>(K);
          float* __restrict__ yr = y + static_cast<size_t>(m) * static_cast<size_t>(K);

    // --- Pass 1: mean ---
    float acc = 0.0f;
    for (int k = tid; k < K; k += lsize) {
        acc += xr[k];
    }
    scratch[tid] = acc;
    __syncthreads();
    for (int stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }
    const float mean = scratch[0] / static_cast<float>(K);
    __syncthreads();  // all threads read scratch[0] before it is overwritten

    // --- Pass 2: variance from (x - mean)^2 ---
    float vacc = 0.0f;
    for (int k = tid; k < K; k += lsize) {
        const float d = xr[k] - mean;
        vacc = __fmaf_rn(d, d, vacc);
    }
    scratch[tid] = vacc;
    __syncthreads();
    for (int stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }
    const float var    = scratch[0] / static_cast<float>(K);
    const float invStd = rsqrtf(var + eps);

    // --- Apply: (x - mean) * invStd * gamma + beta ---
    for (int k = tid; k < K; k += lsize) {
        yr[k] = (xr[k] - mean) * invStd * weight[k] + bias[k];
    }
}
