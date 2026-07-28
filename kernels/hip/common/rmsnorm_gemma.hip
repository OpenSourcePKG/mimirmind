// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Gemma-style RMSNorm: y = x * (1 + weight) * invRms
//
// HIP port of kernels/rmsnorm_gemma.cl. The Gemma family initialises
// norm weights at 0 and uses `(1 + weight)` rather than `weight`, so
// the norm starts as identity. Algorithmically identical to
// rmsnorm.hip otherwise — same tree reduction, same mean / invRms /
// write loop.
//
// Used for all proper norm weights in Gemma 4 (attn_norm,
// post_attention_norm, ffn_norm, pre/post_ffw_norm_1/2, post_ffw_norm,
// attn_q_norm, attn_k_norm, output_norm). The router input still uses
// the regular rmsnorm.hip.
//
// Launch:
//   dim3 grid (M, 1, 1),
//   dim3 block(RMSNORM_GEMMA_LOCAL_SIZE, 1, 1)

#include <hip/hip_runtime.h>

#ifndef RMSNORM_GEMMA_LOCAL_SIZE
#define RMSNORM_GEMMA_LOCAL_SIZE 128
#endif

extern "C" __global__ __launch_bounds__(RMSNORM_GEMMA_LOCAL_SIZE)
void rmsnorm_gemma(
    const float* __restrict__ x,       // [M, K]
    const float* __restrict__ weight,  // [K]
          float* __restrict__ y,       // [M, K]
    const float               eps,
    const int                 K)
{
    __shared__ float scratch[RMSNORM_GEMMA_LOCAL_SIZE];

    const int m     = blockIdx.x;
    const int tid   = threadIdx.x;
    const int lsize = blockDim.x;

    const float* __restrict__ xr = x + static_cast<size_t>(m) * static_cast<size_t>(K);
          float* __restrict__ yr = y + static_cast<size_t>(m) * static_cast<size_t>(K);

    float acc = 0.0f;
    for (int k = tid; k < K; k += lsize) {
        const float v = xr[k];
        acc = __fmaf_rn(v, v, acc);
    }
    scratch[tid] = acc;
    __syncthreads();

    for (int stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }

    const float mean   = scratch[0] / static_cast<float>(K);
    const float invRms = rsqrtf(mean + eps);

    // Apply (1 + weight) per element — the Gemma convention.
    for (int k = tid; k < K; k += lsize) {
        yr[k] = xr[k] * (1.0f + weight[k]) * invRms;
    }
}