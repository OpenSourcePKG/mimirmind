// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
// Ported from kernels_hip/rmsnorm_no_weight.hip — Track 4 mechanical port, no functional change intended.

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// RMSNorm without a per-element weight: y = x * invRms.
//
// HIP port of kernels/rmsnorm_no_weight.cl. Used for Gemma 4 V-norm —
// the V projection passes through a bare `ggml_rms_norm` (no learned
// weight) before going into the KV cache. Same tree-reduction body as
// rmsnorm.hip / rmsnorm_gemma.hip; the final write loop just skips
// the multiplicative weight.
//
// Launch:
//   dim3 grid (M, 1, 1),
//   dim3 block(RMSNORM_NW_LOCAL_SIZE, 1, 1)

#include <cuda_runtime.h>

#ifndef RMSNORM_NW_LOCAL_SIZE
#define RMSNORM_NW_LOCAL_SIZE 128
#endif

extern "C" __global__ __launch_bounds__(RMSNORM_NW_LOCAL_SIZE)
void rmsnorm_no_weight(
    const float* __restrict__ x,   // [M, K]
          float* __restrict__ y,   // [M, K]
    const float               eps,
    const int                 K)
{
    __shared__ float scratch[RMSNORM_NW_LOCAL_SIZE];

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

    for (int k = tid; k < K; k += lsize) {
        yr[k] = xr[k] * invRms;
    }
}