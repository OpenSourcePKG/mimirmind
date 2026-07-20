// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
// Ported from kernels_hip/add_rmsnorm.hip — Track 4 mechanical port, no functional change intended.

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Fused residual-add + RMSNorm.
//   x[m, k] += delta[m, k]     (in-place)
//   y[m, k] = x[m, k] * weight[k] / sqrt(mean(x[m, :]^2) + eps)
//
// HIP port of kernels/add_rmsnorm.cl. Same tree-reduction shape as
// rmsnorm.hip — one workgroup per row m, phase-1 does the in-place
// add plus per-thread partial sum-of-squares in a single pass, phase-2
// applies (weight * invRms) after the LDS reduction. `y` may alias
// `x` (Gemma/Qwen backends do; the write happens after all reads for
// the current thread's stride so aliasing is safe).
//
// Launch:
//   dim3 grid (M, 1, 1),
//   dim3 block(ADD_RMSNORM_LOCAL_SIZE, 1, 1)   // 128 == 4 waves on gfx1101
//
// Overridable at compile time via -D ADD_RMSNORM_LOCAL_SIZE=<n>. Must
// be a power of two + multiple of 32 (warpSize on RDNA/RDNA2/RDNA3).

#include <cuda_runtime.h>

#ifndef ADD_RMSNORM_LOCAL_SIZE
#define ADD_RMSNORM_LOCAL_SIZE 128
#endif

extern "C" __global__ __launch_bounds__(ADD_RMSNORM_LOCAL_SIZE)
void add_rmsnorm(
          float* __restrict__ x,       // [M, K] in-place accumulator
    const float* __restrict__ delta,   // [M, K]
    const float* __restrict__ weight,  // [K]
          float* __restrict__ y,       // [M, K] (may alias x)
    const float               eps,
    const int                 K)
{
    __shared__ float scratch[ADD_RMSNORM_LOCAL_SIZE];

    const int m     = blockIdx.x;
    const int tid   = threadIdx.x;
    const int lsize = blockDim.x;

          float* __restrict__ xr = x     + static_cast<size_t>(m) * static_cast<size_t>(K);
    const float* __restrict__ dr = delta + static_cast<size_t>(m) * static_cast<size_t>(K);
          float* __restrict__ yr = y     + static_cast<size_t>(m) * static_cast<size_t>(K);

    // Phase 1: fused add-in-place + per-thread partial sum-of-squares.
    // Each thread walks its stride so the loop over K happens exactly
    // once. `__fmaf_rn` mirrors the L0 kernel's `mad()` — the compiler
    // is free to fuse but the intent is explicit.
    float acc = 0.0f;
    for (int k = tid; k < K; k += lsize) {
        const float v = xr[k] + dr[k];
        xr[k] = v;
        acc = __fmaf_rn(v, v, acc);
    }
    scratch[tid] = acc;
    __syncthreads();

    // Power-of-two tree reduction over sum-of-squares. Same shape as
    // rmsnorm.hip — a WAVE32 shuffle-based tail is a perf pass follow-up
    // once bit-parity is confirmed.
    for (int stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }

    const float mean   = scratch[0] / static_cast<float>(K);
    const float invRms = rsqrtf(mean + eps);

    // Phase 2: apply weight * invRms.
    for (int k = tid; k < K; k += lsize) {
        yr[k] = xr[k] * weight[k] * invRms;
    }
}