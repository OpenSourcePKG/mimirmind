// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
// Ported from kernels_hip/rmsnorm.hip — Track 4 mechanical port, no functional change intended.

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// RMSNorm: y[m, k] = x[m, k] * weight[k] / sqrt(mean(x[m, :]^2) + eps)
//
// First real HIP kernel port. 1:1 algorithmic mirror of
// kernels/rmsnorm.cl (the L0 / OpenCL reference) — same launch shape
// (one workgroup per row m), same power-of-two tree-reduction in LDS,
// same per-thread strided pass for both the sum-of-squares and the
// scaling step. Not tuned for RDNA3 yet (a WAVE32 shuffle-based tail
// reduction would kill the last few __syncthreads(), but we start
// with the parity port before we start optimising).
//
// Launch:
//   dim3 grid (M, 1, 1),
//   dim3 block(RMSNORM_LOCAL_SIZE, 1, 1)   // 128 == 4 waves on gfx1101
//
// Overridable at compile time via -D RMSNORM_LOCAL_SIZE=<n>. Must be a
// power of two for the tree reduction to bottom out cleanly, and a
// multiple of 32 (warpSize on RDNA/RDNA2/RDNA3) to avoid half-warp
// waste. Default 128 matches the L0-side default.

#include <cuda_runtime.h>

#ifndef RMSNORM_LOCAL_SIZE
#define RMSNORM_LOCAL_SIZE 128
#endif

extern "C" __global__ __launch_bounds__(RMSNORM_LOCAL_SIZE)
void rmsnorm(
    const float* __restrict__ x,       // [M, K]
    const float* __restrict__ weight,  // [K]
          float* __restrict__ y,       // [M, K]
    const float               eps,
    const int                 K)
{
    __shared__ float scratch[RMSNORM_LOCAL_SIZE];

    const int m     = blockIdx.x;
    const int tid   = threadIdx.x;
    const int lsize = blockDim.x;

    const float* __restrict__ xr = x + static_cast<size_t>(m) * static_cast<size_t>(K);
          float* __restrict__ yr = y + static_cast<size_t>(m) * static_cast<size_t>(K);

    // Per-thread partial sum of squares.
    float acc = 0.0f;
    for (int k = tid; k < K; k += lsize) {
        const float v = xr[k];
        acc = __fmaf_rn(v, v, acc);
    }
    scratch[tid] = acc;
    __syncthreads();

    // Power-of-two tree reduction in shared memory. RDNA3 could do the
    // last log2(32)=5 rounds via __shfl_xor_sync without __syncthreads,
    // but this is the parity port — perf pass comes as its own commit.
    for (int stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }

    const float mean   = scratch[0] / static_cast<float>(K);
    const float invRms = rsqrtf(mean + eps);

    // Apply scale.
    for (int k = tid; k < K; k += lsize) {
        yr[k] = xr[k] * weight[k] * invRms;
    }
}