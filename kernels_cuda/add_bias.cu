// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
// Ported from kernels_hip/add_bias.hip — Track 4 mechanical port, no functional change intended.

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// In-place broadcast bias add:
//   y[m, k] += bias[k]   for m in [0, M), k in [0, K)
//
// HIP port of kernels/add_bias.cl. Row-major y of shape (M, K), bias of
// length K broadcast across all rows. Flat 1D thread index; per-thread
// modulo picks the bias slot.
//
// Launch:
//   dim3 grid ( ceil(M*K / ADD_BIAS_LOCAL), 1, 1 )
//   dim3 block( ADD_BIAS_LOCAL, 1, 1 )

#include <cuda_runtime.h>

#ifndef ADD_BIAS_LOCAL
#define ADD_BIAS_LOCAL 256
#endif

extern "C" __global__ __launch_bounds__(ADD_BIAS_LOCAL)
void add_bias(
          float* __restrict__ y,      // in + out, shape (M, K)
    const float* __restrict__ bias,   // shape (K,)
    const int                 M,
    const int                 K)
{
    const int gid   = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = M * K;
    if (gid >= total) {
        return;
    }
    const int k = gid % K;
    y[gid] += bias[k];
}