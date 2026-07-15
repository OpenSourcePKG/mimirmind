// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// In-place 1D add: y[i] += x[i] for i in [0, n).
//
// HIP port of kernels/add_residual.cl. Post-attention and post-FFN
// residual stream — every transformer block calls this twice.
//
// Launch:
//   dim3 grid ( ceil(n / ADD_RESIDUAL_LOCAL), 1, 1 )
//   dim3 block( ADD_RESIDUAL_LOCAL, 1, 1 )

#include <hip/hip_runtime.h>

#ifndef ADD_RESIDUAL_LOCAL
#define ADD_RESIDUAL_LOCAL 256
#endif

extern "C" __global__ __launch_bounds__(ADD_RESIDUAL_LOCAL)
void add_residual(
          float* __restrict__ y,   // in + out
    const float* __restrict__ x,
    const int                 n)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) {
        return;
    }
    y[gid] += x[gid];
}