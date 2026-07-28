// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// In-place scalar multiply: y[i] *= s   for i in [0, n).
//
// HIP port of kernels/mul_scalar.cl. Used by Gemma 4 for the per-block
// layer_output_scale (one F32 scalar per block that multiplies the
// block's final output). The scalar arrives as a kernel value argument,
// not via a device pointer — first HIP kernel in the port list to bind
// a plain-float POD via HipKernel::setValue<float>.
//
// Launch:
//   dim3 grid ( ceil(n / MUL_SCALAR_LOCAL), 1, 1 )
//   dim3 block( MUL_SCALAR_LOCAL, 1, 1 )

#include <hip/hip_runtime.h>

#ifndef MUL_SCALAR_LOCAL
#define MUL_SCALAR_LOCAL 256
#endif

extern "C" __global__ __launch_bounds__(MUL_SCALAR_LOCAL)
void mul_scalar(
    float*    __restrict__ y,   // in + out
    const float            s,
    const int              n)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) {
        return;
    }
    y[gid] *= s;
}