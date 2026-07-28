// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// In-place logistic sigmoid. CUDA port of kernels/sigmoid_inplace.cl.
// CPU reference: compute::sigmoidInPlace. Launch: grid ceil(n/L), block L.

#include <cuda_runtime.h>

#ifndef SIGMOID_INPLACE_LOCAL
#define SIGMOID_INPLACE_LOCAL 256
#endif

extern "C" __global__ __launch_bounds__(SIGMOID_INPLACE_LOCAL)
void sigmoid_inplace(
    float*    __restrict__ y,
    const int              n)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) {
        return;
    }
    y[gid] = 1.0f / (1.0f + expf(-y[gid]));
}
