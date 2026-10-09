// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// In-place ReLU: x[i] = max(x[i], 0). The FFN activation of the standard torch
// nn.TransformerEncoderLayer (the Laya decision head's 2 head layers use the
// default relu activation). One thread per element.
//
// Launch:
//   dim3 grid ( ceil(n / RELU_LOCAL), 1, 1 )
//   dim3 block( RELU_LOCAL, 1, 1 )

#include <cuda_runtime.h>

#ifndef RELU_LOCAL
#define RELU_LOCAL 256
#endif

extern "C" __global__ __launch_bounds__(RELU_LOCAL)
void relu(float* __restrict__ x, const int n)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) {
        return;
    }
    x[gid] = fmaxf(x[gid], 0.0f);
}
