// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// In-place tanh — the activation in the RoBERTa/XLM-R sequence-classification
// head (dense -> tanh -> out_proj) used by the cross-encoder reranker
// (EncoderRunner). One thread per element, no reduction.
//
// Launch: grid( ceil(n / TANH_LOCAL) ), block( TANH_LOCAL ).

#include <cuda_runtime.h>

#ifndef TANH_LOCAL
#define TANH_LOCAL 256
#endif

extern "C" __global__ __launch_bounds__(TANH_LOCAL)
void tanh_inplace(float* __restrict__ x, const int n)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) {
        return;
    }
    x[gid] = tanhf(x[gid]);
}
