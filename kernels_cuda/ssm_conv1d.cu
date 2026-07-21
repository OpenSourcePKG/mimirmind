// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Causal depthwise 1-D convolution + SiLU (Qwen3-Next `ssm_conv1d`). CUDA
// port of kernels/ssm_conv1d.cl — see that file for the layout contract.
// CPU reference: compute::causalConv1dSilu.
//
// Launch: grid ceil(T*channels / SSM_CONV1D_LOCAL), block SSM_CONV1D_LOCAL.

#include <cuda_runtime.h>

#ifndef SSM_CONV1D_LOCAL
#define SSM_CONV1D_LOCAL 256
#endif

extern "C" __global__ __launch_bounds__(SSM_CONV1D_LOCAL)
void ssm_conv1d(
    const float* __restrict__ convInput,
    const float* __restrict__ kern,
    float*       __restrict__ out,
    const int                 T,
    const int                 channels,
    const int                 K)
{
    const int gid   = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = T * channels;
    if (gid >= total) {
        return;
    }
    const int c = gid % channels;
    const int t = gid / channels;

    float acc = 0.0f;
    for (int kk = 0; kk < K; ++kk) {
        acc += convInput[(size_t)(t + kk) * channels + c] *
               kern[(size_t)kk * channels + c];
    }
    out[gid] = acc / (1.0f + expf(-acc));   // SiLU
}