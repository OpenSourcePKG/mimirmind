// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

// Causal depthwise 1-D convolution + SiLU (Qwen3-Next `ssm_conv1d`).
//   out[t,c] = silu( sum_{kk=0..K-1} convInput[t+kk, c] * kern[kk, c] )
// convInput : [(K-1)+T, channels] row-major (state-prepended input)
// kern      : [K, channels]       tap-major  (GGUF ssm_conv1d.weight layout)
// out       : [T, channels]
// CPU reference: compute::causalConv1dSilu.
//
// Launch: 1D global = T*channels work-items (one per output element).

#ifndef SSM_CONV1D_LOCAL
#define SSM_CONV1D_LOCAL 256
#endif

__attribute__((reqd_work_group_size(SSM_CONV1D_LOCAL, 1, 1)))
__kernel void ssm_conv1d(
    __global const float* convInput,
    __global const float* kern,
    __global       float* out,
    const int             T,
    const int             channels,
    const int             K)
{
    const int gid   = (int)get_global_id(0);
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
    out[gid] = acc / (1.0f + exp(-acc));   // SiLU
}