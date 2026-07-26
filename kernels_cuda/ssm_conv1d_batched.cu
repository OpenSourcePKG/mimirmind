// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Batched causal depthwise 1-D convolution + SiLU — M-Cuda.Batch
// batched variant of ssm_conv1d (Qwen3-Next GatedDeltaNet). Processes
// nSeq independent sequences, each with its own conv input (the caller
// prepends that sequence's rolling conv-tail), in ONE launch. Math per
// (seq, t, channel) is byte-identical to the single-sequence kernel;
// only a per-sequence offset (blockIdx.y) is added. Cat C-P0 of the
// hybrid batch-dim audit 2026-07-24 (per-sequence conv-tail state).
//
// Layout (per-sequence strides derive from T,channels,K — no new param):
//   convInput : [nSeq, (K-1+T), channels]  seqStride = (K-1+T)*channels
//   out       : [nSeq, T, channels]        seqStride = T*channels
//   kern      : [channels, K]              shared across sequences
// Launch: grid = dim3(ceil(T*channels / LOCAL), nSeq, 1), block = LOCAL.

#include <cuda_runtime.h>

#ifndef SSM_CONV1D_LOCAL
#define SSM_CONV1D_LOCAL 256
#endif

extern "C" __global__ __launch_bounds__(SSM_CONV1D_LOCAL)
void ssm_conv1d_batched(
    const float* __restrict__ convInput,
    const float* __restrict__ kern,
    float*       __restrict__ out,
    const int                 T,
    const int                 channels,
    const int                 K)
{
    const int seq   = blockIdx.y;
    const int gid   = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = T * channels;
    if (gid >= total) {
        return;
    }
    const int c = gid % channels;
    const int t = gid / channels;

    const size_t inSeqStride  = (size_t)(T + K - 1) * channels;
    const size_t outSeqStride = (size_t)T * channels;
    const float* in = convInput + (size_t)seq * inSeqStride;

    float acc = 0.0f;
    for (int kk = 0; kk < K; ++kk) {
        acc += in[(size_t)(t + kk) * channels + c] *
               kern[(size_t)c * K + kk];
    }
    out[(size_t)seq * outSeqStride + gid] = acc / (1.0f + expf(-acc));  // SiLU
}
