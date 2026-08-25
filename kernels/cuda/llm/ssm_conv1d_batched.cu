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
    const int                 T,             // uniform tokens/slot (varlen: max, for grid)
    const int                 channels,
    const int                 K,
    // 5.21-II varlen (nullptr => uniform, bit-identical): per-slot token count +
    // ragged input/output token offsets. inOff = prefix-sum of (seqT[s]+K-1) rows
    // (each slot's input carries its own K-1 conv-tail); outOff = prefix-sum of
    // seqT (= the recurrence's seqOff).
    const int* __restrict__ seqT   = nullptr,
    const int* __restrict__ inOff  = nullptr,
    const int* __restrict__ outOff = nullptr)
{
    const int seq   = blockIdx.y;
    const int Tseq  = (seqT != nullptr) ? seqT[seq] : T;
    const int gid   = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = Tseq * channels;
    if (gid >= total) {
        return;
    }
    const int c = gid % channels;
    const int t = gid / channels;

    const size_t inBase  = (size_t)((inOff  != nullptr) ? inOff[seq]
                                    : seq * (T + K - 1)) * channels;
    const size_t outBase = (size_t)((outOff != nullptr) ? outOff[seq]
                                    : seq * T) * channels;
    const float* in = convInput + inBase;

    float acc = 0.0f;
    for (int kk = 0; kk < K; ++kk) {
        acc += in[(size_t)(t + kk) * channels + c] *
               kern[(size_t)c * K + kk];
    }
    out[outBase + (size_t)gid] = acc / (1.0f + expf(-acc));  // SiLU
}
