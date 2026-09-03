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

// 5.18.10.2 — batched conv-input PACK. Replaces the host loop of 2*nSeq tiny
// cudaMemcpyAsync per layer (state->convInput + tokens->convInput): the
// 2026-09-03 decode-gap decomposition measured gdn.conv at 18.1 ms/step at
// conc64 (~16x above its ~1 ms traffic floor) — ~7000 micro-copies per step
// across 36 GDN layers made the section launch-bound. One launch builds every
// slot's [conv-tail (K-1 rows) | Tslot token rows] block. Pure copies =
// bit-identical to the memcpy loop. Frozen slots are packed exactly like the
// loop did (the copies always ran; only the SAVE is masked).
//   convState : [nSeq, (K-1)*channels]  per-slot rolling conv tail
//   qkvMixed  : token rows [*, channels], slot seq's tokens at tokOff(seq)
//   convInput : [rows, channels], slot seq's block at inOff(seq) rows
// Decode (seqT==nullptr): Tslot=1, inOff=seq*K, tokOff=seq (pre-varlen layout).
// Launch: grid = dim3(ceil(maxRows*channels / LOCAL), nSeq, 1), block = LOCAL,
//         maxRows = (K-1) + T.
extern "C" __global__ __launch_bounds__(SSM_CONV1D_LOCAL)
void gdn_conv_pack_batched(
    const float* __restrict__ convState,
    const float* __restrict__ qkvMixed,
    float*       __restrict__ convInput,
    const int                 T,             // uniform tokens/slot (varlen: max)
    const int                 channels,
    const int                 K,
    const int* __restrict__ seqT,            // nullptr => uniform T
    const int* __restrict__ inOff,           // nullptr => seq * (T + K - 1)
    const int* __restrict__ tokOff)          // nullptr => seq * T
{
    const int seq       = blockIdx.y;
    const int stateRows = K - 1;
    const int Tseq      = (seqT != nullptr) ? seqT[seq] : T;
    const int rows      = stateRows + Tseq;
    const int gid       = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= rows * channels) {
        return;
    }
    const int r = gid / channels;
    const int c = gid % channels;

    const size_t inBase = (size_t)((inOff != nullptr) ? inOff[seq]
                                   : seq * (T + K - 1)) * channels;
    float v;
    if (r < stateRows) {
        v = convState[((size_t)seq * stateRows + r) * channels + c];
    } else {
        const size_t tokBase = (size_t)((tokOff != nullptr) ? tokOff[seq]
                                        : seq * T) + (r - stateRows);
        v = qkvMixed[tokBase * channels + c];
    }
    convInput[inBase + (size_t)r * channels + c] = v;
}

// 5.18.10.3 — batched row split: in[r, 0..wa) -> a[r], in[r, wa..wa+wb) -> b[r]
// for all rows in ONE launch. Un-interleaves the output of a fused projection
// GEMM ([row, wa+wb]) back into the two compact consumer buffers, so every
// downstream kernel (conv pack, siluMul, recurrence) stays unchanged. Pure
// copies. Launch: grid = dim3(ceil((wa+wb) / LOCAL), rows, 1), block = LOCAL.
extern "C" __global__ __launch_bounds__(SSM_CONV1D_LOCAL)
void gdn_row_split2(
    const float* __restrict__ in,
    float*       __restrict__ a,
    float*       __restrict__ b,
    const int                 wa,
    const int                 wb)
{
    const int r   = blockIdx.y;
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int w   = wa + wb;
    if (gid >= w) {
        return;
    }
    const float v = in[(size_t)r * w + gid];
    if (gid < wa) {
        a[(size_t)r * wa + gid] = v;
    } else {
        b[(size_t)r * wb + (gid - wa)] = v;
    }
}

// 5.18.10.2 — batched conv-tail SAVE (the third memcpy of the host loop):
// slot seq's next rolling conv tail = the LAST (K-1) rows of its packed block,
// i.e. rows [Tslot, Tslot + K - 1). Frozen slots (activeMask[seq]==0) keep
// their tail byte-identical (skip), exactly like the loop's `continue`.
// Launch: grid = dim3(ceil((K-1)*channels / LOCAL), nSeq, 1), block = LOCAL.
extern "C" __global__ __launch_bounds__(SSM_CONV1D_LOCAL)
void gdn_conv_save_batched(
    const float* __restrict__ convInput,
    float*       __restrict__ convState,
    const int                 T,             // uniform tokens/slot (varlen: max)
    const int                 channels,
    const int                 K,
    const unsigned char* __restrict__ activeMask,   // nullptr => all active
    const int* __restrict__ seqT,            // nullptr => uniform T
    const int* __restrict__ inOff)           // nullptr => seq * (T + K - 1)
{
    const int seq = blockIdx.y;
    if (activeMask != nullptr && activeMask[seq] == 0) {
        return;                              // frozen: tail stays byte-identical
    }
    const int stateRows = K - 1;
    const int Tseq      = (seqT != nullptr) ? seqT[seq] : T;
    const int gid       = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= stateRows * channels) {
        return;
    }
    const int r = gid / channels;
    const int c = gid % channels;
    const size_t inBase = (size_t)((inOff != nullptr) ? inOff[seq]
                                   : seq * (T + K - 1)) * channels;
    convState[((size_t)seq * stateRows + r) * channels + c] =
        convInput[inBase + (size_t)(Tseq + r) * channels + c];
}
