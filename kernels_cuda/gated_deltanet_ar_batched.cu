// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Batched autoregressive gated delta-rule recurrence — M-Cuda.Batch
// batched-state variant of gated_deltanet_ar. Processes nSeq independent
// sequences, each with its own [H,S,S] recurrent state, in ONE launch.
// The math per (seq,head) is byte-identical to the single-sequence
// kernel; only a per-sequence offset (blockIdx.y) is added to every
// pointer. This is the first M-Cuda.Batch batched-state kernel (Cat C-P0
// of the hybrid batch-dim audit 2026-07-24).
//
// Layout (per-sequence strides derive from T,H,S — no extra param):
//   q,k,v,out : [nSeq,T,H,S]  seqStride = T*H*S
//   gLog,beta : [nSeq,T,H]    seqStride = T*H
//   state     : [nSeq,H,S,S]  seqStride = H*S*S   s[seq][h][i*S+j]
// Launch: grid = dim3(H, nSeq, 1), block = S threads.

#include <cuda_runtime.h>

#ifndef GATED_DELTANET_AR_MAX_S
#define GATED_DELTANET_AR_MAX_S 256
#endif

extern "C" __global__ __launch_bounds__(GATED_DELTANET_AR_MAX_S)
void gated_deltanet_ar_batched(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ gLog,
    const float* __restrict__ beta,
    float*       __restrict__ state,
    float*       __restrict__ out,
    const int                 T,
    const int                 H,
    const int                 S)
{
    const int seq = blockIdx.y;
    const int h   = blockIdx.x;
    const int j   = threadIdx.x;   // state column; block size == S

    __shared__ float ksh[GATED_DELTANET_AR_MAX_S];
    __shared__ float qsh[GATED_DELTANET_AR_MAX_S];

    const size_t actSeqStride   = (size_t)T * H * S;   // q,k,v,out
    const size_t gateSeqStride  = (size_t)T * H;       // gLog,beta
    const size_t stateSeqStride = (size_t)H * S * S;   // state

    float* s = state + (size_t)seq * stateSeqStride + (size_t)h * S * S;
    const float qScale = 1.0f / sqrtf((float)S);

    for (int t = 0; t < T; ++t) {
        const size_t base    = (size_t)seq * actSeqStride
                             + (size_t)(t * H + h) * S;
        const size_t gateIdx = (size_t)seq * gateSeqStride
                             + (size_t)(t * H + h);

        ksh[j] = k[base + j];
        qsh[j] = q[base + j] * qScale;
        __syncthreads();

        const float g  = expf(gLog[gateIdx]);
        const float b  = beta[gateIdx];
        const float vj = v[base + j];

        float sk = 0.0f;
        for (int i = 0; i < S; ++i) {
            const float sij = s[(size_t)i * S + j] * g;
            s[(size_t)i * S + j] = sij;
            sk += sij * ksh[i];
        }
        const float dj = (vj - sk) * b;

        float oj = 0.0f;
        for (int i = 0; i < S; ++i) {
            const float sij = s[(size_t)i * S + j] + ksh[i] * dj;
            s[(size_t)i * S + j] = sij;
            oj += sij * qsh[i];
        }
        out[base + j] = oj;

        __syncthreads();
    }
}
