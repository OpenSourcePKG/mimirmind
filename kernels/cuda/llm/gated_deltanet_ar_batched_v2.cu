// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Latency-optimised variant of gated_deltanet_ar_batched — BYTE-IDENTICAL math.
//
// The v1 kernel is severely memory-latency bound (ncu 2026-07-30: 98.9% of
// cycles have no eligible warp, ~937-cycle scoreboard stalls on the global
// state loads, occupancy already 87%). Its first inner loop does a
// read-modify-WRITE of every state element to global (s = s*g), which the
// second loop then reads back — the store serialises the load pipeline and
// costs a full extra state read+write pass.
//
// This variant:
//   loop 1: pure READS — sk += (s*g)*k   (no state write)
//   loop 2: recompute s*g, write the final state ONCE, accumulate the output
// so the independent state loads can be pipelined (hiding their latency) and
// the total global state traffic drops from 4 passes (R+W, R+W) to 3 (R, R+W).
// The sk / oj accumulation order is unchanged and s*g is recomputed to the same
// bits, so the result is bit-identical to v1. Same launch geometry.
//
// Launch: grid = dim3(H, nSeq, 1), block = S threads.

#include <cuda_runtime.h>

#ifndef GATED_DELTANET_AR_MAX_S
#define GATED_DELTANET_AR_MAX_S 256
#endif

extern "C" __global__ __launch_bounds__(GATED_DELTANET_AR_MAX_S)
void gated_deltanet_ar_batched_v2(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ gLog,
    const float* __restrict__ beta,
    float*       __restrict__ state,
    float*       __restrict__ out,
    const int                 T,
    const int                 H,
    const int                 S,
    const unsigned char* __restrict__ activeMask)   // 5.21-I: nullptr => all active
{
    const int seq = blockIdx.y;
    if (activeMask != nullptr && activeMask[seq] == 0) return;   // 5.21-I: freeze
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

        // Pass 1: pure reads. sk = sum_i (s[i,j]*g) * k[i]  (order preserved).
        float sk = 0.0f;
#pragma unroll 4
        for (int i = 0; i < S; ++i) {
            sk += (s[(size_t)i * S + j] * g) * ksh[i];
        }
        const float dj = (vj - sk) * b;

        // Pass 2: recompute the decayed state, write it once, accumulate out.
        float oj = 0.0f;
#pragma unroll 4
        for (int i = 0; i < S; ++i) {
            const float sij = s[(size_t)i * S + j] * g + ksh[i] * dj;
            s[(size_t)i * S + j] = sij;
            oj += sij * qsh[i];
        }
        out[base + j] = oj;

        __syncthreads();
    }
}
