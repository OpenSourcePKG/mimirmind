// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Autoregressive gated delta-rule recurrence (Qwen3-Next linear attention).
// HIP port of kernels/gated_deltanet_ar.cl — see that file for the full
// per-token recurrence and the column-parallel scheme. CPU reference:
// compute::gatedDeltaNetRecurrent (golden truth).
//
// Layout: q,k,v [T,H,S]; gLog,beta [T,H]; state [H,S,S] s[h][i*S+j];
// out [T,H,S]. Launch: grid = H blocks, block = S threads.

#include <hip/hip_runtime.h>

#ifndef GATED_DELTANET_AR_MAX_S
#define GATED_DELTANET_AR_MAX_S 256
#endif

extern "C" __global__ __launch_bounds__(GATED_DELTANET_AR_MAX_S)
void gated_deltanet_ar(
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
    const int h = blockIdx.x;
    const int j = threadIdx.x;   // state column; block size == S

    __shared__ float ksh[GATED_DELTANET_AR_MAX_S];
    __shared__ float qsh[GATED_DELTANET_AR_MAX_S];

    float* s = state + (size_t)h * S * S;
    const float qScale = 1.0f / sqrtf((float)S);

    for (int t = 0; t < T; ++t) {
        const int base = (t * H + h) * S;

        ksh[j] = k[base + j];
        qsh[j] = q[base + j] * qScale;
        __syncthreads();

        const float g  = expf(gLog[t * H + h]);
        const float b  = beta[t * H + h];
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