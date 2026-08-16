// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// P2.b — 2-way row-split ("r2") variant of the smem-staged GDN recurrence
// (gated_deltanet_ar_batched_v3), single-sequence prefill.
//
// v3 staged the [S,S] state into dynamic smem (removing the global-state
// stall), but each column j is still owned by ONE thread that walks the
// full length-S row reduction serially — and with S=128 threads = 4 warps
// per block and 64 KiB smem forcing 1 block/SM, only 4 warps/SM cannot hide
// the smem latency of that serial reduction (latency-bound).
//
// r2 gives each column TWO threads (block = 2*S = 8 warps/SM): the pair
// splits the length-S Σ_i reduction in half and combines with a single
// intra-warp __shfl_xor (the pair is lanes 2j, 2j+1 — same warp, no smem /
// __syncthreads for the combine). This halves the per-thread reduction
// length AND doubles the resident warps, so the latency is better hidden.
//
// Math is the same ops as v3 (s*g recompute in pass 2, single state write),
// only the Σ_i summation is split into two halves then added — a different
// FP association, so bit-NEAR (not bit-exact) but within the golden
// tolerance (cuda_gated_deltanet_ar_parity gates it). Opt-in
// MIMIRMIND_GDN_PREFILL_R2=1.
//
// Layout: q,k,v [T,H,S]; gLog,beta [T,H]; state [H,S,S] s[h][i*S+j];
// out [T,H,S]. Launch: grid = H, block = 2*S, sharedMemBytes = S*S*4.

#include <cuda_runtime.h>

#ifndef GATED_DELTANET_AR_MAX_S
#define GATED_DELTANET_AR_MAX_S 256
#endif

extern "C" __global__ __launch_bounds__(GATED_DELTANET_AR_MAX_S)
void gated_deltanet_ar_r2(
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
    const int h    = blockIdx.x;
    const int tid  = threadIdx.x;          // 0 .. 2S-1
    const int j    = tid >> 1;             // state column (0 .. S-1)
    const int half = tid & 1;             // 0 = rows [0,S/2), 1 = rows [S/2,S)
    const int i0   = half * (S >> 1);
    const int i1   = i0 + (S >> 1);

    extern __shared__ float sstate[];      // [S,S] state, row-major i*S + j
    __shared__ float ksh[GATED_DELTANET_AR_MAX_S];
    __shared__ float qsh[GATED_DELTANET_AR_MAX_S];

    float* s = state + (size_t)h * S * S;
    const float qScale = 1.0f / sqrtf((float)S);

    // Stage state global -> smem cooperatively (coalesced, 2S threads).
    for (int idx = tid; idx < S * S; idx += 2 * S) {
        sstate[idx] = s[idx];
    }
    // k/q staged by the first S threads (one column each).
    __syncthreads();

    for (int t = 0; t < T; ++t) {
        const int base = (t * H + h) * S;

        if (tid < S) {
            ksh[tid] = k[base + tid];
            qsh[tid] = q[base + tid] * qScale;
        }
        __syncthreads();

        const float g  = expf(gLog[t * H + h]);
        const float b  = beta[t * H + h];
        const float vj = v[base + j];

        // Pass 1: half-range reduction, combine the pair with one shuffle.
        float psk = 0.0f;
        for (int i = i0; i < i1; ++i) {
            psk += (sstate[(size_t)i * S + j] * g) * ksh[i];
        }
        const float sk = psk + __shfl_xor_sync(0xffffffffu, psk, 1);
        const float dj = (vj - sk) * b;

        // Pass 2: recompute decayed state, write once, half-range out accum.
        float poj = 0.0f;
        for (int i = i0; i < i1; ++i) {
            const float sij = sstate[(size_t)i * S + j] * g + ksh[i] * dj;
            sstate[(size_t)i * S + j] = sij;
            poj += sij * qsh[i];
        }
        const float oj = poj + __shfl_xor_sync(0xffffffffu, poj, 1);
        if (half == 0) {
            out[base + j] = oj;
        }

        __syncthreads();
    }

    // Write the final state back once (coalesced).
    for (int idx = tid; idx < S * S; idx += 2 * S) {
        s[idx] = sstate[idx];
    }
}
