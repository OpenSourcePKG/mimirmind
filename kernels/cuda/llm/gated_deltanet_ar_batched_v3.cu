// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Shared-memory-staged variant of gated_deltanet_ar_batched — BYTE-IDENTICAL
// math to v1/v2.
//
// ncu (2026-07-30) on the v2 kernel: GDN decode is latency-bound — 98.9% of
// cycles have no eligible warp, ~937-cycle scoreboard stalls on the global
// state loads, occupancy already 87% (so it is NOT an occupancy problem). Both
// inner loops read s[i,j] straight from global; with only unroll-4 memory-level
// parallelism the loads cannot hide their latency.
//
// Key structural fact: state column j is PRIVATE to thread j — thread j only
// ever touches s[i,j] for all i, no other thread reads or writes that column.
// So the recurrence needs no cross-thread state sharing; the only reason the
// state lives in global is that a full column ([S] floats, up to 256) will not
// fit in registers.
//
// This variant stages the whole [S,S] state block into dynamic shared memory:
//   - load  : all threads cooperatively copy state global->smem. The S loads
//             per thread are fully independent (no data dependency) so they all
//             go in flight at once => maximal MLP, the global latency is hidden.
//             The access is coalesced (thread j reads s[i*S+j], contiguous in
//             j at fixed i).
//   - run   : the T timesteps read/modify/write the state in smem (~4-way bank
//             conflict for S=128, ~4-cycle vs ~937-cycle global latency).
//   - store : write the final state smem->global once (coalesced).
// Global state traffic drops from v2's (2R + 1W) per timestep to (1R + 1W)
// for the ENTIRE kernel call, and the latency-critical recurrence reads no
// longer touch global. The sk/oj accumulation order and the s*g recompute are
// unchanged, so the result is bit-identical to v1/v2.
//
// Dynamic shared memory: S*S floats. For the prod width S=128 that is 64 KiB,
// above the 48 KiB static cap, so the host must opt in via
// cuFuncSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES) and pass
// sharedMemBytes = S*S*sizeof(float) at launch. The host only selects this
// kernel when S*S*4 fits the device opt-in cap; otherwise it falls back to v2.
//
// Launch: grid = dim3(H, nSeq, 1), block = S threads,
//         sharedMemBytes = S*S*sizeof(float).

#include <cuda_runtime.h>

#ifndef GATED_DELTANET_AR_MAX_S
#define GATED_DELTANET_AR_MAX_S 256
#endif

extern "C" __global__ __launch_bounds__(GATED_DELTANET_AR_MAX_S)
void gated_deltanet_ar_batched_v3(
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

    // Dynamic [S,S] state staging buffer, same row-major [i*S + j] layout as
    // the global state (so the indexing and thus the arithmetic is unchanged).
    extern __shared__ float sstate[];
    __shared__ float ksh[GATED_DELTANET_AR_MAX_S];
    __shared__ float qsh[GATED_DELTANET_AR_MAX_S];

    const size_t actSeqStride   = (size_t)T * H * S;   // q,k,v,out
    const size_t gateSeqStride  = (size_t)T * H;       // gLog,beta
    const size_t stateSeqStride = (size_t)H * S * S;   // state

    float* s = state + (size_t)seq * stateSeqStride + (size_t)h * S * S;
    const float qScale = 1.0f / sqrtf((float)S);

    // Stage state global -> smem. Thread j copies its own column j; the S loads
    // are independent, so they pipeline and hide the global latency. Coalesced
    // in j at each fixed i.
    for (int i = 0; i < S; ++i) {
        sstate[(size_t)i * S + j] = s[(size_t)i * S + j];
    }
    __syncthreads();

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
            sk += (sstate[(size_t)i * S + j] * g) * ksh[i];
        }
        const float dj = (vj - sk) * b;

        // Pass 2: recompute the decayed state, write it once, accumulate out.
        float oj = 0.0f;
#pragma unroll 4
        for (int i = 0; i < S; ++i) {
            const float sij = sstate[(size_t)i * S + j] * g + ksh[i] * dj;
            sstate[(size_t)i * S + j] = sij;
            oj += sij * qsh[i];
        }
        out[base + j] = oj;

        __syncthreads();
    }

    // Write the final recurrent state back to global once (coalesced). Only the
    // final state is observable outside this kernel call, so intermediate
    // timesteps never touch global — this is bit-identical to v2's final store.
    for (int i = 0; i < S; ++i) {
        s[(size_t)i * S + j] = sstate[(size_t)i * S + j];
    }
}
