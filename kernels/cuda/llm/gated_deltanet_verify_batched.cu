// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// M-Cuda.MTP-VerifyChunked MV-a — batched gated-DeltaNet VERIFY kernel.
//
// Processes the K+1-token speculative verify window for N slots in ONE launch,
// with the recurrent [S,S] state staged resident in shared memory across all
// T timesteps (no per-token global state round-trip), and — the verify-specific
// bit — EXPORTS the recurrent state at EVERY timestep so a partial accept can
// pick the accepted-prefix state without re-forwarding and without the
// per-step full-slab snapshot the sequential path used (ssmSnap/restoreSlotSsm).
//
// Byte-identical per-timestep math to gated_deltanet_ar_batched_v3 (same sk/oj
// accumulation order, same s*g recompute), just:
//   * split the state in/out: read S_0 from `stateIn`, never write it back;
//   * write the post-step state of timestep t to `stateOut[t]` (per position),
//     instead of only the final state.
//
// Activation layout matches the AR-batched path: q,k,v,out are [nSeq, T, H, S]
// (seq-major, then time), gLog/beta [nSeq, T, H]. The per-position state export
// is [T, nSeq, H, S, S] (time-major on T so the accept-commit gathers position
// a for each slot with a single contiguous per-(slot,head) block).
//
// Launch: grid = dim3(H, nSeq, 1), block = S threads,
//         sharedMemBytes = S*S*sizeof(float) (>48 KiB for S=128 -> host opts in
//         via CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES).

#include <cuda_runtime.h>

#ifndef GATED_DELTANET_AR_MAX_S
#define GATED_DELTANET_AR_MAX_S 256
#endif

extern "C" __global__ __launch_bounds__(GATED_DELTANET_AR_MAX_S)
void gated_deltanet_verify_batched(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ gLog,
    const float* __restrict__ beta,
    const float* __restrict__ stateIn,   // [nSeq, H, S, S]  initial state S_0
    float*       __restrict__ stateOut,  // [T, nSeq, H, S, S] per-position state
    float*       __restrict__ out,       // [nSeq, T, H, S]
    const int                 T,
    const int                 nSeq,
    const int                 H,
    const int                 S)
{
    const int seq = blockIdx.y;
    const int h   = blockIdx.x;
    const int j   = threadIdx.x;   // state column; block size == S

    extern __shared__ float sstate[];   // [S,S] row-major [i*S + j]
    __shared__ float ksh[GATED_DELTANET_AR_MAX_S];
    __shared__ float qsh[GATED_DELTANET_AR_MAX_S];

    const size_t actSeqStride   = (size_t)T * H * S;   // q,k,v,out per seq
    const size_t gateSeqStride  = (size_t)T * H;       // gLog,beta per seq
    const size_t stateSeqStride = (size_t)H * S * S;   // state per seq (one image)

    const float* s0 = stateIn + (size_t)seq * stateSeqStride + (size_t)h * S * S;
    const float qScale = 1.0f / sqrtf((float)S);

    // Stage S_0 global -> smem (coalesced in j, independent loads pipeline).
    for (int i = 0; i < S; ++i) {
        sstate[(size_t)i * S + j] = s0[(size_t)i * S + j];
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

        // Pass 1: sk = sum_i (s[i,j]*g) * k[i]  (order preserved for parity).
        float sk = 0.0f;
#pragma unroll 4
        for (int i = 0; i < S; ++i) {
            sk += (sstate[(size_t)i * S + j] * g) * ksh[i];
        }
        const float dj = (vj - sk) * b;

        // Pass 2: recompute decayed state, write it to smem, accumulate out.
        float oj = 0.0f;
#pragma unroll 4
        for (int i = 0; i < S; ++i) {
            const float sij = sstate[(size_t)i * S + j] * g + ksh[i] * dj;
            sstate[(size_t)i * S + j] = sij;
            oj += sij * qsh[i];
        }
        out[base + j] = oj;

        // Export the post-step state of timestep t (per position, per slot,head).
        // stateOut layout [T, nSeq, H, S, S]: contiguous S*S block per (t,seq,h).
        float* sOut = stateOut
                    + (((size_t)t * nSeq + seq) * H + h) * (size_t)S * S;
        for (int i = 0; i < S; ++i) {
            sOut[(size_t)i * S + j] = sstate[(size_t)i * S + j];
        }

        __syncthreads();
    }
}
