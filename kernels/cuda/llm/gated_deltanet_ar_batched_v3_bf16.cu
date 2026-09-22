// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// 5.18.10.5 Inc-1 — BF16-STATE variant of gated_deltanet_ar_batched_v3 (the
// smem-staged GatedDeltaNet AR/decode recurrence). IDENTICAL to the F32 v3
// kernel EXCEPT the recurrent `state` is stored in global as bf16: the single
// global load converts bf16->f32 into the f32 smem staging buffer, the T
// timesteps run entirely in f32 smem (math byte-identical to v3), and the single
// final store converts f32->bf16. This halves the global state R+W (the only
// remaining reducible gdn.recur decode term, DRAM-floor-bound at conc per
// 5.18.10.1) at a coherence cost already gated GO (needle 14/14 F32-parity to
// ~29k, MIMIRMIND_SSM_BF16_SIM, 5.18.10.4). Isolated (own bf16 state buffer);
// the full shared-buffer wiring (prefill chunk-forward + checkpoint + cross-slot)
// is the follow-on increment.
//
// Also carries a small f32->bf16 cast helper so a parity/timing harness can build
// a bf16 state from the same f32 state the F32 kernel runs on.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>

#ifndef GATED_DELTANET_AR_MAX_S
#define GATED_DELTANET_AR_MAX_S 256
#endif

extern "C" __global__ __launch_bounds__(GATED_DELTANET_AR_MAX_S)
void gated_deltanet_ar_batched_v3_bf16(
    const float*        __restrict__ q,
    const float*        __restrict__ k,
    const float*        __restrict__ v,
    const float*        __restrict__ gLog,
    const float*        __restrict__ beta,
    __nv_bfloat16*      __restrict__ state,          // bf16 recurrent state (in+out)
    float*              __restrict__ out,
    const int                        T,
    const int                        H,
    const int                        S,
    const unsigned char* __restrict__ activeMask,
    const int*          __restrict__ seqT,
    const int*          __restrict__ seqOff)
{
    const int seq = blockIdx.y;
    if (activeMask != nullptr && activeMask[seq] == 0) return;
    const int h   = blockIdx.x;
    const int j   = threadIdx.x;

    extern __shared__ float sstate[];   // [S,S] f32 staging (math unchanged)
    __shared__ float ksh[GATED_DELTANET_AR_MAX_S];
    __shared__ float qsh[GATED_DELTANET_AR_MAX_S];

    const size_t stateSeqStride = (size_t)H * S * S;
    const int    Tseq    = (seqT   != nullptr) ? seqT[seq]   : T;
    const size_t tokBase = (seqOff != nullptr) ? (size_t)seqOff[seq]
                                               : (size_t)seq * (size_t)T;

    __nv_bfloat16* s = state + (size_t)seq * stateSeqStride + (size_t)h * S * S;
    const float qScale = 1.0f / sqrtf((float)S);

    // Stage bf16 global state -> f32 smem (convert on load).
    for (int i = 0; i < S; ++i) {
        sstate[(size_t)i * S + j] = __bfloat162float(s[(size_t)i * S + j]);
    }
    __syncthreads();

    for (int t = 0; t < Tseq; ++t) {
        const size_t base    = (tokBase + (size_t)t) * (size_t)H * S
                             + (size_t)h * S;
        const size_t gateIdx = (tokBase + (size_t)t) * (size_t)H + (size_t)h;

        ksh[j] = k[base + j];
        qsh[j] = q[base + j] * qScale;
        __syncthreads();

        const float g  = expf(gLog[gateIdx]);
        const float b  = beta[gateIdx];
        const float vj = v[base + j];

        float sk = 0.0f;
#pragma unroll 4
        for (int i = 0; i < S; ++i) {
            sk += (sstate[(size_t)i * S + j] * g) * ksh[i];
        }
        const float dj = (vj - sk) * b;

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

    // Write final f32 smem state -> bf16 global (convert on store), once.
    for (int i = 0; i < S; ++i) {
        s[(size_t)i * S + j] = __float2bfloat16(sstate[(size_t)i * S + j]);
    }
}

// f32 -> bf16 elementwise cast (build a bf16 state buffer from an f32 one for the
// parity/timing harness).
#ifndef CAST_F32_BF16_LOCAL
#define CAST_F32_BF16_LOCAL 256
#endif
extern "C" __global__ __launch_bounds__(CAST_F32_BF16_LOCAL)
void cast_f32_to_bf16(const float* __restrict__ src,
                      __nv_bfloat16* __restrict__ dst, const int n)
{
    const int gid = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    dst[gid] = __float2bfloat16(src[gid]);
}
