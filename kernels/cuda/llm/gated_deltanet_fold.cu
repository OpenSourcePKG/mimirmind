// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// GDN ReplaySSM fold — replay the ACCEPTED prefix of a gated delta-rule verify
// window into the committed recurrent state, in place. State-only: the SSM
// state update depends solely on k, v, gLog and beta (q only makes the output),
// so this is gated_deltanet_ar with the q/output readout removed. Byte-identical
// per-token math to gated_deltanet_ar / compute::gatedDeltaNetRecurrent — same
// s*g decay recompute, same sk accumulation order, same (v-sk)*beta gated error,
// same rank-1 update. The delta-rule recurrence is contractive, so replaying the
// accepted prefix from the raw inputs lands the committed state with no
// length-dependent error (ReplaySSM; SGLang gdn_replayssm_spec_decode).
//
// Used by DFlash / MTP partial-accept: instead of re-forwarding the trunk for
// the accepted prefix, restore the pre-window checkpoint and fold the accepted
// tokens' cached (k, v, gLog, beta) here.
//
// Layout (single request; matches gated_deltanet_ar): k,v [L,H,S]; gLog,beta
// [L,H]; state [H,S,S] as s[h][i*S + j] (in/out, folded in place). Replays
// timesteps t = 0 .. acceptLen-1 (acceptLen <= L). Launch: grid = H blocks,
// block = S threads.

#include <cuda_runtime.h>

#ifndef GATED_DELTANET_AR_MAX_S
#define GATED_DELTANET_AR_MAX_S 256
#endif

extern "C" __global__ __launch_bounds__(GATED_DELTANET_AR_MAX_S)
void gated_deltanet_fold(
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ gLog,
    const float* __restrict__ beta,
    float*       __restrict__ state,
    const int                 acceptLen,
    const int                 H,
    const int                 S)
{
    const int h = blockIdx.x;
    const int j = threadIdx.x;   // state column; block size == S

    __shared__ float ksh[GATED_DELTANET_AR_MAX_S];

    float* s = state + (size_t)h * S * S;

    for (int t = 0; t < acceptLen; ++t) {
        const int base = (t * H + h) * S;

        ksh[j] = k[base + j];
        __syncthreads();

        const float g  = expf(gLog[t * H + h]);
        const float b  = beta[t * H + h];
        const float vj = v[base + j];

        // Decay the state column and predict sk[j] = sum_i (s[i,j]*g) * k[i]
        // from the decayed state (fused, identical order to the AR kernel).
        float sk = 0.0f;
        for (int i = 0; i < S; ++i) {
            const float sij = s[(size_t)i * S + j] * g;
            s[(size_t)i * S + j] = sij;
            sk += sij * ksh[i];
        }
        const float dj = (vj - sk) * b;

        // Rank-1 update s[i,j] += k[i] * d[j]. No output readout (state only).
        for (int i = 0; i < S; ++i) {
            s[(size_t)i * S + j] += ksh[i] * dj;
        }

        __syncthreads();
    }
}
