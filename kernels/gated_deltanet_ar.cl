// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

// Autoregressive gated delta-rule recurrence (Qwen3-Next linear attention,
// decode + slow-prefill reference path). Processes T tokens strictly in
// order, per value-head. CPU reference: compute::gatedDeltaNetRecurrent
// (the golden truth this kernel must reproduce).
//
// Per token t, per head h, with head_dim S:
//   qs   = q[t,h,:] / sqrt(S)
//   g    = exp(gLog[t,h]) ; b = beta[t,h]
//   s[i,j] *= g
//   sk[j]  = sum_i s[i,j]*k[t,h,i]
//   d[j]   = (v[t,h,j] - sk[j]) * b
//   s[i,j] += k[t,h,i]*d[j]
//   out[t,h,j] = sum_i s[i,j]*qs[i]
//
// Parallelisation: one work-group per (v-)head, one work-item per state
// COLUMN j. Thread j touches only column j of the [S,S] state (fully
// independent across columns), except k[i] / qs[i] which are shared via
// local memory (loaded once per token). Single sequence (n_seqs == 1).
//
// Layout: q,k,v [T,H,S]; gLog,beta [T,H]; state [H,S,S] s[h][i*S+j];
// out [T,H,S]. Launch: local size = S, global = H*S (H work-groups).

#ifndef GATED_DELTANET_AR_MAX_S
#define GATED_DELTANET_AR_MAX_S 256
#endif

__kernel void gated_deltanet_ar(
    __global const float* q,
    __global const float* k,
    __global const float* v,
    __global const float* gLog,
    __global const float* beta,
    __global       float* state,
    __global       float* out,
    const int             T,
    const int             H,
    const int             S)
{
    const int h = (int)get_group_id(0);
    const int j = (int)get_local_id(0);   // state column; local size == S

    __local float ksh[GATED_DELTANET_AR_MAX_S];
    __local float qsh[GATED_DELTANET_AR_MAX_S];

    __global float* s = state + (size_t)h * S * S;
    const float qScale = 1.0f / sqrt((float)S);

    for (int t = 0; t < T; ++t) {
        const int base = (t * H + h) * S;

        // Load this token's k / (scaled) q into local memory.
        ksh[j] = k[base + j];
        qsh[j] = q[base + j] * qScale;
        barrier(CLK_LOCAL_MEM_FENCE);

        const float g  = exp(gLog[t * H + h]);
        const float b  = beta[t * H + h];
        const float vj = v[base + j];

        // Decay the state, then sk[j] = sum_i s[i,j]*k[i] from decayed s.
        float sk = 0.0f;
        for (int i = 0; i < S; ++i) {
            const float sij = s[(size_t)i * S + j] * g;
            s[(size_t)i * S + j] = sij;
            sk += sij * ksh[i];
        }
        const float dj = (vj - sk) * b;

        // Rank-1 update column j, and read out from the updated column.
        float oj = 0.0f;
        for (int i = 0; i < S; ++i) {
            const float sij = s[(size_t)i * S + j] + ksh[i] * dj;
            s[(size_t)i * S + j] = sij;
            oj += sij * qsh[i];
        }
        out[base + j] = oj;

        barrier(CLK_LOCAL_MEM_FENCE);   // before next token overwrites ksh/qsh
    }
}