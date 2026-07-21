// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

// GatedDeltaNet decay gate: gLog[t,h] = -exp(ssmA[h]) *
// softplus(alpha[t,h] + ssmDt[h]). Numerically-stable softplus. Raw
// log-decay fed to gated_deltanet_ar. CPU reference: compute::deltanetGate.
//   alpha [T,H]; ssmA, ssmDt [H]; gLog (out) [T,H].
// Launch: 1D global = T*H.

#ifndef DELTANET_GATE_LOCAL
#define DELTANET_GATE_LOCAL 256
#endif

__attribute__((reqd_work_group_size(DELTANET_GATE_LOCAL, 1, 1)))
__kernel void deltanet_gate(
    __global const float* alpha,
    __global const float* ssmA,
    __global const float* ssmDt,
    __global       float* gLog,
    const int             T,
    const int             H)
{
    const int gid = (int)get_global_id(0);
    if (gid >= T * H) {
        return;
    }
    const int h = gid % H;
    const float x  = alpha[gid] + ssmDt[h];
    const float sp = x > 0.0f ? x + log1p(exp(-x)) : log1p(exp(x));
    gLog[gid] = -exp(ssmA[h]) * sp;
}
