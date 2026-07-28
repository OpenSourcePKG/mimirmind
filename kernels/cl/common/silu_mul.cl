// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

// SwiGLU fused step: gate[i] = (silu(gate[i])) * up[i]
//
//   silu(x) = x / (1 + exp(-x))
//
// Replaces the two CPU passes siluInPlace(gate) followed by
// mulInPlace(gate, up) — half the memory traffic and no intermediate
// sync between the silu and the multiply.
//
// Launch: 1D global = n work-items in groups of SILU_MUL_LOCAL.

#ifndef SILU_MUL_LOCAL
#define SILU_MUL_LOCAL 256
#endif

__attribute__((reqd_work_group_size(SILU_MUL_LOCAL, 1, 1)))
__kernel void silu_mul(
    __global       float* gate,    // input + output
    __global const float* up,
    const int             n)
{
    const int gid = (int)get_global_id(0);
    if (gid >= n) {
        return;
    }
    const float g = gate[gid];
    const float s = g / (1.0f + exp(-g));   // silu(g)
    gate[gid] = s * up[gid];
}