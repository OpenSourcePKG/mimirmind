// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

// In-place 1D add: y[i] += x[i] for i in [0, n).
//
// Used for the post-attention and post-FFN residual streams.
// Launch: 1D global = n work-items in groups of ADD_RESIDUAL_LOCAL.

#ifndef ADD_RESIDUAL_LOCAL
#define ADD_RESIDUAL_LOCAL 256
#endif

__attribute__((reqd_work_group_size(ADD_RESIDUAL_LOCAL, 1, 1)))
__kernel void add_residual(
    __global       float* y,
    __global const float* x,
    const int             n)
{
    const int gid = (int)get_global_id(0);
    if (gid >= n) {
        return;
    }
    y[gid] += x[gid];
}