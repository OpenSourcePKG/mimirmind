// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

// In-place logistic sigmoid: y[i] = 1/(1+exp(-y[i])). GatedDeltaNet beta
// gate. CPU reference: compute::sigmoidInPlace.
// Launch: 1D global = n.

#ifndef SIGMOID_INPLACE_LOCAL
#define SIGMOID_INPLACE_LOCAL 256
#endif

__attribute__((reqd_work_group_size(SIGMOID_INPLACE_LOCAL, 1, 1)))
__kernel void sigmoid_inplace(
    __global float* y,
    const int       n)
{
    const int gid = (int)get_global_id(0);
    if (gid >= n) {
        return;
    }
    y[gid] = 1.0f / (1.0f + exp(-y[gid]));
}
