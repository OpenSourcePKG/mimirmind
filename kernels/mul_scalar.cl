// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

// In-place scalar multiply: y[i] *= s   for i in [0, n).
//
// Used by Gemma 4 for layer_output_scale (one F32 scalar per block
// that multiplies the block's final output) and potentially for any
// other learned scalar gates. The scalar arrives as a kernel argument
// rather than via a USM pointer dereference — caller reads the F32
// from the weight tensor and passes it through.
//
// Launch: 1D global = n in groups of MUL_SCALAR_LOCAL.

#ifndef MUL_SCALAR_LOCAL
#define MUL_SCALAR_LOCAL 256
#endif

__attribute__((reqd_work_group_size(MUL_SCALAR_LOCAL, 1, 1)))
__kernel void mul_scalar(
    __global       float* y,
    const float           s,
    const int             n)
{
    const int gid = (int)get_global_id(0);
    if (gid >= n) {
        return;
    }
    y[gid] *= s;
}