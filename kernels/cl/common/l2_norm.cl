// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

// In-place L2 normalisation over the innermost `dim` (head_dim). Matches
// ggml `ggml_l2_norm` used on q/k in the Qwen3-Next linear layer:
//   x[r,:] /= max(sqrt(sum_j x[r,j]^2), eps)
// L2, NOT RMS — no learned weight, no 1/dim factor. CPU reference:
// compute::l2NormInPlace.
//
// Launch: 1D global = rows work-items (one per length-`dim` vector).

#ifndef L2_NORM_LOCAL
#define L2_NORM_LOCAL 64
#endif

__attribute__((reqd_work_group_size(L2_NORM_LOCAL, 1, 1)))
__kernel void l2_norm(
    __global       float* x,
    const int             rows,
    const int             dim,
    const float           eps)
{
    const int r = (int)get_global_id(0);
    if (r >= rows) {
        return;
    }
    __global float* row = x + (size_t)r * dim;
    float sumsq = 0.0f;
    for (int j = 0; j < dim; ++j) {
        sumsq += row[j] * row[j];
    }
    const float scale = 1.0f / fmax(sqrt(sumsq), eps);
    for (int j = 0; j < dim; ++j) {
        row[j] *= scale;
    }
}