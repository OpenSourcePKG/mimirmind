// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// In-place L2 normalisation over the innermost `dim`. HIP port of
// kernels/l2_norm.cl — ggml `ggml_l2_norm` (Qwen3-Next q/k). CPU
// reference: compute::l2NormInPlace.
//
// Launch: grid ceil(rows / L2_NORM_LOCAL), block L2_NORM_LOCAL.

#include <hip/hip_runtime.h>

#ifndef L2_NORM_LOCAL
#define L2_NORM_LOCAL 64
#endif

extern "C" __global__ __launch_bounds__(L2_NORM_LOCAL)
void l2_norm(
    float*    __restrict__ x,
    const int              rows,
    const int              dim,
    const float            eps)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) {
        return;
    }
    float* row = x + (size_t)r * dim;
    float sumsq = 0.0f;
    for (int j = 0; j < dim; ++j) {
        sumsq += row[j] * row[j];
    }
    const float scale = 1.0f / fmaxf(sqrtf(sumsq), eps);
    for (int j = 0; j < dim; ++j) {
        row[j] *= scale;
    }
}