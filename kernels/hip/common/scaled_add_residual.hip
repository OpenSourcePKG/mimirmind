// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Fused scaled-accumulate: dst[i] += scale * src[i]
//
// HIP port of kernels/scaled_add_residual.cl. Collapses what the
// Gemma 4 MoE per-expert loop did as two passes (mulScalar +
// addResidual) into one kernel. `fmaf` gives the compiler the fused-
// multiply-add signal — RDNA3 has native FMA on `v_fmac_f32`, so
// this is the correct primitive.
//
// Launch:
//   dim3 grid ( ceil(n / SCALED_ADD_RESIDUAL_LOCAL), 1, 1 )
//   dim3 block( SCALED_ADD_RESIDUAL_LOCAL, 1, 1 )

#include <hip/hip_runtime.h>

#ifndef SCALED_ADD_RESIDUAL_LOCAL
#define SCALED_ADD_RESIDUAL_LOCAL 256
#endif

extern "C" __global__ __launch_bounds__(SCALED_ADD_RESIDUAL_LOCAL)
void scaled_add_residual(
          float* __restrict__ dst,   // in + out
    const float* __restrict__ src,
    const float               scale,
    const int                 n)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) {
        return;
    }
    dst[gid] = fmaf(scale, src[gid], dst[gid]);
}