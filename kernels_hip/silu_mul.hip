// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// SwiGLU fused step: gate[i] = silu(gate[i]) * up[i]
//
//   silu(x) = x / (1 + exp(-x))
//
// HIP port of kernels/silu_mul.cl. Trivial element-wise fused kernel
// — one thread per element, one input, one in/out, no reduction, no
// LDS. Ported unchanged in shape from the L0 reference.
//
// Launch:
//   dim3 grid ( ceil(n / SILU_MUL_LOCAL), 1, 1 )
//   dim3 block( SILU_MUL_LOCAL, 1, 1 )

#include <hip/hip_runtime.h>

#ifndef SILU_MUL_LOCAL
#define SILU_MUL_LOCAL 256
#endif

extern "C" __global__ __launch_bounds__(SILU_MUL_LOCAL)
void silu_mul(
          float* __restrict__ gate,   // input + output
    const float* __restrict__ up,
    const int                 n)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) {
        return;
    }
    const float g = gate[gid];
    // expf is the accurate HIP intrinsic; __expf is the fast-approx
    // variant. Correctness-first for the port; a follow-up perf pass
    // can swap in __expf once we have a baseline.
    const float s = g / (1.0f + expf(-g));   // silu(g)
    gate[gid] = s * up[gid];
}