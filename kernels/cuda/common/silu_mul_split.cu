// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// SwiGLU split for a stacked-w13 grouped-GEMM output (roadmap 5.18.8).
//
// The fused gate+up grouped GEMM writes each row as [gate(nff) | up(nff)],
// i.e. w13 is [rows][2*nff] with gate at column offset 0 and up at offset nff.
// This kernel applies SwiGLU and de-interleaves into a contiguous [rows][nff]
// buffer that the down projection consumes:
//
//   out[r*nff + j] = silu(w13[r*2nff + j]) * w13[r*2nff + nff + j]
//   silu(x) = x / (1 + exp(-x))
//
// One thread per output element. Bit-identical to the two-launch
// silu_mul(gate, up) path (same math, just a strided read).
//
// Launch:
//   dim3 grid ( ceil(rows*nff / SILU_MUL_SPLIT_LOCAL), 1, 1 )
//   dim3 block( SILU_MUL_SPLIT_LOCAL, 1, 1 )

#include <cuda_runtime.h>

#ifndef SILU_MUL_SPLIT_LOCAL
#define SILU_MUL_SPLIT_LOCAL 256
#endif

extern "C" __global__ __launch_bounds__(SILU_MUL_SPLIT_LOCAL)
void silu_mul_split(
    const float* __restrict__ w13,   // [rows][2*nff], row = [gate(nff) | up(nff)]
          float* __restrict__ out,   // [rows][nff]
    const int                 rows,
    const int                 nff)
{
    const long gid   = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long total = static_cast<long>(rows) * nff;
    if (gid >= total) {
        return;
    }
    const int  r    = static_cast<int>(gid / nff);
    const int  j    = static_cast<int>(gid - static_cast<long>(r) * nff);
    const long base = static_cast<long>(r) * (2 * nff);
    const float g = w13[base + j];
    const float u = w13[base + nff + j];
    const float s = g / (1.0f + expf(-g));   // silu(g)
    out[gid] = s * u;
}
