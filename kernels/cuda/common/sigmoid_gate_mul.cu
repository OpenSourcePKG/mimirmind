// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// In-place sigmoid gating: y[r,c] *= sigmoid(g[r, gateDim==1?0:c]). CUDA
// port of kernels/sigmoid_gate_mul.cl — see that file for the two
// Qwen3-Next call sites (attention output gate, shared-expert gate).
//
// Launch:
//   dim3 grid ( ceil(rows*dim / SIGMOID_GATE_MUL_LOCAL), 1, 1 )
//   dim3 block( SIGMOID_GATE_MUL_LOCAL, 1, 1 )

#include <cuda_runtime.h>

#ifndef SIGMOID_GATE_MUL_LOCAL
#define SIGMOID_GATE_MUL_LOCAL 256
#endif

extern "C" __global__ __launch_bounds__(SIGMOID_GATE_MUL_LOCAL)
void sigmoid_gate_mul(
    float*       __restrict__ y,
    const float* __restrict__ g,
    const int                 rows,
    const int                 dim,
    const int                 gateDim)
{
    const int gid   = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * dim;
    if (gid >= total) {
        return;
    }

    const int c    = gid % dim;
    const int r    = gid / dim;
    const int gcol = (gateDim == 1) ? 0 : c;
    const float gv  = g[(size_t)r * gateDim + gcol];
    const float sig = 1.0f / (1.0f + expf(-gv));
    y[gid] *= sig;
}