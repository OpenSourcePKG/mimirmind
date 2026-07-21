// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// GatedDeltaNet decay gate. HIP port of kernels/deltanet_gate.cl.
// CPU reference: compute::deltanetGate. Launch: grid ceil(T*H/L), block L.

#include <hip/hip_runtime.h>

#ifndef DELTANET_GATE_LOCAL
#define DELTANET_GATE_LOCAL 256
#endif

extern "C" __global__ __launch_bounds__(DELTANET_GATE_LOCAL)
void deltanet_gate(
    const float* __restrict__ alpha,
    const float* __restrict__ ssmA,
    const float* __restrict__ ssmDt,
    float*       __restrict__ gLog,
    const int                 T,
    const int                 H)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= T * H) {
        return;
    }
    const int h = gid % H;
    const float x  = alpha[gid] + ssmDt[h];
    const float sp = x > 0.0f ? x + log1pf(expf(-x)) : log1pf(expf(x));
    gLog[gid] = -expf(ssmA[h]) * sp;
}
