// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// In-place element-wise y = y + 1 over an F32 tensor. Used by the NVFP4
// materializer for the transformer RMSNorm weights (attn / q / k / post /
// output norms), which the HF checkpoint stores centred at 0 under the
// (1 + w) convention. llama.cpp bakes the +1 into the GGUF norm tensors and
// the runtime norm kernel multiplies by the stored weight directly, so the
// NVFP4 path adds 1 here at load time. The GatedDeltaNet ssm_norm uses the
// plain (uncentred) convention and is excluded by the planner.
//
// Launch:
//   dim3 grid ( ceil(n / ADD_ONE_LOCAL), 1, 1 )
//   dim3 block( ADD_ONE_LOCAL, 1, 1 )

#include <cuda_runtime.h>

#ifndef ADD_ONE_LOCAL
#define ADD_ONE_LOCAL 256
#endif

extern "C" __global__ __launch_bounds__(ADD_ONE_LOCAL)
void add_one_f32(
    float*     __restrict__ x, // (n,) F32, transformed in place
    const long              n)
{
    const long gid = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= n) {
        return;
    }
    x[gid] = x[gid] + 1.0f;
}
