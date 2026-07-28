// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// In-place element-wise y = -exp(y) over an F32 tensor. Used by the NVFP4
// materializer to turn the HF `linear_attn.A_log` parameter into the GGUF
// `ssm_a` (= SSM_A_NOSCAN = -exp(A_log)) that the GatedDeltaNet decay gate
// consumes directly (gLog = ssm_a * softplus(...); see GatedDeltaNet.cpp and
// deltanet_gate.cu — no further exp is applied downstream). The GGUF loader
// gets this pre-computed from llama.cpp's conversion; the NVFP4 checkpoint
// stores the raw A_log, so we apply -exp() here once at load time.
//
// Launch:
//   dim3 grid ( ceil(n / NEG_EXP_LOCAL), 1, 1 )
//   dim3 block( NEG_EXP_LOCAL, 1, 1 )

#include <cuda_runtime.h>

#ifndef NEG_EXP_LOCAL
#define NEG_EXP_LOCAL 256
#endif

extern "C" __global__ __launch_bounds__(NEG_EXP_LOCAL)
void neg_exp_f32(
    float*     __restrict__ x, // (n,) F32, transformed in place
    const long              n)
{
    const long gid = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= n) {
        return;
    }
    x[gid] = -expf(x[gid]);
}