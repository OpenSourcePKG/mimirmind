// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// GLU with EXACT (erf) GELU activation, in place on `gate`:
//
//   gate[i] = gelu_erf(gate[i]) * up[i]
//
// where gelu_erf is the PyTorch/HF nn.GELU default (as ModernBERT's
// hidden_activation="gelu"):
//
//   gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
//
// Distinct from gelu_mul.cu (the tanh approximation used by Gemma's FFN):
// ModernBERT's GeGLU splits Wi into [input | gate] and computes
// gelu_erf(input) * gate, so the parity path must use erf, not tanh.
// One thread per element; no reduction, no LDS.
//
// Launch:
//   dim3 grid ( ceil(n / GELU_ERF_MUL_LOCAL), 1, 1 )
//   dim3 block( GELU_ERF_MUL_LOCAL, 1, 1 )

#include <cuda_runtime.h>

#ifndef GELU_ERF_MUL_LOCAL
#define GELU_ERF_MUL_LOCAL 256
#endif

extern "C" __global__ __launch_bounds__(GELU_ERF_MUL_LOCAL)
void gelu_erf_mul(
          float* __restrict__ gate,   // input (gets erf-GELU) + output
    const float* __restrict__ up,     // multiplier
    const int                 n)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) {
        return;
    }
    const float g  = gate[gid];
    // erff is the accurate CUDA intrinsic (matches std::erf on the CPU ref).
    const float ge = 0.5f * g * (1.0f + erff(g * 0.70710678118654752440f));
    gate[gid] = ge * up[gid];
}
