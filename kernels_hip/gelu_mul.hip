// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// SwiGLU with GELU activation (parallel form, like llama.cpp's
// LLM_FFN_GELU / LLM_FFN_PAR — used by Gemma 3 / Gemma 4):
//
//   gate[i] = gelu_tanh(gate[i]) * up[i]
//
// where gelu_tanh is the standard tanh approximation:
//
//   gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//
// HIP port of kernels/gelu_mul.cl. Structure identical to silu_mul —
// one thread per element, one input, one in/out, no reduction, no
// LDS. Constants: 0.7978845608 ≈ sqrt(2/pi), 0.044715 is the
// canonical Hendrycks-Gimpel value.
//
// Launch:
//   dim3 grid ( ceil(n / GELU_MUL_LOCAL), 1, 1 )
//   dim3 block( GELU_MUL_LOCAL, 1, 1 )

#include <hip/hip_runtime.h>

#ifndef GELU_MUL_LOCAL
#define GELU_MUL_LOCAL 256
#endif

extern "C" __global__ __launch_bounds__(GELU_MUL_LOCAL)
void gelu_mul(
          float* __restrict__ gate,   // input + output
    const float* __restrict__ up,
    const int                 n)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) {
        return;
    }
    const float g  = gate[gid];
    const float g3 = g * g * g;
    // tanhf is the accurate HIP intrinsic. The __tanhf fast-approx
    // variant matches the SPV ocloc output on the L0 side; leaving
    // tanhf here keeps the port strictly correctness-first — a perf
    // pass can swap in the intrinsic once bit-parity is established.
    const float t  = tanhf(0.7978845608f * (g + 0.044715f * g3));
    const float ge = 0.5f * g * (1.0f + t);
    gate[gid] = ge * up[gid];
}