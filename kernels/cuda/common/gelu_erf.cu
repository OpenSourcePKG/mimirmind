// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Exact (erf) GELU, in place — PyTorch/HF nn.GELU default, as BERT /
// RoBERTa / XLM-R use it in the encoder FFN (EncoderRunner / cross-encoder
// reranker path):
//
//   x[i] = 0.5 * x[i] * (1 + erf(x[i] / sqrt(2)))
//
// Distinct from gelu_mul.cu (the tanh approximation used by Gemma's
// SwiGLU-parallel FFN). One thread per element, no reduction, no LDS.
//
// Launch:
//   dim3 grid ( ceil(n / GELU_ERF_LOCAL), 1, 1 ), dim3 block( GELU_ERF_LOCAL )

#include <cuda_runtime.h>

#ifndef GELU_ERF_LOCAL
#define GELU_ERF_LOCAL 256
#endif

extern "C" __global__ __launch_bounds__(GELU_ERF_LOCAL)
void gelu_erf(
          float* __restrict__ x,   // input + output
    const int                 n)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) {
        return;
    }
    const float v = x[gid];
    // erff is the accurate CUDA intrinsic (matches std::erf on the CPU ref).
    x[gid] = 0.5f * v * (1.0f + erff(v * 0.70710678118654752440f));
}
