// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
// Ported from kernels_hip/rope_inplace_ff.hip — Track 4 mechanical port, no functional change intended.

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Rotary positional embedding (RoPE) with per-pair frequency factors,
// in-place. Mirrors ggml_rope_ext()'s `freq_factors` argument used by
// Gemma-family models for "proportional RoPE" / YaRN-style long-context
// scaling on global-attention layers.
//
//   theta_i = (startPos + p) * base^(-2i / headDim) / freq_factors[i]
//   c = cos(theta_i), s = sin(theta_i)
//   x'[i]           = x[i] * c - x[i + halfDim] * s
//   x'[i + halfDim] = x[i] * s + x[i + halfDim] * c
//
// HIP port of kernels/rope_inplace_ff.cl. `freq_factors` has shape
// [halfDim] (one factor per rotated pair). Layout of `x` and dispatch
// geometry identical to rope_inplace.hip.

#include <cuda_runtime.h>

#ifndef ROPE_LOCAL
#define ROPE_LOCAL 256
#endif

extern "C" __global__ __launch_bounds__(ROPE_LOCAL)
void rope_inplace_ff(
          float* __restrict__ x_base,
    const float* __restrict__ freq_factors,   // [halfDim]
    const int                 seqLen,
    const int                 numHeads,
    const int                 headDim,
    const int* __restrict__   startPosPtr,
    const float               base,
    const int                 writeOffsetStride)
{
    const int gid     = blockIdx.x * blockDim.x + threadIdx.x;
    const int halfDim = headDim / 2;
    const int total   = seqLen * numHeads * halfDim;
    if (gid >= total) {
        return;
    }

    const int i  = gid % halfDim;
    const int hp = gid / halfDim;
    const int h  = hp % numHeads;
    const int p  = hp / numHeads;

    const int   startPos = startPosPtr[0];
    const float pos      = static_cast<float>(startPos + p);
    float*      x        =
        x_base + static_cast<size_t>(startPos)
               * static_cast<size_t>(writeOffsetStride);
    const float invDim   = 1.0f / static_cast<float>(headDim);
    const float baseFreq = powf(base, -static_cast<float>(2 * i) * invDim);
    const float ff       = freq_factors[i];
    const float freq     = baseFreq / ff;
    const float theta    = pos * freq;
    const float c        = cosf(theta);
    const float s        = sinf(theta);

    const int   headBase = (p * numHeads + h) * headDim;
    const float a = x[headBase + i];
    const float b = x[headBase + i + halfDim];
    x[headBase + i]           = a * c - b * s;
    x[headBase + i + halfDim] = a * s + b * c;
}