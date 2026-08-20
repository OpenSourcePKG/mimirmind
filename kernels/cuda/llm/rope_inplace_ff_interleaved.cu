// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Rotary positional embedding (RoPE) with per-pair frequency factors,
// in-place, INTERLEAVED ("GPT-J" / llama.cpp LLAMA_ROPE_TYPE_NORM) layout.
// Combines rope_inplace_ff.cu's "llama3" proportional scaling with the
// adjacent-pair (2i, 2i+1) rotation the `llama` architecture requires
// (Llama-3.1/3.2 ship rope_freqs.weight AND are interleaved).
//
//   theta_i = (startPos + p) * base^(-2i / headDim) / freq_factors[i]
//   x'[2i]     = x[2i]   * c - x[2i+1] * s
//   x'[2i+1]   = x[2i]   * s + x[2i+1] * c
//
// `freq_factors` has shape [halfDim]. Layout / dispatch identical to
// rope_inplace.cu. See rope_inplace_interleaved.cu for the pairing note.

#include <cuda_runtime.h>

#ifndef ROPE_LOCAL
#define ROPE_LOCAL 256
#endif

extern "C" __global__ __launch_bounds__(ROPE_LOCAL)
void rope_inplace_ff_interleaved(
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
    const float a = x[headBase + 2 * i];
    const float b = x[headBase + 2 * i + 1];
    x[headBase + 2 * i]     = a * c - b * s;
    x[headBase + 2 * i + 1] = a * s + b * c;
}
