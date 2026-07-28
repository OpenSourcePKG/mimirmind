// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
// Ported from kernels_hip/rope_inplace.hip — Track 4 mechanical port, no functional change intended.

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Rotary positional embedding (RoPE), in-place. llama.cpp "non-interleaved"
// (split) layout used by Llama / Qwen / Gemma family models.
//
//   theta_i = (startPos + p) * base^(-2i / headDim)
//   c = cos(theta_i), s = sin(theta_i)
//   x'[i]           = x[i] * c - x[i + halfDim] * s
//   x'[i + halfDim] = x[i] * s + x[i + halfDim] * c
//
// Layout: x is [seqLen, numHeads, headDim] f32 row-major.
//
// HIP port of kernels/rope_inplace.cl. Element-wise, no reduction, no
// LDS — the simplest of the three kernels we've ported so far. Each
// thread reads two floats into registers and writes them back rotated;
// atomic w.r.t. itself even without explicit sync because HIP kernels
// have no cross-thread aliasing of registers.
//
// Launch:
//   dim3 grid ( ceil(seqLen * numHeads * halfDim / ROPE_LOCAL), 1, 1 )
//   dim3 block( ROPE_LOCAL, 1, 1 )
//
// The `startPos` slot lives in a device int-buffer, matching the L0
// M-CLR.2 command-list-replay contract: host writes the current
// startPos before every launch, kernel picks it up at runtime, no
// re-record needed. `writeOffsetStride` shifts the buffer base by
// `startPos * writeOffsetStride` — Q-rope passes 0 (workspace stays
// fixed), K-rope passes the layer's kvDim so the kernel writes into
// the current KV-cache slot while x_base stays stable across replays.

#include <cuda_runtime.h>

#ifndef ROPE_LOCAL
#define ROPE_LOCAL 256
#endif

extern "C" __global__ __launch_bounds__(ROPE_LOCAL)
void rope_inplace(
          float* __restrict__ x_base,
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

    // gid = ((p * numHeads) + h) * halfDim + i
    const int i  = gid % halfDim;
    const int hp = gid / halfDim;
    const int h  = hp % numHeads;
    const int p  = hp / numHeads;

    const int   startPos = startPosPtr[0];
    const float pos      = static_cast<float>(startPos + p);
    float*      x        =
        x_base + static_cast<size_t>(startPos)
               * static_cast<size_t>(writeOffsetStride);
    const float invDim = 1.0f / static_cast<float>(headDim);
    const float freq   = powf(base, -static_cast<float>(2 * i) * invDim);
    const float theta  = pos * freq;
    const float c      = cosf(theta);
    const float s      = sinf(theta);

    const int   headBase = (p * numHeads + h) * headDim;
    const float a = x[headBase + i];
    const float b = x[headBase + i + halfDim];
    x[headBase + i]           = a * c - b * s;
    x[headBase + i + halfDim] = a * s + b * c;
}