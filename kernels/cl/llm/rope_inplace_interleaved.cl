// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

// Rotary positional embedding (RoPE), in-place, INTERLEAVED ("GPT-J" /
// llama.cpp LLAMA_ROPE_TYPE_NORM) layout used by the `llama` architecture
// (Llama-1/2/3, Orpheus TTS). Unlike rope_inplace.cl (NEOX / split pairs
// (i, i+halfDim)), this rotates ADJACENT pairs (2i, 2i+1):
//
//   theta_i = (startPos + p) * base^(-2i / headDim)
//   c = cos(theta_i), s = sin(theta_i)
//   x'[2i]     = x[2i]   * c - x[2i+1] * s
//   x'[2i+1]   = x[2i]   * s + x[2i+1] * c
//
// Layout of x and dispatch geometry are identical to rope_inplace.cl.
// Applying the NEOX kernel to a `llama` checkpoint rotates the wrong
// coordinate pairs and silently degenerates generation.

#ifndef ROPE_LOCAL
#define ROPE_LOCAL 256
#endif

// `startPos` slot + `writeOffsetStride` semantics: see rope_inplace.cl.
__attribute__((reqd_work_group_size(ROPE_LOCAL, 1, 1)))
__kernel void rope_inplace_interleaved(
    __global       float* x_base,
    const int             seqLen,
    const int             numHeads,
    const int             headDim,
    __global const int*   startPosPtr,
    const float           base,
    const int             writeOffsetStride)
{
    const int gid     = (int)get_global_id(0);
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
    const float pos      = (float)(startPos + p);
    __global float* x    =
        x_base + (size_t)startPos * (size_t)writeOffsetStride;
    const float invDim = 1.0f / (float)headDim;
    const float freq   = pow(base, -(float)(2 * i) * invDim);
    const float theta  = pos * freq;
    const float c      = cos(theta);
    const float s      = sin(theta);

    const int headBase = (p * numHeads + h) * headDim;
    const float a = x[headBase + 2 * i];
    const float b = x[headBase + 2 * i + 1];
    x[headBase + 2 * i]     = a * c - b * s;
    x[headBase + 2 * i + 1] = a * s + b * c;
}
