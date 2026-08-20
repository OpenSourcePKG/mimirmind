// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

// Rotary positional embedding (RoPE) with per-pair frequency factors,
// in-place, INTERLEAVED ("GPT-J" / llama.cpp LLAMA_ROPE_TYPE_NORM) layout.
// Combines rope_inplace_ff.cl's "llama3" proportional scaling with the
// adjacent-pair (2i, 2i+1) rotation the `llama` architecture requires
// (Llama-3.1/3.2 ship rope_freqs.weight AND are interleaved).
//
//   theta_i = (startPos + p) * base^(-2i / headDim) / freq_factors[i]
//   x'[2i]     = x[2i]   * c - x[2i+1] * s
//   x'[2i+1]   = x[2i]   * s + x[2i+1] * c
//
// `freq_factors` has shape [halfDim]. Layout / dispatch identical to
// rope_inplace.cl. See rope_inplace_interleaved.cl for the pairing note.

#ifndef ROPE_LOCAL
#define ROPE_LOCAL 256
#endif

// `startPos` slot + `writeOffsetStride` semantics: see rope_inplace.cl.
__attribute__((reqd_work_group_size(ROPE_LOCAL, 1, 1)))
__kernel void rope_inplace_ff_interleaved(
    __global       float* x_base,
    __global const float* freq_factors,
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

    const int i  = gid % halfDim;
    const int hp = gid / halfDim;
    const int h  = hp % numHeads;
    const int p  = hp / numHeads;

    const int   startPos = startPosPtr[0];
    const float pos      = (float)(startPos + p);
    __global float* x    =
        x_base + (size_t)startPos * (size_t)writeOffsetStride;
    const float invDim   = 1.0f / (float)headDim;
    const float baseFreq = pow(base, -(float)(2 * i) * invDim);
    const float ff       = freq_factors[i];
    const float freq     = baseFreq / ff;
    const float theta    = pos * freq;
    const float c        = cos(theta);
    const float s        = sin(theta);

    const int headBase = (p * numHeads + h) * headDim;
    const float a = x[headBase + 2 * i];
    const float b = x[headBase + 2 * i + 1];
    x[headBase + 2 * i]     = a * c - b * s;
    x[headBase + 2 * i + 1] = a * s + b * c;
}
