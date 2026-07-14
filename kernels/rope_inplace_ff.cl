// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

// Rotary positional embedding (RoPE) with per-pair frequency factors,
// in-place. Mirrors ggml_rope_ext()'s `freq_factors` argument used by
// Gemma family models for "proportional RoPE" / YaRN-style long-context
// scaling on global-attention layers.
//
//   theta_i = (startPos + p) * base^(-2i / headDim) / freq_factors[i]
//   c = cos(theta_i), s = sin(theta_i)
//   x'[i]           = x[i] * c - x[i + halfDim] * s
//   x'[i + halfDim] = x[i] * s + x[i + halfDim] * c
//
// `freq_factors` has shape [halfDim] (one factor per rotated pair). Layout
// of `x` and dispatch geometry are identical to rope_inplace.cl.

#ifndef ROPE_LOCAL
#define ROPE_LOCAL 256
#endif

// M-CLR.2: `startPos` moved to a __global int-slot — see rope_inplace.cl.
// M-CLR.2 Wave 3b: `writeOffsetStride` shifts the base pointer by
// `startPos * writeOffsetStride` for K-rope. See rope_inplace.cl.
__attribute__((reqd_work_group_size(ROPE_LOCAL, 1, 1)))
__kernel void rope_inplace_ff(
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
    const float a = x[headBase + i];
    const float b = x[headBase + i + halfDim];
    x[headBase + i]           = a * c - b * s;
    x[headBase + i + halfDim] = a * s + b * c;
}