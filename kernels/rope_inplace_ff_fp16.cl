// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

// FP16-KV variant of rope_inplace_ff.cl. K-cache slots are stored as
// fp16; the rotation (with per-pair freq_factors, ggml_rope_ext-style)
// is done in fp32 in registers so the storage swap doesn't cost
// precision. Layout / dispatch geometry identical to the f32 variant —
// only `x_base`'s pointer type differs.
//
// Called by the K-rope path (Gemma global-attention layers) when
// `KvCache::dtype() == FP16`. Q-rope stays on rope_inplace_ff.cl
// against its fp32 workspace regardless of KV dtype.
//
// M10.2 Phase 0 Commit 4 — bit-parity against pre-M10.2 behaviour is
// preserved because this file is only reached from an explicit
// FP16-dtype dispatch.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifndef ROPE_LOCAL
#define ROPE_LOCAL 256
#endif

__attribute__((reqd_work_group_size(ROPE_LOCAL, 1, 1)))
__kernel void rope_inplace_ff_fp16(
    __global       half*  x_base,
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
    __global half* x     =
        x_base + (size_t)startPos * (size_t)writeOffsetStride;
    const float invDim   = 1.0f / (float)headDim;
    const float baseFreq = pow(base, -(float)(2 * i) * invDim);
    const float ff       = freq_factors[i];
    const float freq     = baseFreq / ff;
    const float theta    = pos * freq;
    const float c        = cos(theta);
    const float s        = sin(theta);

    const int headBase = (p * numHeads + h) * headDim;
    const float a = vload_half(headBase + i,           x);
    const float b = vload_half(headBase + i + halfDim, x);
    vstore_half(a * c - b * s, headBase + i,           x);
    vstore_half(a * s + b * c, headBase + i + halfDim, x);
}