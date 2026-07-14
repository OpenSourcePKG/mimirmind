// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

// FP16-KV variant of rope_inplace.cl. K-cache slots are stored as fp16;
// the rotation is done in fp32 in registers (vload_half → rotate →
// vstore_half) so RoPE precision doesn't degrade from the storage
// swap. Layout / dispatch geometry is identical to the f32 variant —
// only the pointer type of `x_base` differs.
//
// Called by the K-rope path when `KvCache::dtype() == FP16`. The Q-rope
// path still targets the fp32 workspace and continues to use
// rope_inplace.cl regardless of KV dtype.
//
// M10.2 Phase 0 Commit 4 — bit-parity against pre-M10.2 behaviour is
// preserved because this file is only reached from an explicit
// FP16-dtype dispatch.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifndef ROPE_LOCAL
#define ROPE_LOCAL 256
#endif

__attribute__((reqd_work_group_size(ROPE_LOCAL, 1, 1)))
__kernel void rope_inplace_fp16(
    __global       half*  x_base,
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
    const float invDim = 1.0f / (float)headDim;
    const float freq   = pow(base, -(float)(2 * i) * invDim);
    const float theta  = pos * freq;
    const float c      = cos(theta);
    const float s      = sin(theta);

    const int headBase = (p * numHeads + h) * headDim;
    const float a = vload_half(headBase + i,           x);
    const float b = vload_half(headBase + i + halfDim, x);
    vstore_half(a * c - b * s, headBase + i,           x);
    vstore_half(a * s + b * c, headBase + i + halfDim, x);
}