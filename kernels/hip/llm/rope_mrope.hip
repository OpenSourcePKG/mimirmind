// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Interleaved multi-axis RoPE (IMRoPE), in-place. HIP port of
// kernels/rope_mrope.cl — see that file for the full IMRoPE sector-rule
// derivation and the text-only-positions note. Qwen3-Next / Qwen3.5-VL
// full-attention layers (LLM_ROPE_TYPE_IMROPE). Bit-identical to
// rope_inplace for text (all four axis positions equal).
//
// Launch:
//   dim3 grid ( ceil(seqLen*numHeads*halfDim / ROPE_LOCAL), 1, 1 )
//   dim3 block( ROPE_LOCAL, 1, 1 )

#include <hip/hip_runtime.h>

#ifndef ROPE_LOCAL
#define ROPE_LOCAL 256
#endif

extern "C" __global__ __launch_bounds__(ROPE_LOCAL)
void rope_mrope(
    float*       __restrict__ x_base,
    const int                 seqLen,
    const int                 numHeads,
    const int                 headDim,
    const int*   __restrict__ startPosPtr,
    const float               base,
    const int                 writeOffsetStride,
    const int                 sec0,
    const int                 sec1,
    const int                 sec2,
    const int                 sec3)
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
    const float pos      = (float)(startPos + p);
    const float posAxis[4] = { pos, pos, pos, pos };

    const int sectDims = sec0 + sec1 + sec2 + sec3;
    float posSel = posAxis[0];
    if (sectDims > 0) {
        const int sector = i % sectDims;
        if (sector % 3 == 1 && sector < 3 * sec1) {
            posSel = posAxis[1];
        } else if (sector % 3 == 2 && sector < 3 * sec2) {
            posSel = posAxis[2];
        } else if (sector % 3 == 0 && sector < 3 * sec0) {
            posSel = posAxis[0];
        } else {
            posSel = posAxis[3];
        }
    }

    float* x = x_base + (size_t)startPos * (size_t)writeOffsetStride;
    const float invDim = 1.0f / (float)headDim;
    const float freq   = powf(base, -(float)(2 * i) * invDim);
    const float theta  = posSel * freq;
    const float c      = cosf(theta);
    const float s      = sinf(theta);

    const int headBase = (p * numHeads + h) * headDim;
    const float a = x[headBase + i];
    const float b = x[headBase + i + halfDim];
    x[headBase + i]           = a * c - b * s;
    x[headBase + i + halfDim] = a * s + b * c;
}