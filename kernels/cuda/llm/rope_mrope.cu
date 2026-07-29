// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Interleaved multi-axis RoPE (IMRoPE), in-place. CUDA port of
// kernels/rope_mrope.cl — see that file for the full IMRoPE sector-rule
// derivation and the text-only-positions note. Qwen3-Next / Qwen3.5-VL
// full-attention layers (LLM_ROPE_TYPE_IMROPE). Bit-identical to
// rope_inplace for text (all four axis positions equal).
//
// Launch:
//   dim3 grid ( ceil(seqLen*numHeads*halfDim / ROPE_LOCAL), 1, 1 )
//   dim3 block( ROPE_LOCAL, 1, 1 )

#include <cuda_runtime.h>

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

    // PARTIAL ROTARY: sectDims = sum of the mRoPE sections = rotary_dim/2.
    // Only the first `sectDims` pairs (rotary_dim = 2*sectDims head dims) are
    // rotated; the remaining head dims pass through unchanged. For full-rotary
    // IMRoPE models sectDims == headDim/2 and this reduces to rotating every
    // pair, so the change is a no-op there. Matches HF Qwen3-Next/Qwen3.5-MoE:
    //   dim = int(head_dim * partial_rotary_factor);  inv_freq = base^(-2j/dim)
    //   q_rot = q[..., :dim]; q_pass = q[..., dim:]  (rotate_half over q_rot,
    //   i.e. the rotation partner sits at +dim/2 = +sectDims, NOT +halfDim).
    // sectDims (sum of the mRoPE sections) = rotary_dim/2 when sections are
    // provided; sectDims==0 means "no sections" -> plain full-head RoPE. So the
    // number of rotated pairs is sectDims, or halfDim in the no-sections case.
    const int sectDims = sec0 + sec1 + sec2 + sec3;
    const int rotPairs = (sectDims > 0) ? sectDims : halfDim;
    if (i >= rotPairs) {
        return;  // outside the rotary block -> pass through untouched
    }

    // Interleaved-mRoPE axis selection (collapses to a single axis for text,
    // where all four posAxis entries are equal). Only meaningful when sections
    // are present; i < rotPairs == sectDims there so sector == i.
    float posSel = posAxis[0];
    if (sectDims > 0) {
        const int sector = i;
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
    const int   rotaryDim = 2 * rotPairs;            // head_dim*partial_rotary
    const float invDim = 1.0f / (float)rotaryDim;    // freq denom is rotary_dim
    const float freq   = powf(base, -(float)(2 * i) * invDim);
    const float theta  = posSel * freq;
    const float c      = cosf(theta);
    const float s      = sinf(theta);

    const int headBase = (p * numHeads + h) * headDim;
    const float a = x[headBase + i];
    const float b = x[headBase + i + rotPairs];      // partner at +rotary_dim/2
    x[headBase + i]            = a * c - b * s;
    x[headBase + i + rotPairs] = a * s + b * c;
}