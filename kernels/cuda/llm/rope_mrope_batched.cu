// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Batched interleaved multi-axis RoPE (IMRoPE), in-place — M-Cuda.Batch
// batched variant of rope_mrope. Processes nSeq sequences, each with its
// OWN x region and its OWN start position, in ONE launch. Math per
// (seq, position, head) is byte-identical to the single-sequence kernel;
// only a per-sequence offset (blockIdx.y) is added — the x region via
// `xSeqStride` and the position via `startPosPtr[seq]`. Cat B of the
// hybrid batch-dim audit 2026-07-24 (per-sequence RoPE start position).
//
// NOTE: the per-sequence x layout (`xSeqStride`) is a provisional contract
// — the final batched KV-cache layout is settled in Phase D. The kernel
// math is what the parity test locks down.
//
// Layout:
//   x_base      : nSeq regions, region s at x_base + s*xSeqStride (floats)
//   startPosPtr : [nSeq] per-sequence start positions
// Launch: grid = dim3(ceil(seqLen*numHeads*halfDim / ROPE_LOCAL), nSeq, 1),
//         block = ROPE_LOCAL.

#include <cuda_runtime.h>

#ifndef ROPE_LOCAL
#define ROPE_LOCAL 256
#endif

extern "C" __global__ __launch_bounds__(ROPE_LOCAL)
void rope_mrope_batched(
    float*       __restrict__ x_base,
    const int                 seqLen,
    const int                 numHeads,
    const int                 headDim,
    const int*   __restrict__ startPosPtr,
    const float               base,
    const int                 writeOffsetStride,
    const int                 xSeqStride,
    const int                 sec0,
    const int                 sec1,
    const int                 sec2,
    const int                 sec3)
{
    const int seq     = blockIdx.y;
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

    const int   startPos = startPosPtr[seq];
    const float pos      = (float)(startPos + p);
    const float posAxis[4] = { pos, pos, pos, pos };

    // PARTIAL ROTARY — see rope_mrope.cu for the full derivation. sectDims =
    // rotary_dim/2; only the first sectDims pairs rotate, partner at +sectDims,
    // freq denom = rotary_dim = 2*sectDims. No-op for full-rotary (sectDims ==
    // halfDim). Matches HF Qwen3-Next/Qwen3.5-MoE partial_rotary_factor.
    const int sectDims = sec0 + sec1 + sec2 + sec3;
    const int rotPairs = (sectDims > 0) ? sectDims : halfDim;
    if (i >= rotPairs) {
        return;  // outside the rotary block -> pass through untouched
    }

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

    float* x = x_base + (size_t)seq * (size_t)xSeqStride
                      + (size_t)startPos * (size_t)writeOffsetStride;
    const int   rotaryDim = 2 * rotPairs;
    const float invDim = 1.0f / (float)rotaryDim;
    const float freq   = powf(base, -(float)(2 * i) * invDim);
    const float theta  = posSel * freq;
    const float c      = cosf(theta);
    const float s      = sinf(theta);

    const int headBase = (p * numHeads + h) * headDim;
    const float a = x[headBase + i];
    const float b = x[headBase + i + rotPairs];
    x[headBase + i]            = a * c - b * s;
    x[headBase + i + rotPairs] = a * s + b * c;
}
