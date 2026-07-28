// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
// Ported from kernels_hip/rope_inplace_fp16.hip — Track 4 mechanical port, no functional change intended.

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// FP16-KV variant of rope_inplace.hip. K-cache slots are stored as
// __half; the rotation is done in fp32 in registers (__half2float →
// rotate → __float2half) so RoPE precision doesn't degrade from the
// storage swap. Layout / dispatch geometry is identical to the f32
// variant — only the pointer type of `x_base` differs.
//
// HIP port of kernels/rope_inplace_fp16.cl.
//
// Called by the K-rope path when `KvCache::dtype() == FP16`. The
// Q-rope path always targets the fp32 workspace and continues to
// use rope_inplace.hip regardless of KV dtype.
//
// Launch:
//   dim3 grid ( ceil(seqLen * numHeads * halfDim / ROPE_LOCAL), 1, 1 )
//   dim3 block( ROPE_LOCAL, 1, 1 )

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifndef ROPE_LOCAL
#define ROPE_LOCAL 256
#endif

extern "C" __global__ __launch_bounds__(ROPE_LOCAL)
void rope_inplace_fp16(
          __half* __restrict__    x_base,
    const int                     seqLen,
    const int                     numHeads,
    const int                     headDim,
    const int* __restrict__       startPosPtr,
    const float                   base,
    const int                     writeOffsetStride)
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
    __half*     x        =
        x_base + static_cast<size_t>(startPos)
               * static_cast<size_t>(writeOffsetStride);
    const float invDim = 1.0f / static_cast<float>(headDim);
    const float freq   = powf(base, -static_cast<float>(2 * i) * invDim);
    const float theta  = pos * freq;
    const float c      = cosf(theta);
    const float s      = sinf(theta);

    const int   headBase = (p * numHeads + h) * headDim;
    // Half-load → promote → rotate → demote → half-store. Rotation
    // stays fp32 in registers so precision matches the f32 kernel
    // up to the fp16 store round-trip on the two writes.
    const float a = __half2float(x[headBase + i]);
    const float b = __half2float(x[headBase + i + halfDim]);
    x[headBase + i]           = __float2half(a * c - b * s);
    x[headBase + i + halfDim] = __float2half(a * s + b * c);
}