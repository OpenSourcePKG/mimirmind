// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Per-head channel slice + optional GQA head repeat. CUDA port of
// kernels/gather_heads_from_channels.cl. CPU reference:
// compute::gatherHeadsFromChannels. Launch: grid ceil(T*dstHeads*S/L), block L.

#include <cuda_runtime.h>

#ifndef GATHER_HEADS_LOCAL
#define GATHER_HEADS_LOCAL 256
#endif

extern "C" __global__ __launch_bounds__(GATHER_HEADS_LOCAL)
void gather_heads_from_channels(
    const float* __restrict__ src,
    float*       __restrict__ dst,
    const int                 T,
    const int                 offset,
    const int                 srcHeads,
    const int                 dstHeads,
    const int                 S,
    const int                 convTotalWidth)
{
    const int gid   = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = T * dstHeads * S;
    if (gid >= total) {
        return;
    }
    const int s   = gid % S;
    const int rem = gid / S;
    const int hd  = rem % dstHeads;
    const int t   = rem / dstHeads;
    const int srcHead = hd % srcHeads;
    dst[gid] = src[(size_t)t * convTotalWidth + offset + (size_t)srcHead * S + s];
}
