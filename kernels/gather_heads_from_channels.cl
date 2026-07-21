// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

// Per-head channel slice + optional GQA head repeat — turns the fused
// Qwen3-Next conv output into contiguous q / k / v for the delta net.
//   dst[t,hd,s] = src[t*convTotalWidth + offset + (hd % srcHeads)*S + s]
// dst is [T, dstHeads, S]. CPU reference: compute::gatherHeadsFromChannels.
// Launch: 1D global = T*dstHeads*S.

#ifndef GATHER_HEADS_LOCAL
#define GATHER_HEADS_LOCAL 256
#endif

__attribute__((reqd_work_group_size(GATHER_HEADS_LOCAL, 1, 1)))
__kernel void gather_heads_from_channels(
    __global const float* src,
    __global       float* dst,
    const int             T,
    const int             offset,
    const int             srcHeads,
    const int             dstHeads,
    const int             S,
    const int             convTotalWidth)
{
    const int gid   = (int)get_global_id(0);
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
