// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// De-interleave a fused per-head [block_a | block_b] projection into two
// contiguous buffers. HIP port of kernels/split_head_pair.cl — see that
// file for the Qwen3-Next fused query + output-gate layout.
//
//   src : [seqLen, numHeads, 2, headDim]
//   a   : [seqLen, numHeads,    headDim]  <- src[..., 0, ...]
//   b   : [seqLen, numHeads,    headDim]  <- src[..., 1, ...]
//
// Launch:
//   dim3 grid ( ceil(seqLen*numHeads*headDim / SPLIT_HEAD_PAIR_LOCAL), 1, 1 )
//   dim3 block( SPLIT_HEAD_PAIR_LOCAL, 1, 1 )

#include <hip/hip_runtime.h>

#ifndef SPLIT_HEAD_PAIR_LOCAL
#define SPLIT_HEAD_PAIR_LOCAL 256
#endif

extern "C" __global__ __launch_bounds__(SPLIT_HEAD_PAIR_LOCAL)
void split_head_pair(
    const float* __restrict__ src,
    float*       __restrict__ a,
    float*       __restrict__ b,
    const int                 seqLen,
    const int                 numHeads,
    const int                 headDim)
{
    const int gid   = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = seqLen * numHeads * headDim;
    if (gid >= total) {
        return;
    }

    const int d  = gid % headDim;
    const int hp = gid / headDim;
    const int h  = hp % numHeads;
    const int p  = hp / numHeads;

    const size_t headPair = ((size_t)p * numHeads + h) * (size_t)(2 * headDim);
    a[gid] = src[headPair + d];
    b[gid] = src[headPair + headDim + d];
}