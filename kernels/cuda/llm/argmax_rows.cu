// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Per-row argmax over the vocab dimension. One block per row; block-reduce to
// (max value, min index) so ties break to the LOWEST index — byte-identical to
// the host scan `for v: if logits[v] > best { best = logits[v]; idx = v; }`
// (strict >, first occurrence kept). Used by the MTP draft/verify to keep the
// argmax on the device and read back only the nRows token ids (a few bytes)
// instead of the full nRows*vocab logits (tens of MB per round).
//
// Launch: grid = nRows, block = ARGMAX_ROWS_LOCAL (256).

#include <cuda_runtime.h>
#include <math_constants.h>

#ifndef ARGMAX_ROWS_LOCAL
#define ARGMAX_ROWS_LOCAL 256
#endif

extern "C" __global__ __launch_bounds__(ARGMAX_ROWS_LOCAL)
void argmax_rows(const float* __restrict__ logits,
                 int*         __restrict__ out,
                 const int                 nRows,
                 const int                 vocab)
{
    const int row = blockIdx.x;
    if (row >= nRows) {
        return;
    }
    const float* __restrict__ r = logits + static_cast<size_t>(row) * vocab;

    __shared__ float sV[ARGMAX_ROWS_LOCAL];
    __shared__ int   sI[ARGMAX_ROWS_LOCAL];

    float bestV = -CUDART_INF_F;
    int   bestI = 0;
    for (int v = threadIdx.x; v < vocab; v += blockDim.x) {
        const float x = r[v];
        // strict > keeps the first (lowest-index) occurrence of the max, since v
        // ascends within this thread's stride sweep.
        if (x > bestV) { bestV = x; bestI = v; }
    }
    sV[threadIdx.x] = bestV;
    sI[threadIdx.x] = bestI;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const float ov = sV[threadIdx.x + stride];
            const int   oi = sI[threadIdx.x + stride];
            // lowest-index tie-break: take the other only if strictly greater,
            // or equal value with a smaller index.
            if (ov > sV[threadIdx.x] ||
                (ov == sV[threadIdx.x] && oi < sI[threadIdx.x])) {
                sV[threadIdx.x] = ov;
                sI[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[row] = sI[0];
    }
}
