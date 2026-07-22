// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Chunked GatedDeltaNet prefill — stage K0: cumulative decay gate.
//
// Inclusive prefix-sum of gLog within each chunk, per head:
//   G[c0+a, h] = sum_{a'=0..a} gLog[c0+a', h]
// gLog / gCum are [T, H] (row = token, col = head). Chunk size C
// (default 64); the last chunk may be partial. Matches
// compute::deltanetChunkCumGate (GatedDeltaNet.cpp).
//
// One thread per (head, chunk): the scan is only <=C sequential adds and
// every (head, chunk) is independent. Grid = ceil(H*nChunks / LOCAL).

#include <cuda_runtime.h>

// Must match GpuOps kElementwiseLocalSize (the launch block dim); a smaller
// __launch_bounds__ than the launch block size faults cuLaunchKernel.
#ifndef DELTANET_CUMGATE_LOCAL
#define DELTANET_CUMGATE_LOCAL 256
#endif

extern "C" __global__ __launch_bounds__(DELTANET_CUMGATE_LOCAL)
void deltanet_chunk_cumgate(
    const float* __restrict__ gLog,
          float* __restrict__ gCum,
    const int                 T,
    const int                 H,
    const int                 C)
{
    const int nChunks = (T + C - 1) / C;
    const int idx     = blockIdx.x * blockDim.x + threadIdx.x;  // h*nChunks + chunk
    if (idx >= H * nChunks) {
        return;
    }
    const int h     = idx / nChunks;
    const int chunk = idx % nChunks;
    const int c0    = chunk * C;
    int cs = C;
    if (c0 + cs > T) {
        cs = T - c0;
    }

    float run = 0.0f;
    for (int a = 0; a < cs; ++a) {
        const int row = (c0 + a) * H + h;
        run += gLog[row];
        gCum[row] = run;
    }
}