// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Batched chunked GatedDeltaNet prefill stage K0 — M-Cuda.Batch variant of
// deltanet_chunk_cumgate. Prefix-sums gLog within each chunk for nSeq
// sequences in ONE launch. Math per (seq, head, chunk) is byte-identical to
// the single-sequence kernel; a per-sequence offset (blockIdx.y) is added.
//
// Layout (per-sequence stride = T*H): gLog,gCum [nSeq, T, H].
// Launch: grid = dim3(ceil(H*nChunks / LOCAL), nSeq, 1), block = LOCAL.

#include <cuda_runtime.h>

#ifndef DELTANET_CUMGATE_LOCAL
#define DELTANET_CUMGATE_LOCAL 256
#endif

extern "C" __global__ __launch_bounds__(DELTANET_CUMGATE_LOCAL)
void deltanet_chunk_cumgate_batched(
    const float* __restrict__ gLog,
          float* __restrict__ gCum,
    const int                 T,
    const int                 H,
    const int                 C)
{
    const int seq     = blockIdx.y;
    const int nChunks = (T + C - 1) / C;
    const int idx     = blockIdx.x * blockDim.x + threadIdx.x;  // h*nChunks + chunk
    if (idx >= H * nChunks) {
        return;
    }
    const size_t gateStride = (size_t)T * H;
    const float* __restrict__ gLogS = gLog + (size_t)seq * gateStride;
    float*       __restrict__ gCumS = gCum + (size_t)seq * gateStride;

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
        run += gLogS[row];
        gCumS[row] = run;
    }
}
