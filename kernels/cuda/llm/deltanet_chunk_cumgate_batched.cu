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
//
// 5.21.9 ragged serving prefill: optional seqT/seqOff/activeMask (all
// nullptr => uniform T at stride seq*T*H, bit-identical to the original).
// With seqOff the activations are the serving RAGGED token-major pack:
// sequence seq's tokens start at token index seqOff[seq] and run for
// seqT[seq] tokens; T then only sizes the per-seq chunk-count ceiling
// (T = maxSeqT). A frozen slot (activeMask 0) is skipped entirely.

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
    const int                 C,
    const unsigned char* __restrict__ activeMask,   // nullptr => all active
    const int* __restrict__ seqT,                   // nullptr => uniform T
    const int* __restrict__ seqOff)                 // nullptr => seq*T
{
    const int seq = blockIdx.y;
    if (activeMask != nullptr && activeMask[seq] == 0) return;
    const int Tseq = (seqT != nullptr) ? seqT[seq] : T;
    const int nChunks = (Tseq + C - 1) / C;
    const int idx     = blockIdx.x * blockDim.x + threadIdx.x;  // h*nChunks + chunk
    if (idx >= H * nChunks) {
        return;
    }
    const size_t tokBase = (seqOff != nullptr) ? (size_t)seqOff[seq]
                                               : (size_t)seq * (size_t)T;
    const float* __restrict__ gLogS = gLog + tokBase * H;
    float*       __restrict__ gCumS = gCum + tokBase * H;

    const int h     = idx / nChunks;
    const int chunk = idx % nChunks;
    const int c0    = chunk * C;
    int cs = C;
    if (c0 + cs > Tseq) {
        cs = Tseq - c0;
    }

    float run = 0.0f;
    for (int a = 0; a < cs; ++a) {
        const int row = (c0 + a) * H + h;
        run += gLogS[row];
        gCumS[row] = run;
    }
}
