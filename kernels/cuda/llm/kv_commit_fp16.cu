// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// FP16-KV staging commit — the fp16 analogue of kv_quant_commit_q8_0. Models
// whose K/V projection is not fused-QKV (e.g. Qwen3-Next: separate K/V matmuls)
// cannot write straight into an fp16 cache slot, so they project K/V into an
// fp32 scratch, run rmsnorm + (m)rope in fp32 on that scratch, and finally
// commit each row into the fp16 cache with a single __float2half cast — the
// only lossy step, mirroring how Q8_0 defers the quantise to its commit.
//
// xSrc is the per-forward fp32 scratch [T, kvDim] (contiguous, row-major, no
// curLen offset). kvDst is the layer's fp16 cache base; the write lands at the
// curLen row offset: kvDst[curLen*kvDim + t*kvDim + i] = __float2half(xSrc[...]).
// writeOffset (= curLen) arrives through the shared curLen slot so kvDst stays
// a stable layer-base pointer across CUDA-graph replays.
//
// Launch:
//   dim3 grid ( ceil(T*kvDim / KV_COMMIT_FP16_LOCAL), 1, 1 )
//   dim3 block( KV_COMMIT_FP16_LOCAL, 1, 1 )

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifndef KV_COMMIT_FP16_LOCAL
#define KV_COMMIT_FP16_LOCAL 256
#endif

extern "C" __global__ __launch_bounds__(KV_COMMIT_FP16_LOCAL)
void kv_commit_fp16(
    const float* __restrict__ xSrc,        // [T, kvDim] fp32 scratch
          __half* __restrict__ kvDst,       // layer fp16 cache base
    const int                 T,
    const int                 kvDim,
    const int*   __restrict__ curLenPtr)    // device int slot (= curLen)
{
    const long gid   = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = (long)T * (long)kvDim;
    if (gid >= total) {
        return;
    }
    const size_t kvBase = (size_t)curLenPtr[0] * (size_t)kvDim;
    kvDst[kvBase + (size_t)gid] = __float2half(xSrc[gid]);
}
