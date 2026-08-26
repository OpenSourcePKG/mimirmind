// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// FP8/E4M3-KV staging commit — the fp8 analogue of kv_commit_fp16 (5.16
// capacity tier). Models whose K/V projection is not fused-QKV cannot write
// straight into an fp8 cache slot, so they project K/V into an fp32 scratch,
// run rmsnorm + (m)rope in fp32 on that scratch, and finally commit each row
// into the fp8 cache with a single __nv_fp8_e4m3 cast — the only lossy step,
// mirroring how kv_commit_fp16 defers the __float2half cast. E4M3 is unscaled
// (kv_scale == 1.0, vLLM default); a per-tensor scale is a follow-up if the
// quality gate regresses.
//
// xSrc is the per-forward fp32 scratch [T, kvDim] (contiguous, row-major, no
// curLen offset). kvDst is the layer's fp8 cache base; the write lands at the
// curLen row offset: kvDst[curLen*kvDim + t*kvDim + i] = __nv_fp8_e4m3(xSrc[...]).
// writeOffset (= curLen) arrives through the shared curLen slot so kvDst stays
// a stable layer-base pointer across CUDA-graph replays.
//
// Launch:
//   dim3 grid ( ceil(T*kvDim / KV_COMMIT_FP8_LOCAL), 1, 1 )
//   dim3 block( KV_COMMIT_FP8_LOCAL, 1, 1 )

#include <cuda_runtime.h>
#include <cuda_fp8.h>

#ifndef KV_COMMIT_FP8_LOCAL
#define KV_COMMIT_FP8_LOCAL 256
#endif

extern "C" __global__ __launch_bounds__(KV_COMMIT_FP8_LOCAL)
void kv_commit_fp8(
    const float* __restrict__ xSrc,          // [T, kvDim] fp32 scratch
          __nv_fp8_e4m3* __restrict__ kvDst,  // layer fp8 cache base
    const int                 T,
    const int                 kvDim,
    const int*   __restrict__ curLenPtr)      // device int slot (= curLen)
{
    const long gid   = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = (long)T * (long)kvDim;
    if (gid >= total) {
        return;
    }
    const size_t kvBase = (size_t)curLenPtr[0] * (size_t)kvDim;
    kvDst[kvBase + (size_t)gid] = __nv_fp8_e4m3(xSrc[gid]);
}
