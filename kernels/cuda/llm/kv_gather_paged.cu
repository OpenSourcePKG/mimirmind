// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Gather one sequence's paged K (or V) prefix [0, Tkv) into a contiguous
// [Tkv, nKvHeads, headSize] FLOAT buffer — the cuDNN-SDPA prefill path needs
// contiguous f32 K/V, but the serving KV lives block-scattered in the paged pool
// ([num_blocks, block_size, num_kv_heads, head_size]) in the pool's element dtype
// (f32 / fp16 / fp8-e4m3). The gather DEQUANTISES to f32 on the fly (the cast is
// identity for f32, __half2float for fp16, E4M3->float for fp8), so cuDNN always
// sees f32. Bandwidth-trivial (<1% of the cuDNN attention).
//
// bt = the block-table slice for this sequence ([maxBlocksPerSeq]); position p
// maps to physical (blk = bt[p / block_size], slot = p % block_size).

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace {
template <typename KVT>
__device__ __forceinline__ void gatherBody(
    const KVT* __restrict__ pool, const int* __restrict__ bt,
    float* __restrict__ dst, int Tkv, int nKvHeads, int headSize, int blockSize)
{
    const long i     = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long total = static_cast<long>(Tkv) * nKvHeads * headSize;
    if (i >= total) {
        return;
    }
    const int d   = static_cast<int>(i % headSize);
    const int hkv = static_cast<int>((i / headSize) % nKvHeads);
    const int p   = static_cast<int>(i / (static_cast<long>(headSize) * nKvHeads));
    const int blk  = bt[p / blockSize];
    const int slot = p % blockSize;
    const long src = (static_cast<long>(blk * blockSize + slot) * nKvHeads + hkv)
                     * headSize + d;
    dst[i] = static_cast<float>(pool[src]);
}
} // namespace

extern "C" __global__ void kv_gather_paged(
    const float* __restrict__ pool, const int* __restrict__ bt,
    float* __restrict__ dst, int Tkv, int nKvHeads, int headSize, int blockSize) {
    gatherBody<float>(pool, bt, dst, Tkv, nKvHeads, headSize, blockSize);
}

extern "C" __global__ void kv_gather_paged_fp16(
    const void* __restrict__ pool, const int* __restrict__ bt,
    float* __restrict__ dst, int Tkv, int nKvHeads, int headSize, int blockSize) {
    gatherBody<__half>(reinterpret_cast<const __half*>(pool), bt, dst,
                       Tkv, nKvHeads, headSize, blockSize);
}

extern "C" __global__ void kv_gather_paged_fp8(
    const void* __restrict__ pool, const int* __restrict__ bt,
    float* __restrict__ dst, int Tkv, int nKvHeads, int headSize, int blockSize) {
    gatherBody<__nv_fp8_e4m3>(reinterpret_cast<const __nv_fp8_e4m3*>(pool), bt, dst,
                              Tkv, nKvHeads, headSize, blockSize);
}
