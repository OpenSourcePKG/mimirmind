// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Scatter this decode step's K/V rows into their paged-KV pool slots using
// DEVICE block/slot indices, so the write is CUDA-graph-capturable: the paged
// slot address is computed on-device from writeBlockId[seq]/writeSlot[seq]
// (persistent device buffers updated OUTSIDE the graph each step) instead of
// host-side (which would bake the capture step's destination address and make
// replay write every token to the same slot).
//
//   off = (writeBlockId[seq] * blockSize + writeSlot[seq]) * width
//   kPool[off + j] = kProj[seq*width + j]   (same for V)
//
// kPool/vPool: flat [numBlocks * blockSize * width] pool for ONE layer.
// kProj/vProj: [nSeq, width] this step's K/V, one row per sequence.
// Launch: grid.x = nSeq (one block per sequence), block.x = min(width, 256).

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

extern "C" __global__ void kv_write_tokens_batched(
    const float*        __restrict__ kProj,        // [nSeq, width]
    const float*        __restrict__ vProj,        // [nSeq, width]
    const unsigned int* __restrict__ writeBlockId, // [nSeq]
    const int*          __restrict__ writeSlot,    // [nSeq]
          float*        __restrict__ kPool,
          float*        __restrict__ vPool,
    const int                        nSeq,
    const int                        blockSize,
    const int                        width,
    const unsigned char* __restrict__ activeMask)   // 5.21-I: nullptr => all active
{
    const int seq = static_cast<int>(blockIdx.x);
    if (seq >= nSeq) {
        return;
    }
    if (activeMask != nullptr && activeMask[seq] == 0) return;   // 5.21-I: freeze
    const size_t off =
        (static_cast<size_t>(writeBlockId[seq]) * static_cast<size_t>(blockSize)
         + static_cast<size_t>(writeSlot[seq])) * static_cast<size_t>(width);
    const float* __restrict__ ks = kProj + static_cast<size_t>(seq) * width;
    const float* __restrict__ vs = vProj + static_cast<size_t>(seq) * width;
    for (int j = static_cast<int>(threadIdx.x); j < width;
         j += static_cast<int>(blockDim.x)) {
        kPool[off + j] = ks[j];
        vPool[off + j] = vs[j];
    }
}

// FP16 pool variant (5.14 I1): same scatter, but the K/V rows are still the
// F32 projections and are cast to __half on store (half the KV bytes). Only
// the pool pointers change type; the block/slot index math is identical.
extern "C" __global__ void kv_write_tokens_batched_fp16(
    const float*        __restrict__ kProj,        // [nSeq, width]
    const float*        __restrict__ vProj,        // [nSeq, width]
    const unsigned int* __restrict__ writeBlockId, // [nSeq]
    const int*          __restrict__ writeSlot,    // [nSeq]
          __half*       __restrict__ kPool,
          __half*       __restrict__ vPool,
    const int                        nSeq,
    const int                        blockSize,
    const int                        width,
    const unsigned char* __restrict__ activeMask)   // 5.21-I: nullptr => all active
{
    const int seq = static_cast<int>(blockIdx.x);
    if (seq >= nSeq) {
        return;
    }
    if (activeMask != nullptr && activeMask[seq] == 0) return;   // 5.21-I: freeze
    const size_t off =
        (static_cast<size_t>(writeBlockId[seq]) * static_cast<size_t>(blockSize)
         + static_cast<size_t>(writeSlot[seq])) * static_cast<size_t>(width);
    const float* __restrict__ ks = kProj + static_cast<size_t>(seq) * width;
    const float* __restrict__ vs = vProj + static_cast<size_t>(seq) * width;
    for (int j = static_cast<int>(threadIdx.x); j < width;
         j += static_cast<int>(blockDim.x)) {
        kPool[off + j] = __float2half(ks[j]);
        vPool[off + j] = __float2half(vs[j]);
    }
}

// FP8/E4M3 pool variant (5.16 capacity tier): same scatter, but the F32 K/V
// rows are cast to __nv_fp8_e4m3 on store (quarter the KV bytes vs F32). Only
// the pool pointer type changes; the block/slot index math is identical. E4M3
// is unscaled here (kv_scale == 1.0, mirroring vLLM's default) — the lossy
// cast is the single quality-gated step; a per-tensor scale is a follow-up if
// the DE-goldset gate regresses.
extern "C" __global__ void kv_write_tokens_batched_fp8(
    const float*        __restrict__ kProj,        // [nSeq, width]
    const float*        __restrict__ vProj,        // [nSeq, width]
    const unsigned int* __restrict__ writeBlockId, // [nSeq]
    const int*          __restrict__ writeSlot,    // [nSeq]
          __nv_fp8_e4m3* __restrict__ kPool,
          __nv_fp8_e4m3* __restrict__ vPool,
    const int                        nSeq,
    const int                        blockSize,
    const int                        width,
    const unsigned char* __restrict__ activeMask)   // 5.21-I: nullptr => all active
{
    const int seq = static_cast<int>(blockIdx.x);
    if (seq >= nSeq) {
        return;
    }
    if (activeMask != nullptr && activeMask[seq] == 0) return;   // 5.21-I: freeze
    const size_t off =
        (static_cast<size_t>(writeBlockId[seq]) * static_cast<size_t>(blockSize)
         + static_cast<size_t>(writeSlot[seq])) * static_cast<size_t>(width);
    const float* __restrict__ ks = kProj + static_cast<size_t>(seq) * width;
    const float* __restrict__ vs = vProj + static_cast<size_t>(seq) * width;
    for (int j = static_cast<int>(threadIdx.x); j < width;
         j += static_cast<int>(blockDim.x)) {
        kPool[off + j] = __nv_fp8_e4m3(ks[j]);
        vPool[off + j] = __nv_fp8_e4m3(vs[j]);
    }
}
