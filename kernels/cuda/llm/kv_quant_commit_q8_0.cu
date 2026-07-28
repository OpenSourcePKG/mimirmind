// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
// Ported from kernels_hip/kv_quant_commit_q8_0.hip — Track 4 mechanical port, no functional change intended.

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// FP32 → Q8_0 KV write kernel. Consumes an fp32 workspace holding
// `T` fresh rows of K or V (one per new token, kvDim elements each,
// post-rmsnorm + post-RoPE), splits every row into 32-element blocks
// and writes them into the Q8_0 KV cache slot starting at row
// `curLenPtr[0]`.
//
// Block layout (34 B, matches ggml block_q8_0):
//   [0..1]   fp16 scale = absmax / 127
//   [2..33]  int8 quants = round(srcBlock[i] / scale) clamped to
//            [-127, 127]
//
// Zero-input block: scale=0 stored and all quants=0 — matches the CPU
// reference at src/compute/quant/Q8_0.cpp:79. Dequant round-trips to
// zero, which is the correct behaviour for an all-zero KV row.
//
// New kernel-class characteristic: FULL wave on RDNA3 (WgSize=32 ==
// warpSize). Every lane in the wave is active — no `width=16`
// scoping in the reduction, straight 5-round tree reduction in SLM.
//
// Launch geometry:
//   dim3 grid ( T, nBlocksPerRow, 1 )
//   dim3 block( KV_QUANT_COMMIT_LOCAL, 1, 1 )     // 32 == full wave

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifndef KV_QUANT_COMMIT_LOCAL
#define KV_QUANT_COMMIT_LOCAL 32
#endif

#define Q8_0_BLOCK_ELEMENTS 32
#define Q8_0_BLOCK_BYTES    34

extern "C" __global__ __launch_bounds__(KV_QUANT_COMMIT_LOCAL)
void kv_quant_commit_q8_0(
    const float*    __restrict__ xSrc,        // [T, kvDim] fp32 workspace
          unsigned char* __restrict__ kvDst,  // Q8_0 cache slot base
    const int                    kvDim,       // must be multiple of 32
    const int*      __restrict__ curLenPtr)   // startPos (= cache.length())
{
    const int t   = blockIdx.x;
    const int blk = blockIdx.y;
    const int lid = threadIdx.x;

    const float* __restrict__ srcRow =
        xSrc + static_cast<size_t>(t) * static_cast<size_t>(kvDim);
    const float xv = srcRow[blk * Q8_0_BLOCK_ELEMENTS + lid];

    // Per-block absmax reduction. Tree in SLM — every thread writes
    // its |x|, then log2(32) = 5 rounds of pairwise fmaxf. All 32
    // lanes of the wave participate, no width scoping needed.
    __shared__ float scratch[KV_QUANT_COMMIT_LOCAL];
    scratch[lid] = fabsf(xv);
    __syncthreads();
    for (int s = KV_QUANT_COMMIT_LOCAL >> 1; s > 0; s >>= 1) {
        if (lid < s) {
            scratch[lid] = fmaxf(scratch[lid], scratch[lid + s]);
        }
        __syncthreads();
    }
    const float absMax   = scratch[0];
    const float scale    = (absMax > 0.0f) ? (absMax * (1.0f / 127.0f)) : 0.0f;
    const float invScale = (absMax > 0.0f) ? (127.0f / absMax) : 0.0f;

    const int startPos      = curLenPtr[0];
    const int nBlocksPerRow = kvDim / Q8_0_BLOCK_ELEMENTS;
    const size_t rowBase =
        static_cast<size_t>(startPos + t)
      * static_cast<size_t>(nBlocksPerRow)
      * static_cast<size_t>(Q8_0_BLOCK_BYTES);
    const size_t blkBase = rowBase
                         + static_cast<size_t>(blk)
                         * static_cast<size_t>(Q8_0_BLOCK_BYTES);
    unsigned char* __restrict__ blkPtr = kvDst + blkBase;

    // Lane 0 writes the fp16 scale into the 2-byte header. __float2half
    // uses IEEE-754 round-to-nearest-even, matching vstore_half in the
    // OpenCL kernel and floatToHalf in the CPU reference.
    if (lid == 0) {
        *reinterpret_cast<__half*>(blkPtr) = __float2half(scale);
    }

    // All 32 lanes write their own int8 quant. `roundf` on HIP is C99
    // round-half-away-from-zero, matching OpenCL `round()` and
    // std::lround used by Q8_0::quantizeRow.
    const float qf = roundf(xv * invScale);
    const float qc = fminf(fmaxf(qf, -127.0f), 127.0f);
    reinterpret_cast<signed char*>(blkPtr)[2 + lid] =
        static_cast<signed char>(static_cast<int>(qc));
}