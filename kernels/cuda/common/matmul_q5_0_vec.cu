// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
// Ported from kernels_hip/matmul_q5_0_vec.hip — Track 4 mechanical port, no functional change intended.

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Matrix-vector multiply with Q5_0 weights, on-the-fly dequant.
//
//   Y[n] = sum_{k=0..K-1} X[k] * dequant_q5_0(W, n, k)
//
//   X:  [K]     F32 dense vector (single token / M=1)
//   W:  [N, K]  Q5_0 — each row is (K/32) blocks of 22 bytes.
//   Y:  [N]     F32 dense vector
//
// Sibling of matmul_q8_0_vec.hip — identical launch geometry and warp
// layout (RDNA3 warpSize=32, one warp per output row, LOCAL=128 by
// default = 4 warps per WG). Only the per-lane dequant math differs:
// Q5_0 uses 5-bit signed quants split across a nibble array and a
// separate high-bit u32, both per block.
//
// Launch:
//   dim3 grid ( ceil(N / OUTPUTS_PER_GROUP), 1, 1 )
//   dim3 block( MATMUL_Q5_0_LOCAL, 1, 1 )
//
// Q5_0 block layout (matches ggml block_q5_0, bit-identical to
// llama.cpp dequantize_row_q5_0):
//   fp16    d        (2 B)  — block scale
//   uint32  qh       (4 B)  — one high-bit per element, little-endian
//                             (qh bit j → element j's bit 4)
//   uint8   qs[16]   (16 B) — packed nibbles; element j (j<16) uses
//                             low nibble of qs[j], element j+16 uses
//                             high nibble of qs[j]
//   value[j]      = d * (((qs[j]      & 0x0F) | ((qh >> j)      & 1) << 4) - 16)
//   value[j+16]   = d * (((qs[j] >>  4)        | ((qh >> (j+16)) & 1) << 4) - 16)

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifndef MATMUL_Q5_0_LOCAL
#define MATMUL_Q5_0_LOCAL 128
#endif

#define MATMUL_Q5_0_WARPS             (MATMUL_Q5_0_LOCAL / 32)
#define MATMUL_Q5_0_OUTPUTS_PER_GROUP MATMUL_Q5_0_WARPS

#define Q5_0_BLOCK_ELEMENTS 32
#define Q5_0_BLOCK_BYTES    22
#define X_TILE_ELEMENTS     1024

namespace {

// Full-warp inclusive reduction via shuffle-down. warpSize is 32 on
// RDNA3, so five rounds of halving cover the whole warp. Returns the
// full sum in lane 0.
__device__ __forceinline__ float warpReduceSum(float v) {
    v += __shfl_down_sync(0xffffffffu, v, 16);
    v += __shfl_down_sync(0xffffffffu, v,  8);
    v += __shfl_down_sync(0xffffffffu, v,  4);
    v += __shfl_down_sync(0xffffffffu, v,  2);
    v += __shfl_down_sync(0xffffffffu, v,  1);
    return v;
}

} // namespace

extern "C" __global__ __launch_bounds__(MATMUL_Q5_0_LOCAL)
void matmul_q5_0_vec(
    const float*         __restrict__ X,
    const unsigned char* __restrict__ W,
          float*         __restrict__ Y,
    const int                          K,
    const int                          N)
{
    __shared__ float xTile[X_TILE_ELEMENTS];

    const int wg      = blockIdx.x;
    const int tid     = threadIdx.x;
    const int lsize   = blockDim.x;
    const int warpId  = tid / 32;
    const int laneId  = tid % 32;
    const int n       = wg * MATMUL_Q5_0_OUTPUTS_PER_GROUP + warpId;
    const bool active = (n < N);
    const int nBlocks = K / Q5_0_BLOCK_ELEMENTS;

    // Precompute this lane's per-block dequant helpers. All 32 lanes
    // extract 5 bits from the block: lanes 0..15 read qs[l] low nibble
    // + qh bit l, lanes 16..31 read qs[l-16] high nibble + qh bit l.
    // qhShift is just laneId — the high-bit slot in the packed u32 is
    // element-indexed, low bits for elements 0..15, high bits for
    // elements 16..31.
    const int qsIdx     = laneId & 15;       // 0..15 for both halves
    const int nibbleHi  = (laneId >> 4) & 1; // 0 for l<16, 1 for l>=16

    float sum = 0.0f;

    for (int tile = 0; tile < K; tile += X_TILE_ELEMENTS) {
        const int tileK = min(X_TILE_ELEMENTS, K - tile);

        for (int i = tid; i < tileK; i += lsize) {
            xTile[i] = X[tile + i];
        }
        __syncthreads();

        if (active) {
            const unsigned char* row =
                W + static_cast<size_t>(n)
                  * static_cast<size_t>(nBlocks)
                  * Q5_0_BLOCK_BYTES;

            const int blockStart   = tile / Q5_0_BLOCK_ELEMENTS;
            const int blocksInTile = X_TILE_ELEMENTS / Q5_0_BLOCK_ELEMENTS;
            const int blockEnd     = min(blockStart + blocksInTile, nBlocks);

            for (int b = blockStart; b < blockEnd; ++b) {
                const unsigned char* block = row + b * Q5_0_BLOCK_BYTES;

                // Scale.
                const __half* d_ptr = reinterpret_cast<const __half*>(block);
                const float   d     = __half2float(d_ptr[0]);

                // qh: 4 little-endian bytes — one high-bit per element.
                // Unaligned load is safe (blocks are packed, no natural
                // 4-byte alignment guaranteed for the second block onward).
                unsigned int qh;
                qh  = static_cast<unsigned int>(block[2]);
                qh |= static_cast<unsigned int>(block[3]) << 8;
                qh |= static_cast<unsigned int>(block[4]) << 16;
                qh |= static_cast<unsigned int>(block[5]) << 24;

                // qs: 16 packed nibbles at bytes [6..21].
                const unsigned char* qs = block + 6;

                // Extract this lane's 5-bit signed value.
                const unsigned int nibble =
                    nibbleHi ? static_cast<unsigned int>(qs[qsIdx] >> 4)
                             : static_cast<unsigned int>(qs[qsIdx] & 0x0Fu);
                const unsigned int hiBit = (qh >> laneId) & 1u;
                const int x5 = static_cast<int>(nibble | (hiBit << 4)) - 16;

                const int xLocalBase = (b - blockStart) * Q5_0_BLOCK_ELEMENTS;
                const float xv = xTile[xLocalBase + laneId];
                const float dq = d * static_cast<float>(x5);
                sum = __fmaf_rn(xv, dq, sum);
            }
        }

        __syncthreads();
    }

    sum = warpReduceSum(sum);

    if (active && laneId == 0) {
        Y[n] = sum;
    }
}