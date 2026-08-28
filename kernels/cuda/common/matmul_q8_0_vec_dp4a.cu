// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// DP4A matvec for Q8_0 weights with pre-quantised int8 activation —
// warp32 revision (one full 32-lane warp per output row), per-block
// activation scale (produced by x_quant_q8_1_blocks) instead of one
// scale for the whole row — matches llama.cpp's vec_dot_q8_0_q8_1_impl
// (d8_0 × d8_1 per 32-element block), avoiding one row-wide outlier
// crushing precision for the rest of the row.
//
//   Y[n] = sum_{b=0..nBlocks-1}
//              d[n,b] × Xscale[b] × sum_{i=0..31}( Xq[b*32+i] × Wq[n, b, i] )
//
//   Xq:     [K]        int8  (produced by x_quant_q8_1_blocks)
//   Xscale: [K/32]     F32   per-block activation scale
//   W:      [N, K]     Q8_0  (K/32 blocks of 34 B: fp16 d + int8 qs[32])
//   Y:      [N]        F32
//
// Prior revision split each 32-lane hardware warp into two 16-lane
// sub-groups computing TWO DIFFERENT output rows in lockstep (ported from
// the RDNA3/gfx1101 sub-group-16 geometry via the L0/HIP siblings). On
// GB10 that halves effective memory coalescing: the two half-warps issue
// loads against two unrelated weight rows every step instead of one warp
// reading one contiguous row region, which measured at only ~7-8% of the
// 273 GB/s roofline (see lesson moe-fuseddown-toggle-neutral-gb10 +
// q8_0-gemv-gb10-two-negative-tuning-attempts). This revision follows
// llama.cpp's ggml-cuda mmvq.cu design instead: one full 32-lane warp
// commits to a single output row, so every load in a warp-step targets
// the SAME row and is coalesced across contiguous blocks.
//
// Geometry: 4 Q8_0 blocks (128 elements) processed per iteration — 8
// lanes per block (matching the 8 char4 chunks in a 32-element block) ×
// 4 blocks = the full 32-lane warp, all reading one row's next 136 B.
// A tail of 1-3 blocks (K not a multiple of 128) is guarded per-lane.
//
// Final reduction is a plain warp32 __shfl_xor_sync tree — every lane
// holds a partial sum over disjoint blocks of the SAME row, so summing
// all 32 lanes gives the exact row total (same math as before, wider
// reduction).

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifndef MATMUL_Q8_0_DP4A_LOCAL
#define MATMUL_Q8_0_DP4A_LOCAL 128
#endif

#define MATMUL_Q8_0_DP4A_WARP 32
#define MATMUL_Q8_0_DP4A_OUTPUTS_PER_GROUP \
    (MATMUL_Q8_0_DP4A_LOCAL / MATMUL_Q8_0_DP4A_WARP)

#define Q8_0_BLOCK_ELEMENTS 32
#define Q8_0_BLOCK_BYTES    34
#define Q8_0_BLOCK_CHAR4S   (Q8_0_BLOCK_ELEMENTS / 4)  // 8
#define Q8_0_BLOCKS_PER_ITER (MATMUL_Q8_0_DP4A_WARP / Q8_0_BLOCK_CHAR4S) // 4

// 1024 x elements = 32 blocks = 1 KiB SLM per workgroup (int8).
#define X_TILE_ELEMENTS 1024

static __device__ __forceinline__ float warp32_reduce_sum(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 16, 32);
    v += __shfl_xor_sync(0xffffffffu, v, 8, 32);
    v += __shfl_xor_sync(0xffffffffu, v, 4, 32);
    v += __shfl_xor_sync(0xffffffffu, v, 2, 32);
    v += __shfl_xor_sync(0xffffffffu, v, 1, 32);
    return v;
}

// Pack 4 int8 bytes into one int32 (little-endian) for __dp4a. Byte-wise
// on purpose: the Q8_0 qs payload starts at block offset 2 (after the fp16
// scale), so weight pointers are NOT 4-byte aligned and a plain int load
// faults on CUDA ("misaligned address").
static __device__ __forceinline__ int load_char4_as_int(const signed char* p) {
    return  static_cast<int>(static_cast<unsigned char>(p[0]))
         | (static_cast<int>(static_cast<unsigned char>(p[1])) << 8)
         | (static_cast<int>(static_cast<unsigned char>(p[2])) << 16)
         | (static_cast<int>(static_cast<unsigned char>(p[3])) << 24);
}

extern "C" __global__ __launch_bounds__(MATMUL_Q8_0_DP4A_LOCAL)
void matmul_q8_0_vec_dp4a(
    const signed char*   __restrict__ Xq,
    const float*         __restrict__ Xscale,   // [K/32], one per block
    const unsigned char* __restrict__ W,
          float*         __restrict__ Y,
    const int                         K,
    const int                         N)
{
    __shared__ signed char xTile[X_TILE_ELEMENTS];
    __shared__ float       xScaleTile[X_TILE_ELEMENTS / Q8_0_BLOCK_ELEMENTS];

    const int  wg      = blockIdx.x;
    const int  tid     = threadIdx.x;
    const int  lsize   = blockDim.x;
    const int  warpInWg = tid / MATMUL_Q8_0_DP4A_WARP;
    const int  lane     = tid % MATMUL_Q8_0_DP4A_WARP;
    const int  n        = wg * MATMUL_Q8_0_DP4A_OUTPUTS_PER_GROUP + warpInWg;
    const bool active    = (n < N);
    const int  nBlocks   = K / Q8_0_BLOCK_ELEMENTS;

    // Lane -> (block-in-group, char4-index) for the 4-blocks-per-iter,
    // full-warp-per-row layout.
    const int laneBlockOff = lane / Q8_0_BLOCK_CHAR4S;   // 0..3
    const int laneChar4Idx = lane % Q8_0_BLOCK_CHAR4S;   // 0..7

    float sum = 0.0f;

    for (int tile = 0; tile < K; tile += X_TILE_ELEMENTS) {
        const int tileK = (X_TILE_ELEMENTS < K - tile)
                            ? X_TILE_ELEMENTS : (K - tile);
        for (int i = tid; i < tileK; i += lsize) {
            xTile[i] = Xq[tile + i];
        }
        const int tileBlocks = tileK / Q8_0_BLOCK_ELEMENTS;
        for (int i = tid; i < tileBlocks; i += lsize) {
            xScaleTile[i] = Xscale[tile / Q8_0_BLOCK_ELEMENTS + i];
        }
        __syncthreads();

        if (active) {
            const unsigned char* __restrict__ row =
                W + static_cast<size_t>(n) * static_cast<size_t>(nBlocks)
                  * static_cast<size_t>(Q8_0_BLOCK_BYTES);

            const int blockStart   = tile / Q8_0_BLOCK_ELEMENTS;
            const int blocksInTile = X_TILE_ELEMENTS / Q8_0_BLOCK_ELEMENTS;
            const int blockEnd     = (blockStart + blocksInTile < nBlocks)
                                       ? (blockStart + blocksInTile)
                                       : nBlocks;

            for (int b = blockStart; b < blockEnd; b += Q8_0_BLOCKS_PER_ITER) {
                const int bMy = b + laneBlockOff;
                if (bMy < blockEnd) {
                    const unsigned char* __restrict__ block =
                        row + bMy * Q8_0_BLOCK_BYTES;
                    const float d =
                        __half2float(*reinterpret_cast<const __half*>(block));
                    const signed char* wq_ptr =
                        reinterpret_cast<const signed char*>(block + 2);

                    const int localBlock = bMy - blockStart;
                    const int xLocalBase = localBlock * Q8_0_BLOCK_ELEMENTS;
                    const signed char* xq_ptr = xTile + xLocalBase;
                    const float xScale = xScaleTile[localBlock];

                    const int wq_packed =
                        load_char4_as_int(wq_ptr + laneChar4Idx * 4);
                    const int xq_packed =
                        load_char4_as_int(xq_ptr + laneChar4Idx * 4);
                    const int dp = __dp4a(wq_packed, xq_packed, 0);

                    sum = __fmaf_rn(static_cast<float>(dp),
                                    d * xScale, sum);
                }
            }
        }

        __syncthreads();
    }

    sum = warp32_reduce_sum(sum);

    if (active && lane == 0) {
        Y[n] = sum;
    }
}
