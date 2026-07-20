// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
// Ported from kernels_hip/matmul_q8_0_vec_dp4a.hip — Track 4 mechanical port, no functional change intended.

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// DP4A matvec for Q8_0 weights with pre-quantised int8 activation.
//
//   Y[n] = xScale × sum_{b=0..nBlocks-1}
//              d[b] × sum_{i=0..31}( Xq[b*32+i] × Wq[n, b, i] )
//
//   Xq:     [K]     int8  (produced by an x-quant kernel)
//   Xscale: scalar F32    (device pointer to a single float)
//   W:      [N, K]  Q8_0  (K/32 blocks of 34 B: fp16 d + int8 qs[32])
//   Y:      [N]     F32
//
// The inner accumulator stays in int32 across each 32-element block
// via a 4-way int8 dot, with only one fp32 multiply per block for
// d × xScale. That's 4× fewer FP32 muls in the hot loop compared to
// matmul_q8_0_vec.
//
// gfx1101 gotcha: RDNA3 (Navi 32) does NOT expose the pure signed×
// signed DP4A instruction. It requires `dot1-insts` which is a
// CDNA/gfx9 feature. gfx1101's dot-insts family only includes
// `v_dot4_i32_iu8` (signed × unsigned, via `dot8-insts`). Rather than
// rebias one side to make sudot4 work, the port uses a manual
// byte-wise expansion here — correct, portable, and the compiler is
// free to fuse it into v_dot4_i32_iu8 if profitable. Trading the DP4A
// speedup for correctness is the right call for the parity port; a
// follow-up perf pass can measure if the manual expansion is
// competitive or if the rebias-then-sudot4 route wins.
//
// Launch geometry matches matmul_q8_0_vec: WG=64, sg16 mapped
// explicitly (tid/16, tid%16), warp16_reduce_sum via
// `__shfl_xor(v, off, 16)`.
//
// With 16 lanes but only 8 char4 chunks per 32-element block, each
// outer iteration processes TWO consecutive Q8_0 blocks: lanes 0..7
// cover block `b`, lanes 8..15 cover block `b+1`.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifndef MATMUL_Q8_0_DP4A_LOCAL
#define MATMUL_Q8_0_DP4A_LOCAL 64
#endif

#ifndef MATMUL_Q8_0_DP4A_SG
#define MATMUL_Q8_0_DP4A_SG 16
#endif

#define MATMUL_Q8_0_DP4A_OUTPUTS_PER_GROUP \
    (MATMUL_Q8_0_DP4A_LOCAL / MATMUL_Q8_0_DP4A_SG)

#define Q8_0_BLOCK_ELEMENTS 32
#define Q8_0_BLOCK_BYTES    34
#define Q8_0_BLOCK_CHAR4S   (Q8_0_BLOCK_ELEMENTS / 4)  // 8

// 1024 x elements = 32 blocks = 1 KiB SLM per workgroup (int8, so
// 4× smaller than the fp32 tile in matmul_q8_0_vec).
#define X_TILE_ELEMENTS 1024

static __device__ __forceinline__ float warp16_reduce_sum(float v) {
    v += __shfl_xor(v, 8, 16);
    v += __shfl_xor(v, 4, 16);
    v += __shfl_xor(v, 2, 16);
    v += __shfl_xor(v, 1, 16);
    return v;
}

// Load 4 aligned int8 bytes as one packed int32.
static __device__ __forceinline__ int load_char4_as_int(const signed char* p) {
    return *reinterpret_cast<const int*>(p);
}

// Signed int8×4 dot product. gfx1101 lacks the pure sdot4 instruction
// (dot1-insts is CDNA-only). Manual expansion here — the compiler is
// free to reshape it into v_dot4_i32_iu8 with implicit bias
// compensation if that turns out faster on Navi 32.
static __device__ __forceinline__ int dot4_i8(int a, int b) {
    const signed char* pa = reinterpret_cast<const signed char*>(&a);
    const signed char* pb = reinterpret_cast<const signed char*>(&b);
    return static_cast<int>(pa[0]) * static_cast<int>(pb[0])
         + static_cast<int>(pa[1]) * static_cast<int>(pb[1])
         + static_cast<int>(pa[2]) * static_cast<int>(pb[2])
         + static_cast<int>(pa[3]) * static_cast<int>(pb[3]);
}

extern "C" __global__ __launch_bounds__(MATMUL_Q8_0_DP4A_LOCAL)
void matmul_q8_0_vec_dp4a(
    const signed char*   __restrict__ Xq,
    const float*         __restrict__ Xscale,
    const unsigned char* __restrict__ W,
          float*         __restrict__ Y,
    const int                         K,
    const int                         N)
{
    __shared__ signed char xTile[X_TILE_ELEMENTS];

    const int  wg      = blockIdx.x;
    const int  tid     = threadIdx.x;
    const int  lsize   = blockDim.x;
    const int  sgInWg  = tid / MATMUL_Q8_0_DP4A_SG;
    const int  sgLocal = tid % MATMUL_Q8_0_DP4A_SG;
    const int  n       = wg * MATMUL_Q8_0_DP4A_OUTPUTS_PER_GROUP + sgInWg;
    const bool active  = (n < N);
    const int  nBlocks = K / Q8_0_BLOCK_ELEMENTS;

    const float xScale = *Xscale;

    // Lane assignment for the 2-blocks-per-iteration pattern:
    //   lanes 0..7  → block b,   char4 index sgLocal
    //   lanes 8..15 → block b+1, char4 index sgLocal - 8
    const int laneBlockOff = sgLocal >> 3;
    const int laneChar4Idx = sgLocal & (Q8_0_BLOCK_CHAR4S - 1);

    float sum = 0.0f;

    for (int tile = 0; tile < K; tile += X_TILE_ELEMENTS) {
        const int tileK = (X_TILE_ELEMENTS < K - tile)
                            ? X_TILE_ELEMENTS : (K - tile);
        for (int i = tid; i < tileK; i += lsize) {
            xTile[i] = Xq[tile + i];
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

            for (int b = blockStart; b < blockEnd; b += 2) {
                const int bMy = b + laneBlockOff;
                if (bMy < blockEnd) {
                    const unsigned char* __restrict__ block =
                        row + bMy * Q8_0_BLOCK_BYTES;
                    const float d =
                        __half2float(*reinterpret_cast<const __half*>(block));
                    const signed char* wq_ptr =
                        reinterpret_cast<const signed char*>(block + 2);

                    const int xLocalBase =
                        (bMy - blockStart) * Q8_0_BLOCK_ELEMENTS;
                    const signed char* xq_ptr = xTile + xLocalBase;

                    const int wq_packed =
                        load_char4_as_int(wq_ptr + laneChar4Idx * 4);
                    const int xq_packed =
                        load_char4_as_int(xq_ptr + laneChar4Idx * 4);
                    const int dp = dot4_i8(wq_packed, xq_packed);

                    sum = __fmaf_rn(static_cast<float>(dp),
                                    d * xScale, sum);
                }
            }
        }

        __syncthreads();
    }

    sum = warp16_reduce_sum(sum);

    if (active && sgLocal == 0) {
        Y[n] = sum;
    }
}