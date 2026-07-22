// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// M-Cuda.MMQ B1 — Q8_0 int8 quantized-matmul (MMQ) GEMM for prefill (M>1).
//
// The crossover A/B (2026-07-22, GB10) proved prefill is matmul-/compute-bound
// (linear in T), and matmul_q8_0_gemm_v2 accumulates in fp32 — leaving the int8
// path of the ALU unused. This kernel keeps gemm_v2's tiling but replaces the
// per-element fp32 MAC with a per-K-block int8 dot (__dp4a), scaled once per
// block by (weight_block_scale * activation_block_scale).
//
// Q8_0 is already int8 x per-32-block fp16 scale, so the weight needs no
// re-quant; activations are quantized per (m-row, 32-element block) to int8
// with a per-block fp32 scale, mirroring the block granularity of the weight.
// The scale therefore applies exactly per k-block, so raw int32 dots are never
// accumulated across blocks (each block's dot is scaled to fp32 first).
//
// Scope: prefill / M>1 ONLY. The M=1 decode path is launch-bound (see the
// dp4a-launch-bound-decode lesson) and stays on the existing GEMV kernel.
//
// Correctness structure: each of the 16 subgroup lanes owns WHOLE k-blocks
// (block b = blockStart + lane, strided by 16), computes each block's full
// int8 dot and scales it to fp32 independently, and accumulates into per-m
// fp32 partials. Only ONE cross-lane reduction happens, at the very end — no
// intra-block cross-lane reduction fights the per-block scaling.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifndef MATMUL_Q8_0_MMQ_LOCAL
#define MATMUL_Q8_0_MMQ_LOCAL 64
#endif

#ifndef MATMUL_Q8_0_MMQ_SG
#define MATMUL_Q8_0_MMQ_SG 16
#endif

#ifndef MATMUL_Q8_0_MMQ_M_TILE
#define MATMUL_Q8_0_MMQ_M_TILE 8
#endif

#define MATMUL_Q8_0_MMQ_OUTPUTS_PER_GROUP \
    (MATMUL_Q8_0_MMQ_LOCAL / MATMUL_Q8_0_MMQ_SG)

#define Q8_0_BLOCK_ELEMENTS 32
#define Q8_0_BLOCK_BYTES    34

// 512 elements = 16 blocks per K-tile -> one block per subgroup lane (SG=16),
// so all 16 lanes are active. SLM: fp32 staging 512*8*4 = 16 KiB + int8 xq
// 8*512 = 4 KiB + scales 16*8*4 = 512 B.
#define X_TILE_ELEMENTS 512
#define BLOCKS_IN_TILE  (X_TILE_ELEMENTS / Q8_0_BLOCK_ELEMENTS)  // 16

static __device__ __forceinline__ float warp16_reduce_sum(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 8, 16);
    v += __shfl_xor_sync(0xffffffffu, v, 4, 16);
    v += __shfl_xor_sync(0xffffffffu, v, 2, 16);
    v += __shfl_xor_sync(0xffffffffu, v, 1, 16);
    return v;
}

// Four contiguous signed int8 packed little-endian into one int32 for __dp4a.
static __device__ __forceinline__ int load_char4_as_int(const signed char* p) {
    return  static_cast<int>(static_cast<unsigned char>(p[0]))
         | (static_cast<int>(static_cast<unsigned char>(p[1])) << 8)
         | (static_cast<int>(static_cast<unsigned char>(p[2])) << 16)
         | (static_cast<int>(static_cast<unsigned char>(p[3])) << 24);
}

extern "C" __global__ __launch_bounds__(MATMUL_Q8_0_MMQ_LOCAL)
void matmul_q8_0_mmq(
    const float*         __restrict__ X,   // [M, K] fp32 activations
    const unsigned char* __restrict__ W,   // [N, K/32] Q8_0 blocks (34 B each)
          float*         __restrict__ Y,   // [M, N] fp32
    const int                         K,
    const int                         N,
    const int                         M)
{
    __shared__ float       xTile [X_TILE_ELEMENTS][MATMUL_Q8_0_MMQ_M_TILE];
    __shared__ signed char xqTile[MATMUL_Q8_0_MMQ_M_TILE][X_TILE_ELEMENTS];
    __shared__ float       sTile [BLOCKS_IN_TILE][MATMUL_Q8_0_MMQ_M_TILE];

    const int  wgN     = blockIdx.x;
    const int  wgM     = blockIdx.y;
    const int  tid     = threadIdx.x;
    const int  lsize   = blockDim.x;
    const int  sgInWg  = tid / MATMUL_Q8_0_MMQ_SG;
    const int  sgLocal = tid % MATMUL_Q8_0_MMQ_SG;
    const int  n       = wgN * MATMUL_Q8_0_MMQ_OUTPUTS_PER_GROUP + sgInWg;
    const int  mBase   = wgM * MATMUL_Q8_0_MMQ_M_TILE;
    const bool nActive = (n < N);
    const int  nBlocks = K / Q8_0_BLOCK_ELEMENTS;

    float sum[MATMUL_Q8_0_MMQ_M_TILE];
    #pragma unroll
    for (int mm = 0; mm < MATMUL_Q8_0_MMQ_M_TILE; ++mm) sum[mm] = 0.0f;

    for (int tile = 0; tile < K; tile += X_TILE_ELEMENTS) {
        const int tileK = (X_TILE_ELEMENTS < K - tile)
                            ? X_TILE_ELEMENTS : (K - tile);

        // --- stage fp32 activations [iSlot][mSlot], zero-padded past tileK ---
        const int loadTotal = MATMUL_Q8_0_MMQ_M_TILE * X_TILE_ELEMENTS;
        for (int idx = tid; idx < loadTotal; idx += lsize) {
            const int  mSlot = idx / X_TILE_ELEMENTS;
            const int  iSlot = idx - mSlot * X_TILE_ELEMENTS;
            const int  mAct  = mBase + mSlot;
            const bool valid = (mAct < M) && (iSlot < tileK);
            xTile[iSlot][mSlot] =
                valid ? X[static_cast<size_t>(mAct) * static_cast<size_t>(K)
                        + tile + iSlot]
                      : 0.0f;
        }
        __syncthreads();

        // --- quantize per (m-row, 32-elem block) to int8 + fp32 scale --------
        // BLOCKS_IN_TILE * M_TILE = 16 * 8 = 128 pairs; lsize=64 -> 2 per thread.
        const int qPairs = BLOCKS_IN_TILE * MATMUL_Q8_0_MMQ_M_TILE;
        for (int pair = tid; pair < qPairs; pair += lsize) {
            const int mSlot = pair / BLOCKS_IN_TILE;
            const int blk   = pair - mSlot * BLOCKS_IN_TILE;
            const int base  = blk * Q8_0_BLOCK_ELEMENTS;
            float amax = 0.0f;
            #pragma unroll
            for (int l = 0; l < Q8_0_BLOCK_ELEMENTS; ++l) {
                const float a = fabsf(xTile[base + l][mSlot]);
                amax = a > amax ? a : amax;
            }
            const float scale = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
            const float inv    = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
            sTile[blk][mSlot] = scale;
            #pragma unroll
            for (int l = 0; l < Q8_0_BLOCK_ELEMENTS; ++l) {
                const float q = rintf(xTile[base + l][mSlot] * inv);
                xqTile[mSlot][base + l] = static_cast<signed char>(q);
            }
        }
        __syncthreads();

        // --- one whole block per lane; full int8 dot, scaled to fp32 --------
        if (nActive) {
            const unsigned char* __restrict__ wrow =
                W + static_cast<size_t>(n) * static_cast<size_t>(nBlocks)
                  * static_cast<size_t>(Q8_0_BLOCK_BYTES);

            const int blockStart = tile / Q8_0_BLOCK_ELEMENTS;
            const int b          = blockStart + sgLocal;   // lane owns one block
            if (sgLocal < BLOCKS_IN_TILE && b < nBlocks) {
                const unsigned char* __restrict__ block =
                    wrow + static_cast<size_t>(b) * Q8_0_BLOCK_BYTES;
                const float d =
                    __half2float(*reinterpret_cast<const __half*>(block));
                const signed char* wq =
                    reinterpret_cast<const signed char*>(block + 2);

                // Pack the 8 weight int32 chunks once (reused across all m).
                int w4[Q8_0_BLOCK_ELEMENTS / 4];
                #pragma unroll
                for (int c = 0; c < Q8_0_BLOCK_ELEMENTS / 4; ++c) {
                    w4[c] = load_char4_as_int(wq + c * 4);
                }

                const int localBlk = sgLocal;   // == b - blockStart (SG==16==tile)
                #pragma unroll
                for (int mm = 0; mm < MATMUL_Q8_0_MMQ_M_TILE; ++mm) {
                    const signed char* xqrow = &xqTile[mm][localBlk * Q8_0_BLOCK_ELEMENTS];
                    int acc = 0;
                    #pragma unroll
                    for (int c = 0; c < Q8_0_BLOCK_ELEMENTS / 4; ++c) {
                        acc = __dp4a(w4[c], load_char4_as_int(xqrow + c * 4), acc);
                    }
                    sum[mm] = __fmaf_rn(static_cast<float>(acc),
                                        d * sTile[localBlk][mm], sum[mm]);
                }
            }
        }
        __syncthreads();
    }

    // --- reduce the per-lane partials (each lane held different blocks) ------
    #pragma unroll
    for (int mm = 0; mm < MATMUL_Q8_0_MMQ_M_TILE; ++mm) {
        float s = warp16_reduce_sum(sum[mm]);
        if (nActive && sgLocal == 0) {
            const int mAct = mBase + mm;
            if (mAct < M) {
                Y[static_cast<size_t>(mAct) * static_cast<size_t>(N) + n] = s;
            }
        }
    }
}