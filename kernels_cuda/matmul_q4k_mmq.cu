// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// M-Cuda.MMQ B2 — Q4_K int8 quantized-matmul (MMQ) GEMM for prefill (M>1).
//
// Q4_K has NO CUDA GEMM at all (matmul_q4k_vec is a per-row GEMV, re-read the
// whole weight matrix once per token at prefill). This kernel gives Q4_K a
// tiled GEMM AND runs the dot in int8 (__dp4a) instead of fp32.
//
// Q4_K dequant is affine per 32-element sub-block j:
//   w[j][e] = a_j * nibble[j][e] - b_j,   a_j = d*scale_j,  b_j = dmin*min_j
// so the sub-block's contribution decomposes into two int8-reducible sums:
//   sum_e w*x = a_j * sum_e(nibble*xq) * act_scale  -  b_j * sum_e(x)
//             = act_scale * ( a_j * dp4a(nibble, xq)  -  b_j * sum_e(xq) )
// where activations are quantised per (m-row, sub-block) to int8 with a
// per-sub-block fp32 scale. The two int32 reductions (nibble·xq and Σxq) are
// scaled to fp32 per sub-block, so no raw int32 is accumulated across blocks.
//
// Scope: prefill / M>1 ONLY. Each subgroup lane owns ONE 32-element sub-block
// (16 lanes = 2 super-blocks per K-tile), one final cross-lane reduction.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifndef MATMUL_Q4K_MMQ_LOCAL
#define MATMUL_Q4K_MMQ_LOCAL 64
#endif
#ifndef MATMUL_Q4K_MMQ_SG
#define MATMUL_Q4K_MMQ_SG 16
#endif
#ifndef MATMUL_Q4K_MMQ_M_TILE
#define MATMUL_Q4K_MMQ_M_TILE 8
#endif
#define MATMUL_Q4K_MMQ_OUTPUTS_PER_GROUP \
    (MATMUL_Q4K_MMQ_LOCAL / MATMUL_Q4K_MMQ_SG)

#define Q4K_BLOCK_ELEMENTS 256
#define Q4K_BLOCK_BYTES    144
#define Q4K_SUB            32                        // sub-block size
#define X_TILE_ELEMENTS    512                       // 2 super-blocks
#define SUBS_IN_TILE       (X_TILE_ELEMENTS / Q4K_SUB)    // 16
#define SUBS_PER_SB        (Q4K_BLOCK_ELEMENTS / Q4K_SUB) // 8

static __device__ __forceinline__ float warp16_reduce_sum(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 8, 16);
    v += __shfl_xor_sync(0xffffffffu, v, 4, 16);
    v += __shfl_xor_sync(0xffffffffu, v, 2, 16);
    v += __shfl_xor_sync(0xffffffffu, v, 1, 16);
    return v;
}

static __device__ __forceinline__ int load_char4_as_int(const signed char* p) {
    return  static_cast<int>(static_cast<unsigned char>(p[0]))
         | (static_cast<int>(static_cast<unsigned char>(p[1])) << 8)
         | (static_cast<int>(static_cast<unsigned char>(p[2])) << 16)
         | (static_cast<int>(static_cast<unsigned char>(p[3])) << 24);
}

// One of the 8 (scale, min) pairs from the 12-byte packed scales field.
// Matches ggml get_scale_min_k4 / matmul_q4k_vec.
static __device__ __forceinline__ void getScaleMinK4(
    int j, const unsigned char* q, unsigned int& scale, unsigned int& mn) {
    if (j < 4) {
        scale = static_cast<unsigned int>(q[j]     & 0x3Fu);
        mn    = static_cast<unsigned int>(q[j + 4] & 0x3Fu);
    } else {
        scale = static_cast<unsigned int>(
            (q[j + 4] & 0x0Fu) | ((q[j - 4] >> 6) << 4));
        mn    = static_cast<unsigned int>(
            (q[j + 4] >> 4)    | ((q[j    ] >> 6) << 4));
    }
}

extern "C" __global__ __launch_bounds__(MATMUL_Q4K_MMQ_LOCAL)
void matmul_q4k_mmq(
    const float*         __restrict__ X,   // [M, K] fp32
    const unsigned char* __restrict__ W,   // [N, K/256] Q4_K super-blocks
          float*         __restrict__ Y,   // [M, N] fp32
    const int                         K,
    const int                         N,
    const int                         M)
{
    __shared__ float       xTile  [X_TILE_ELEMENTS][MATMUL_Q4K_MMQ_M_TILE];
    __shared__ signed char xqTile [MATMUL_Q4K_MMQ_M_TILE][X_TILE_ELEMENTS];
    __shared__ float       sTile  [SUBS_IN_TILE][MATMUL_Q4K_MMQ_M_TILE];
    __shared__ int         sumTile[SUBS_IN_TILE][MATMUL_Q4K_MMQ_M_TILE];

    const int  tid     = threadIdx.x;
    const int  lsize   = blockDim.x;
    const int  sgInWg  = tid / MATMUL_Q4K_MMQ_SG;
    const int  sgLocal = tid % MATMUL_Q4K_MMQ_SG;
    const int  n       = blockIdx.x * MATMUL_Q4K_MMQ_OUTPUTS_PER_GROUP + sgInWg;
    const int  mBase   = blockIdx.y * MATMUL_Q4K_MMQ_M_TILE;
    const bool nActive = (n < N);
    const int  nSuper  = K / Q4K_BLOCK_ELEMENTS;   // super-blocks per row

    float sum[MATMUL_Q4K_MMQ_M_TILE];
    #pragma unroll
    for (int mm = 0; mm < MATMUL_Q4K_MMQ_M_TILE; ++mm) sum[mm] = 0.0f;

    for (int tile = 0; tile < K; tile += X_TILE_ELEMENTS) {
        const int tileK = (X_TILE_ELEMENTS < K - tile)
                            ? X_TILE_ELEMENTS : (K - tile);

        const int loadTotal = MATMUL_Q4K_MMQ_M_TILE * X_TILE_ELEMENTS;
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

        // Quantize per (m-row, 32-elem sub-block) -> int8 + fp32 scale + Sxq.
        const int qPairs = SUBS_IN_TILE * MATMUL_Q4K_MMQ_M_TILE;
        for (int pair = tid; pair < qPairs; pair += lsize) {
            const int mSlot = pair / SUBS_IN_TILE;
            const int sb    = pair - mSlot * SUBS_IN_TILE;
            const int base  = sb * Q4K_SUB;
            float amax = 0.0f;
            #pragma unroll
            for (int e = 0; e < Q4K_SUB; ++e) {
                const float a = fabsf(xTile[base + e][mSlot]);
                amax = a > amax ? a : amax;
            }
            const float scale = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
            const float inv   = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
            int ssum = 0;
            #pragma unroll
            for (int e = 0; e < Q4K_SUB; ++e) {
                const int q = static_cast<int>(rintf(xTile[base + e][mSlot] * inv));
                xqTile[mSlot][base + e] = static_cast<signed char>(q);
                ssum += q;
            }
            sTile[sb][mSlot]   = scale;
            sumTile[sb][mSlot] = ssum;
        }
        __syncthreads();

        if (nActive) {
            const unsigned char* __restrict__ row =
                W + static_cast<size_t>(n) * static_cast<size_t>(nSuper)
                  * static_cast<size_t>(Q4K_BLOCK_BYTES);

            const int superStart = tile / Q4K_BLOCK_ELEMENTS;
            const int superIdx   = superStart + sgLocal / SUBS_PER_SB; // 0..7->+0, 8..15->+1
            const int j          = sgLocal % SUBS_PER_SB;              // sub-block in super-block
            if (sgLocal < SUBS_IN_TILE && superIdx < nSuper) {
                const unsigned char* __restrict__ block =
                    row + static_cast<size_t>(superIdx) * Q4K_BLOCK_BYTES;
                const float d    = __half2float(*reinterpret_cast<const __half*>(block));
                const float dmin = __half2float(*reinterpret_cast<const __half*>(block + 2));
                const unsigned char* scales = block + 4;
                const unsigned char* qs     = block + 16;

                unsigned int sc, mn;
                getScaleMinK4(j, scales, sc, mn);
                const float a_j = d    * static_cast<float>(sc);
                const float b_j = dmin * static_cast<float>(mn);

                // Extract this sub-block's 32 nibbles and pack into 8 int32.
                const int            p   = j / 2;         // qs 32-byte chunk
                const bool           hi  = (j & 1);
                const unsigned char* qsc = qs + p * Q4K_SUB;
                int w4[Q4K_SUB / 4];
                #pragma unroll
                for (int c = 0; c < Q4K_SUB / 4; ++c) {
                    signed char nib[4];
                    #pragma unroll
                    for (int t = 0; t < 4; ++t) {
                        const unsigned char byte = qsc[c * 4 + t];
                        nib[t] = static_cast<signed char>(hi ? (byte >> 4)
                                                             : (byte & 0x0Fu));
                    }
                    w4[c] = load_char4_as_int(nib);
                }

                const int localBase = sgLocal * Q4K_SUB;   // xqTile column base
                #pragma unroll
                for (int mm = 0; mm < MATMUL_Q4K_MMQ_M_TILE; ++mm) {
                    const signed char* xqrow = &xqTile[mm][localBase];
                    int dot = 0;
                    #pragma unroll
                    for (int c = 0; c < Q4K_SUB / 4; ++c) {
                        dot = __dp4a(w4[c], load_char4_as_int(xqrow + c * 4), dot);
                    }
                    const float contrib = sTile[sgLocal][mm] *
                        (a_j * static_cast<float>(dot)
                         - b_j * static_cast<float>(sumTile[sgLocal][mm]));
                    sum[mm] += contrib;
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int mm = 0; mm < MATMUL_Q4K_MMQ_M_TILE; ++mm) {
        float s = warp16_reduce_sum(sum[mm]);
        if (nActive && sgLocal == 0) {
            const int mAct = mBase + mm;
            if (mAct < M) {
                Y[static_cast<size_t>(mAct) * static_cast<size_t>(N) + n] = s;
            }
        }
    }
}