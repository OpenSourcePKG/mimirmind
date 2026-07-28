// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
// Ported from kernels_hip/matmul_q5k_vec.hip — Track 4 mechanical port, no functional change intended.

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Matrix-vector multiply with Q5_K weights, on-the-fly dequant.
//
//   Y[n] = sum_{k=0..K-1} X[k] * dequant_q5_k(W, n, k)
//
//   X:  [K]     F32 dense vector (single token / M=1)
//   W:  [N, K]  Q5_K — each row is (K/256) super-blocks of 176 bytes.
//   Y:  [N]     F32 dense vector
//
// Sibling of matmul_q4k_vec.hip / matmul_q6k_vec.hip: same warp layout
// (RDNA3 warpSize=32, LOCAL=128 = 4 warps × 32 lanes, OUTPUTS_PER_GROUP=4).
// Q5_K is the asymmetric 5-bit K-quant — nibbles from qs[], one extra
// bit per quant from qh[], plus per-sub-block scale + min like Q4_K:
//
//   value = d * scale_j * (nibble + (qh_bit ? 16 : 0)) - dmin * min_j
//
// Launch:
//   dim3 grid ( ceil(N / OUTPUTS_PER_GROUP), 1, 1 )
//   dim3 block( MATMUL_Q5K_LOCAL, 1, 1 )
//
// Q5_K super-block layout (bit-identical to llama.cpp
// dequantize_row_q5_K and src/compute/quant/Q5K.cpp:62-113):
//
//   fp16    d          (bytes 0..1)    — scale-of-scales
//   fp16    dmin       (bytes 2..3)    — scale-of-mins
//   uint8   scales[12] (bytes 4..15)   — 8 × (6-bit scale, 6-bit min),
//                                        packed via ggml's
//                                        get_scale_min_k4 format
//                                        (identical to Q4_K)
//   uint8   qh[32]     (bytes 16..47)  — 256 high-bits (1 per element).
//                                        qh[l] bit `u1` covers sub-block
//                                        jLo, bit `u2` covers sub-block
//                                        jHi; u1/u2 shift left by 2 per
//                                        pair (u1: 0x01,0x04,0x10,0x40;
//                                        u2: 0x02,0x08,0x20,0x80).
//   uint8   qs[128]    (bytes 48..175) — 256 nibbles (low 4 bits),
//                                        low-nibble first
//
// Lane assignment: laneId l ∈ [0..32) handles the 8 elements at
// positions {l, 32+l, 64+l, ..., 224+l} — one element per sub-block,
// eight sub-blocks per super-block, 32 lanes × 8 = 256.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifndef MATMUL_Q5K_LOCAL
#define MATMUL_Q5K_LOCAL 128
#endif

#define MATMUL_Q5K_WARPS             (MATMUL_Q5K_LOCAL / 32)
#define MATMUL_Q5K_OUTPUTS_PER_GROUP MATMUL_Q5K_WARPS

#define Q5K_BLOCK_ELEMENTS 256
#define Q5K_BLOCK_BYTES    176
#define X_TILE_ELEMENTS    1024

namespace {

__device__ __forceinline__ float warpReduceSum(float v) {
    v += __shfl_down_sync(0xffffffffu, v, 16);
    v += __shfl_down_sync(0xffffffffu, v,  8);
    v += __shfl_down_sync(0xffffffffu, v,  4);
    v += __shfl_down_sync(0xffffffffu, v,  2);
    v += __shfl_down_sync(0xffffffffu, v,  1);
    return v;
}

// Shared with matmul_q4k_vec — matches Q4K.cpp:48-61 and Q5K.cpp:20-33.
__device__ __forceinline__ void getScaleMinK4(
    int j, const unsigned char* q, unsigned int& scale, unsigned int& min) {
    if (j < 4) {
        scale = static_cast<unsigned int>(q[j]     & 0x3Fu);
        min   = static_cast<unsigned int>(q[j + 4] & 0x3Fu);
    } else {
        scale = static_cast<unsigned int>(
            (q[j + 4] & 0x0Fu) | ((q[j - 4] >> 6) << 4));
        min   = static_cast<unsigned int>(
            (q[j + 4] >> 4)    | ((q[j    ] >> 6) << 4));
    }
}

} // namespace

extern "C" __global__ __launch_bounds__(MATMUL_Q5K_LOCAL)
void matmul_q5k_vec(
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
    const int n       = wg * MATMUL_Q5K_OUTPUTS_PER_GROUP + warpId;
    const bool active = (n < N);
    const int nBlocks = K / Q5K_BLOCK_ELEMENTS;

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
                  * Q5K_BLOCK_BYTES;

            const int blockStart   = tile / Q5K_BLOCK_ELEMENTS;
            const int blocksInTile = X_TILE_ELEMENTS / Q5K_BLOCK_ELEMENTS;
            const int blockEnd     = min(blockStart + blocksInTile, nBlocks);

            for (int b = blockStart; b < blockEnd; ++b) {
                const unsigned char* block = row + b * Q5K_BLOCK_BYTES;

                const __half* d_ptr    = reinterpret_cast<const __half*>(block);
                const __half* dmin_ptr = reinterpret_cast<const __half*>(block + 2);
                const float d    = __half2float(d_ptr[0]);
                const float dmin = __half2float(dmin_ptr[0]);

                const unsigned char* scales = block + 4;
                const unsigned char* qh     = block + 16;
                const unsigned char* qs     = block + 48;

                const int xLocalBase = (b - blockStart) * Q5K_BLOCK_ELEMENTS;
                const int l          = laneId;
                const unsigned int qhByte = qh[l];

                // 4 pairs of sub-blocks. u1/u2 shift by 2 per pair —
                // one qh byte covers all 4 pairs of a lane's element
                // positions. Iterations unrolled for register clarity.
                #pragma unroll
                for (int pair = 0; pair < 4; ++pair) {
                    const int jLo = 2 * pair;
                    const int jHi = jLo + 1;
                    const unsigned int u1 = 1u << jLo;
                    const unsigned int u2 = 1u << jHi;

                    unsigned int sLo, mLo, sHi, mHi;
                    getScaleMinK4(jLo, scales, sLo, mLo);
                    getScaleMinK4(jHi, scales, sHi, mHi);

                    const float dLo  = d    * static_cast<float>(sLo);
                    const float mmLo = dmin * static_cast<float>(mLo);
                    const float dHi  = d    * static_cast<float>(sHi);
                    const float mmHi = dmin * static_cast<float>(mHi);

                    const unsigned char qb = qs[pair * 32 + l];
                    const unsigned int hiLo = (qhByte & u1) ? 16u : 0u;
                    const unsigned int hiHi = (qhByte & u2) ? 16u : 0u;
                    const float qLo = static_cast<float>((qb & 0x0Fu) + hiLo);
                    const float qHi = static_cast<float>((qb >> 4)    + hiHi);

                    // Sub-block jLo → element (jLo*32 + l),
                    // sub-block jHi → element (jHi*32 + l).
                    const float wLo = dLo * qLo - mmLo;
                    const float wHi = dHi * qHi - mmHi;

                    const float xLo = xTile[xLocalBase + jLo * 32 + l];
                    const float xHi = xTile[xLocalBase + jHi * 32 + l];

                    sum = __fmaf_rn(xLo, wLo, sum);
                    sum = __fmaf_rn(xHi, wHi, sum);
                }
            }
        }

        __syncthreads();
    }

    sum = warpReduceSum(sum);

    if (active && laneId == 0) {
        Y[n] = sum;
    }
}