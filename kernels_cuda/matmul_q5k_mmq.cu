// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// M-Cuda.MMQ B2 — Q5_K int8 quantized-matmul (MMQ) GEMM for prefill (M>1).
//
// Identical structure to matmul_q4k_mmq — Q5_K is Q4_K plus one high bit per
// quant (from qh[]), so the quant value is q = nibble + 16*qh_bit (0..31) and
// the affine dequant per 32-element sub-block j is the same:
//   w[j][e] = a_j * q[j][e] - b_j,  a_j = d*scale_j,  b_j = dmin*min_j
//   sum_e w*x = act_scale * ( a_j * dp4a(q, xq) - b_j * sum_e(xq) )
// Q5_K has no CUDA GEMM otherwise (matmul_q5k_vec is a per-row GEMV).
//
// Scope: prefill / M>1 ONLY. One 32-element sub-block per subgroup lane.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifndef MATMUL_Q5K_MMQ_LOCAL
#define MATMUL_Q5K_MMQ_LOCAL 64
#endif
#ifndef MATMUL_Q5K_MMQ_SG
#define MATMUL_Q5K_MMQ_SG 16
#endif
#ifndef MATMUL_Q5K_MMQ_M_TILE
#define MATMUL_Q5K_MMQ_M_TILE 8
#endif
#define MATMUL_Q5K_MMQ_OUTPUTS_PER_GROUP \
    (MATMUL_Q5K_MMQ_LOCAL / MATMUL_Q5K_MMQ_SG)

#define Q5K_BLOCK_ELEMENTS 256
#define Q5K_BLOCK_BYTES    176
#define Q5K_SUB            32
#define X_TILE_ELEMENTS    512
#define SUBS_IN_TILE       (X_TILE_ELEMENTS / Q5K_SUB)     // 16
#define SUBS_PER_SB        (Q5K_BLOCK_ELEMENTS / Q5K_SUB)  // 8

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

extern "C" __global__ __launch_bounds__(MATMUL_Q5K_MMQ_LOCAL)
void matmul_q5k_mmq(
    const float*         __restrict__ X,   // [M, K] fp32
    const unsigned char* __restrict__ W,   // [N, K/256] Q5_K super-blocks
          float*         __restrict__ Y,   // [M, N] fp32
    const int                         K,
    const int                         N,
    const int                         M)
{
    __shared__ float       xTile  [X_TILE_ELEMENTS][MATMUL_Q5K_MMQ_M_TILE];
    __shared__ signed char xqTile [MATMUL_Q5K_MMQ_M_TILE][X_TILE_ELEMENTS];
    __shared__ float       sTile  [SUBS_IN_TILE][MATMUL_Q5K_MMQ_M_TILE];
    __shared__ int         sumTile[SUBS_IN_TILE][MATMUL_Q5K_MMQ_M_TILE];

    const int  tid     = threadIdx.x;
    const int  lsize   = blockDim.x;
    const int  sgInWg  = tid / MATMUL_Q5K_MMQ_SG;
    const int  sgLocal = tid % MATMUL_Q5K_MMQ_SG;
    const int  n       = blockIdx.x * MATMUL_Q5K_MMQ_OUTPUTS_PER_GROUP + sgInWg;
    const int  mBase   = blockIdx.y * MATMUL_Q5K_MMQ_M_TILE;
    const bool nActive = (n < N);
    const int  nSuper  = K / Q5K_BLOCK_ELEMENTS;

    float sum[MATMUL_Q5K_MMQ_M_TILE];
    #pragma unroll
    for (int mm = 0; mm < MATMUL_Q5K_MMQ_M_TILE; ++mm) sum[mm] = 0.0f;

    for (int tile = 0; tile < K; tile += X_TILE_ELEMENTS) {
        const int tileK = (X_TILE_ELEMENTS < K - tile)
                            ? X_TILE_ELEMENTS : (K - tile);

        const int loadTotal = MATMUL_Q5K_MMQ_M_TILE * X_TILE_ELEMENTS;
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

        const int qPairs = SUBS_IN_TILE * MATMUL_Q5K_MMQ_M_TILE;
        for (int pair = tid; pair < qPairs; pair += lsize) {
            const int mSlot = pair / SUBS_IN_TILE;
            const int sb    = pair - mSlot * SUBS_IN_TILE;
            const int base  = sb * Q5K_SUB;
            float amax = 0.0f;
            #pragma unroll
            for (int e = 0; e < Q5K_SUB; ++e) {
                const float a = fabsf(xTile[base + e][mSlot]);
                amax = a > amax ? a : amax;
            }
            const float scale = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
            const float inv   = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
            int ssum = 0;
            #pragma unroll
            for (int e = 0; e < Q5K_SUB; ++e) {
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
                  * static_cast<size_t>(Q5K_BLOCK_BYTES);

            const int superStart = tile / Q5K_BLOCK_ELEMENTS;
            const int superIdx   = superStart + sgLocal / SUBS_PER_SB;
            const int j          = sgLocal % SUBS_PER_SB;
            if (sgLocal < SUBS_IN_TILE && superIdx < nSuper) {
                const unsigned char* __restrict__ block =
                    row + static_cast<size_t>(superIdx) * Q5K_BLOCK_BYTES;
                const float d    = __half2float(*reinterpret_cast<const __half*>(block));
                const float dmin = __half2float(*reinterpret_cast<const __half*>(block + 2));
                const unsigned char* scales = block + 4;
                const unsigned char* qh     = block + 16;
                const unsigned char* qs     = block + 48;

                unsigned int sc, mn;
                getScaleMinK4(j, scales, sc, mn);
                const float a_j = d    * static_cast<float>(sc);
                const float b_j = dmin * static_cast<float>(mn);

                // q[e] = nibble(qs) + 16*qh_bit(qh[e], bit j), e = 0..31.
                const int            p   = j / 2;
                const bool           hi  = (j & 1);
                const unsigned char* qsc = qs + p * Q5K_SUB;
                int w4[Q5K_SUB / 4];
                #pragma unroll
                for (int c = 0; c < Q5K_SUB / 4; ++c) {
                    signed char qv[4];
                    #pragma unroll
                    for (int t = 0; t < 4; ++t) {
                        const int           e    = c * 4 + t;
                        const unsigned char byte = qsc[e];
                        const int nib = hi ? (byte >> 4) : (byte & 0x0Fu);
                        const int hbit = (qh[e] >> j) & 1u;
                        qv[t] = static_cast<signed char>(nib + (hbit ? 16 : 0));
                    }
                    w4[c] = load_char4_as_int(qv);
                }

                const int localBase = sgLocal * Q5K_SUB;
                #pragma unroll
                for (int mm = 0; mm < MATMUL_Q5K_MMQ_M_TILE; ++mm) {
                    const signed char* xqrow = &xqTile[mm][localBase];
                    int dot = 0;
                    #pragma unroll
                    for (int c = 0; c < Q5K_SUB / 4; ++c) {
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
    for (int mm = 0; mm < MATMUL_Q5K_MMQ_M_TILE; ++mm) {
        float s = warp16_reduce_sum(sum[mm]);
        if (nActive && sgLocal == 0) {
            const int mAct = mBase + mm;
            if (mAct < M) {
                Y[static_cast<size_t>(mAct) * static_cast<size_t>(N) + n] = s;
            }
        }
    }
}