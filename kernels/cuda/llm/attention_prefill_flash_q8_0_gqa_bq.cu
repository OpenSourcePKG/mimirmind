// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// R3 — Query-row (BQ) tiled variant of attention_prefill_flash_q8_0_gqa.cu.
//
// The R1 GQA-head-packed kernel already shares each K/V Q8_0 block's
// dequantisation across the nQPerKv query HEADS in a group. It does not
// share anything across query POSITIONS: grid.y = T_q, so a causal
// prefill re-walks (and re-dequantises) the same K-tile once per query
// row, even though adjacent rows have nearly-identical causal K ranges.
// This kernel additionally batches BQ query ROWS per CTA, so one K/V
// dequant pass is now reused across nQPerKv * BQ query slots instead of
// just nQPerKv — see the research note "Gemma4 head_dim=512 full-attention
// layers — confirmed dominant, super-linear prefill bottleneck at long
// context" for the measurement that motivated this.
//
// Masking is now per-ROW (each of the BQ rows has its own causal bound
// and, for SWA layers, its own sliding-window start) while the K/V
// dequant work stays row-independent — mirrors how
// attention_prefill_flash_f32_gqa_mwtc.cu masks per-row on top of a
// shared K-tile, just with a scalar dot-product instead of wmma.
//
// Static (not dynamic) shared memory, sized for compile-time
// N_Q_PER_KV_MAX/BQ/MAX_HEADDIM bounds — matches the R1 kernel's approach
// (no host-side setMaxDynamicSharedBytes opt-in needed). N_Q_PER_KV_MAX is
// deliberately smaller here (4, vs R1's 8) to keep oRun's footprint
// (NQMAX*BQ*MAXHEADDIM floats) inside the ~48 KiB static-shared budget —
// Gemma4's nQPerKv=2 is well within it; a model needing nQPerKv in (4,8]
// simply isn't eligible for this kernel and dispatch falls back to R1.
//
// Launch geometry: grid( nKvHeads, ceil(T_q/BQ), 1 ), block(LOCAL,1,1).

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <math.h>   // for INFINITY

#ifndef ATTN_FLASH_PREFILL_LOCAL
#define ATTN_FLASH_PREFILL_LOCAL 16
#endif

#ifndef ATTN_FLASH_PREFILL_KTILE
#define ATTN_FLASH_PREFILL_KTILE 128
#endif

#ifndef ATTN_FLASH_PREFILL_MAX_HEADDIM
#define ATTN_FLASH_PREFILL_MAX_HEADDIM 512
#endif

// Max query heads per KV group this BQ kernel supports. Smaller than R1's
// 8 so NQMAX*BQ*MAXHEADDIM stays inside the static-shared budget; Gemma4
// (nQPerKv=2) and typical GQA ratios fit comfortably.
#ifndef ATTN_FLASH_PREFILL_N_Q_PER_KV_MAX
#define ATTN_FLASH_PREFILL_N_Q_PER_KV_MAX 4
#endif

// Query rows (positions) batched per CTA. The whole point of this kernel:
// one K/V Q8_0 dequant pass is shared across BQ*nQPerKv slots instead of
// R1's nQPerKv. NQMAX(4) * BQ(4) * MAXHEADDIM(512) = 8192 floats = 32 KiB
// for oRun, + NQMAX*BQ*KTILE = 8 KiB for scores -> 40 KiB total, comfortably
// under the 48 KiB static-shared limit.
#ifndef ATTN_FLASH_PREFILL_BQ
#define ATTN_FLASH_PREFILL_BQ 4
#endif

#define Q8_0_BLOCK_ELEMENTS 32
#define Q8_0_BLOCK_BYTES    34

static __device__ __forceinline__ float warp16_reduce_max(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 8, 16));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 4, 16));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2, 16));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1, 16));
    return v;
}

static __device__ __forceinline__ float warp16_reduce_sum(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 8, 16);
    v += __shfl_xor_sync(0xffffffffu, v, 4, 16);
    v += __shfl_xor_sync(0xffffffffu, v, 2, 16);
    v += __shfl_xor_sync(0xffffffffu, v, 1, 16);
    return v;
}

extern "C" __global__ __launch_bounds__(ATTN_FLASH_PREFILL_LOCAL)
void attention_prefill_flash_q8_0_gqa_bq(
    const float*         __restrict__ q,
    const unsigned char* __restrict__ k,
    const unsigned char* __restrict__ v,
          float*         __restrict__ out,
    const int                         T_q,
    const int                         nHeads,
    const int                         nKvHeads,
    const int                         headDim,
    const int*           __restrict__ curLenPtr,
    const float                       scale,
    const int                         slidingWindow)
{
    constexpr int NQMAX = ATTN_FLASH_PREFILL_N_Q_PER_KV_MAX;
    constexpr int BQ    = ATTN_FLASH_PREFILL_BQ;

    const int hkv     = blockIdx.x;                 // KV-head index
    const int q0      = blockIdx.y * BQ;             // first query position
    const int lid     = threadIdx.x;
    const int nQPerKv = nHeads / nKvHeads;           // >= 1, <= NQMAX

    const int qStride        = nHeads   * headDim;
    const int kvDim          = nKvHeads * headDim;
    const int nBlocksPerRow  = kvDim   / Q8_0_BLOCK_ELEMENTS;
    const int nBlocksPerHead = headDim / Q8_0_BLOCK_ELEMENTS;
    const int headBlockBase  = (hkv * headDim) / Q8_0_BLOCK_ELEMENTS;
    const int positionOffset = curLenPtr[0];

    // Last valid row in this tile (T_q may not be a multiple of BQ).
    const int lastM = (q0 + BQ - 1 < T_q) ? (BQ - 1) : (T_q - 1 - q0);

    // Per-row causal bound (exclusive) and, for SWA, per-row window start.
    // kMax_m is monotonically increasing in m, so row 0 has the earliest
    // window start and row `lastM` has the latest (= widest) causal bound —
    // that pair brackets the whole tile's K range.
    auto kMaxFor = [&](int m) { return positionOffset + q0 + m + 1; };
    auto kMinFor = [&](int m) {
        const int km = kMaxFor(m);
        return (slidingWindow > 0 && km > slidingWindow) ? (km - slidingWindow) : 0;
    };
    const int kMinBlock = kMinFor(0);
    const int kMaxBlock = kMaxFor(lastM);
    const int ktStart   = kMinBlock / ATTN_FLASH_PREFILL_KTILE;
    const int nKTiles   = (kMaxBlock + ATTN_FLASH_PREFILL_KTILE - 1)
                         / ATTN_FLASH_PREFILL_KTILE;

    // Q/out vector bases: [qh][m], global-memory pointers (Q is small
    // enough — <=512 floats/row — that re-reading it from global per
    // K-tile-row thread is cheap and L1/L2-cached; matches R1's approach).
    const float* qVecs[NQMAX][BQ];
          float* oVecs[NQMAX][BQ];
    #pragma unroll
    for (int qh = 0; qh < NQMAX; ++qh) {
        const int hq = hkv * nQPerKv + qh;
        #pragma unroll
        for (int m = 0; m < BQ; ++m) {
            const int pq = q0 + m;
            qVecs[qh][m] = q   + static_cast<size_t>(pq) * static_cast<size_t>(qStride)
                                + static_cast<size_t>(hq) * static_cast<size_t>(headDim);
            oVecs[qh][m] = out + static_cast<size_t>(pq) * static_cast<size_t>(qStride)
                                + static_cast<size_t>(hq) * static_cast<size_t>(headDim);
        }
    }

    __shared__ float scores[NQMAX][BQ][ATTN_FLASH_PREFILL_KTILE];
    __shared__ float oRun  [NQMAX][BQ][ATTN_FLASH_PREFILL_MAX_HEADDIM];

    float m_reg[NQMAX][BQ];
    float l_reg[NQMAX][BQ];
    #pragma unroll
    for (int qh = 0; qh < NQMAX; ++qh) {
        #pragma unroll
        for (int m = 0; m < BQ; ++m) {
            m_reg[qh][m] = -INFINITY;
            l_reg[qh][m] = 0.0f;
        }
    }
    #pragma unroll
    for (int qh = 0; qh < NQMAX; ++qh) {
        if (qh >= nQPerKv) continue;
        #pragma unroll
        for (int m = 0; m < BQ; ++m) {
            for (int d = lid; d < headDim; d += ATTN_FLASH_PREFILL_LOCAL) {
                oRun[qh][m][d] = 0.0f;
            }
        }
    }
    __syncthreads();

    for (int kt = ktStart; kt < nKTiles; ++kt) {
        const int kStartRaw = kt * ATTN_FLASH_PREFILL_KTILE;
        const int kStart    = (kStartRaw > kMinBlock) ? kStartRaw : kMinBlock;
        const int kEndRaw   = kStartRaw + ATTN_FLASH_PREFILL_KTILE;
        const int kEnd      = (kEndRaw < kMaxBlock) ? kEndRaw : kMaxBlock;
        const int tileLen   = kEnd - kStart;

        // -- Pass A — Q·K scaled scores. K-blocks loaded ONCE per (kk,
        // blk) pair and reused across every active (Q-head, row) slot —
        // the R1 saving (across heads) times the new BQ saving (across
        // rows). Per-row causal/window masking applied when writing the
        // score, not when accumulating the dot product.
        for (int kk = lid; kk < tileLen; kk += ATTN_FLASH_PREFILL_LOCAL) {
            const int kkAbs = kStart + kk;
            const unsigned char* __restrict__ kRow = k
                + static_cast<size_t>(kkAbs)
                * static_cast<size_t>(nBlocksPerRow)
                * static_cast<size_t>(Q8_0_BLOCK_BYTES);
            const unsigned char* __restrict__ kHead = kRow
                + static_cast<size_t>(headBlockBase)
                * static_cast<size_t>(Q8_0_BLOCK_BYTES);

            float acc[NQMAX][BQ];
            #pragma unroll
            for (int qh = 0; qh < NQMAX; ++qh) {
                #pragma unroll
                for (int m = 0; m < BQ; ++m) acc[qh][m] = 0.0f;
            }

            for (int blk = 0; blk < nBlocksPerHead; ++blk) {
                const unsigned char* __restrict__ blkPtr = kHead
                    + static_cast<size_t>(blk)
                    * static_cast<size_t>(Q8_0_BLOCK_BYTES);
                const float bscale =
                    __half2float(*reinterpret_cast<const __half*>(blkPtr));
                const signed char* qArr =
                    reinterpret_cast<const signed char*>(blkPtr + 2);
                const int dBase = blk * Q8_0_BLOCK_ELEMENTS;
                for (int in = 0; in < Q8_0_BLOCK_ELEMENTS; ++in) {
                    // Each K-quant scaled once; broadcast to every active
                    // (Q-head, row) slot's dot-product accumulator.
                    const float k_val = bscale * static_cast<float>(qArr[in]);
                    #pragma unroll
                    for (int qh = 0; qh < NQMAX; ++qh) {
                        #pragma unroll
                        for (int m = 0; m < BQ; ++m) {
                            acc[qh][m] += qVecs[qh][m][dBase + in] * k_val;
                        }
                    }
                }
            }

            #pragma unroll
            for (int m = 0; m < BQ; ++m) {
                const bool rowValid = (m <= lastM);
                const int  kMax_m   = kMaxFor(m);
                const int  kMin_m   = kMinFor(m);
                const bool masked   = !rowValid || kkAbs >= kMax_m || kkAbs < kMin_m;
                #pragma unroll
                for (int qh = 0; qh < NQMAX; ++qh) {
                    if (qh >= nQPerKv) continue;
                    scores[qh][m][kk] = masked ? -INFINITY : acc[qh][m] * scale;
                }
            }
        }
        __syncthreads();

        // -- Pass B — Online-softmax rescale, per (Q-head, row). --------
        float alpha[NQMAX][BQ];
        #pragma unroll
        for (int qh = 0; qh < NQMAX; ++qh) {
            #pragma unroll
            for (int m = 0; m < BQ; ++m) {
                if (qh >= nQPerKv) { alpha[qh][m] = 1.0f; continue; }

                float mTilePart = -INFINITY;
                for (int kk = lid; kk < tileLen; kk += ATTN_FLASH_PREFILL_LOCAL) {
                    const float s = scores[qh][m][kk];
                    if (s > mTilePart) mTilePart = s;
                }
                const float mTile = warp16_reduce_max(mTilePart);
                const float mNew  = (m_reg[qh][m] > mTile) ? m_reg[qh][m] : mTile;
                alpha[qh][m]      = expf(m_reg[qh][m] - mNew);

                float lTilePart = 0.0f;
                for (int kk = lid; kk < tileLen; kk += ATTN_FLASH_PREFILL_LOCAL) {
                    const float e = expf(scores[qh][m][kk] - mNew);
                    scores[qh][m][kk] = e;
                    lTilePart += e;
                }
                const float lTile = warp16_reduce_sum(lTilePart);
                l_reg[qh][m] = alpha[qh][m] * l_reg[qh][m] + lTile;
                m_reg[qh][m] = mNew;
            }
        }
        __syncthreads();

        // -- Pass C — V·softmax -> oRun, shared V-load across (Q-head,
        // row) slots. Each thread strides over headDim; per (d, kk) one V
        // Q8_0 block header (scale + int8 quant) is loaded once and
        // broadcast to every active (qh, m) accumulator.
        for (int d = lid; d < headDim; d += ATTN_FLASH_PREFILL_LOCAL) {
            const int blk = d / Q8_0_BLOCK_ELEMENTS;
            const int in  = d % Q8_0_BLOCK_ELEMENTS;

            float acc_v[NQMAX][BQ];
            #pragma unroll
            for (int qh = 0; qh < NQMAX; ++qh) {
                #pragma unroll
                for (int m = 0; m < BQ; ++m) {
                    acc_v[qh][m] = (qh < nQPerKv) ? (alpha[qh][m] * oRun[qh][m][d]) : 0.0f;
                }
            }

            for (int kk = 0; kk < tileLen; ++kk) {
                const unsigned char* __restrict__ vRow = v
                    + static_cast<size_t>(kStart + kk)
                    * static_cast<size_t>(nBlocksPerRow)
                    * static_cast<size_t>(Q8_0_BLOCK_BYTES);
                const unsigned char* __restrict__ blkPtr = vRow
                    + static_cast<size_t>(headBlockBase + blk)
                    * static_cast<size_t>(Q8_0_BLOCK_BYTES);
                const float bscale =
                    __half2float(*reinterpret_cast<const __half*>(blkPtr));
                const signed char qi =
                    reinterpret_cast<const signed char*>(blkPtr)[2 + in];
                const float v_val = bscale * static_cast<float>(qi);

                #pragma unroll
                for (int qh = 0; qh < NQMAX; ++qh) {
                    #pragma unroll
                    for (int m = 0; m < BQ; ++m) {
                        acc_v[qh][m] += scores[qh][m][kk] * v_val;
                    }
                }
            }

            #pragma unroll
            for (int qh = 0; qh < NQMAX; ++qh) {
                if (qh >= nQPerKv) continue;
                #pragma unroll
                for (int m = 0; m < BQ; ++m) {
                    oRun[qh][m][d] = acc_v[qh][m];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int qh = 0; qh < NQMAX; ++qh) {
        if (qh >= nQPerKv) continue;
        #pragma unroll
        for (int m = 0; m < BQ; ++m) {
            if (m > lastM) continue;
            const float invL = (l_reg[qh][m] > 0.0f) ? (1.0f / l_reg[qh][m]) : 0.0f;
            for (int d = lid; d < headDim; d += ATTN_FLASH_PREFILL_LOCAL) {
                oVecs[qh][m][d] = oRun[qh][m][d] * invL;
            }
        }
    }
}
