// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// P3.b — TF32 tensor-core variant of attention_prefill_flash_f32_gqa.
//
// Same GQA-head-packed tiling as the scalar F32 GQA kernel (one workgroup
// per (kv-head, query-position), all nQPerKv query heads of the group
// handled together), but QK^T and P.V run on TF32 tensor cores.
//
// One CTA owns a single query position pq, so all M = nQPerKv query rows
// share ONE causal mask (kMax = positionOffset + pq + 1); there is no
// per-row masking inside the tile — only the partial last K-tile's
// out-of-range score columns are zeroed before the softmax.
//
// v2: to amortise shared-staging/sync overhead the K^T and V tiles are
// staged in STAGE-wide slabs (STAGE keys/d per __syncthreads) rather than
// per 8-wide MMA step, and a single scratch buffer `stg` is reused for the
// K^T stage, the V stage, and the per-d-tile O store (they live in
// disjoint phases). OOB slab elements are masked to 0 on stage, and OOB
// score columns are zeroed by assignment (kills any NaN), so no OOB global
// read and no NaN can reach the accumulator.
//
// TF32 (10-bit mantissa) rounds Q/K/V on stage → bit-NEAR (not bit-exact);
// dispatch is opt-in (MIMIRMIND_ATTN_TC_PREFILL=1), parity-gated.
//
// Layouts (row-major fp32, identical to attention_prefill_flash_f32_gqa):
//   q [T_q, nHeads, headDim]  k,v [T_k, nKvHeads, headDim]  out [T_q, nHeads, headDim]
//
// Launch: grid( nKvHeads, T_q, 1 )  block( 32, 1, 1 )  // one warp

#include <cuda_runtime.h>
#include <mma.h>

#include <math.h>   // for INFINITY

using namespace nvcuda;

#ifndef ATTN_TC_KTILE
#define ATTN_TC_KTILE 128            // keys per K-tile
#endif

#ifndef ATTN_TC_MAX_HEADDIM
#define ATTN_TC_MAX_HEADDIM 256      // Qwen3-Next head_dim; bounds shared
#endif

#ifndef ATTN_TC_M
#define ATTN_TC_M 16                 // MMA M (>= nQPerKv, padded)
#endif

#define ATTN_TC_TN 16                // MMA N
#define ATTN_TC_TK 8                 // TF32 mma contraction (m16n16k8)
#define ATTN_TC_STAGE 64             // K^T / V slab staged per __syncthreads

extern "C" __global__ __launch_bounds__(32)
void attention_prefill_flash_f32_gqa_tc(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
          float* __restrict__ out,
    const int                 T_q,
    const int                 nHeads,
    const int                 nKvHeads,
    const int                 headDim,
    const int* __restrict__   curLenPtr,
    const float               scale,
    const int                 slidingWindow)
{
    (void)T_q;

    const int hkv     = blockIdx.x;
    const int pq      = blockIdx.y;
    const int lane    = threadIdx.x;
    const int nQPerKv = nHeads / nKvHeads;

    const int qStride        = nHeads   * headDim;
    const int kvStride       = nKvHeads * headDim;
    const int positionOffset = curLenPtr[0];
    const int absPos         = positionOffset + pq;
    const int kMax           = absPos + 1;
    const int kMin           = (slidingWindow > 0 && kMax > slidingWindow)
                                 ? (kMax - slidingWindow) : 0;
    const int ktStart        = kMin / ATTN_TC_KTILE;
    const int nKTiles        = (kMax + ATTN_TC_KTILE - 1) / ATTN_TC_KTILE;

    __shared__ float qS  [ATTN_TC_M][ATTN_TC_MAX_HEADDIM];   // Q (F32)
    __shared__ float oRun[ATTN_TC_M][ATTN_TC_MAX_HEADDIM];   // running O
    __shared__ float sS  [ATTN_TC_M][ATTN_TC_KTILE];         // scores / P
    __shared__ float stg [ATTN_TC_STAGE][ATTN_TC_TN];        // K^T / V / O scratch
    __shared__ float mSh [ATTN_TC_M];
    __shared__ float lSh [ATTN_TC_M];
    __shared__ float aSh [ATTN_TC_M];

    for (int i = lane; i < ATTN_TC_M * headDim; i += 32) {
        const int r = i / headDim;
        const int d = i % headDim;
        const int hq = hkv * nQPerKv + r;
        qS[r][d]   = (r < nQPerKv)
                       ? q[static_cast<size_t>(pq) * qStride
                           + static_cast<size_t>(hq) * headDim + d]
                       : 0.0f;
        oRun[r][d] = 0.0f;
    }
    for (int r = lane; r < ATTN_TC_M; r += 32) {
        mSh[r] = -INFINITY; lSh[r] = 0.0f; aSh[r] = 1.0f;
    }
    __syncthreads();

    const int nDTiles = headDim / ATTN_TC_TN;

    for (int kt = ktStart; kt < nKTiles; ++kt) {
        const int kStartRaw = kt * ATTN_TC_KTILE;
        const int kStart    = (kStartRaw > kMin) ? kStartRaw : kMin;
        const int kEndRaw   = kStartRaw + ATTN_TC_KTILE;
        const int kEnd      = (kEndRaw < kMax) ? kEndRaw : kMax;
        const int tileLen   = kEnd - kStart;
        const int nNTiles   = (tileLen + ATTN_TC_TN - 1) / ATTN_TC_TN;

        // ---- QK^T: sS[M, .] = Q[M, D] . K[., D]^T ----------------------
        for (int nt = 0; nt < nNTiles; ++nt) {
            const int nBase = nt * ATTN_TC_TN;

            wmma::fragment<wmma::accumulator, ATTN_TC_M, ATTN_TC_TN,
                           ATTN_TC_TK, float> cFrag;
            wmma::fill_fragment(cFrag, 0.0f);

            for (int ds = 0; ds < headDim; ds += ATTN_TC_STAGE) {
                // Stage K^T slab: stg[dl][n] = K[kStart+nBase+n][ds+dl].
                for (int i = lane; i < ATTN_TC_STAGE * ATTN_TC_TN; i += 32) {
                    const int dl = i / ATTN_TC_TN;
                    const int nn = i % ATTN_TC_TN;
                    stg[dl][nn] = (nBase + nn < tileLen)
                        ? k[static_cast<size_t>(kStart + nBase + nn) * kvStride
                            + static_cast<size_t>(hkv) * headDim + (ds + dl)]
                        : 0.0f;
                }
                __syncthreads();

                for (int kk = 0; kk < ATTN_TC_STAGE; kk += ATTN_TC_TK) {
                    wmma::fragment<wmma::matrix_a, ATTN_TC_M, ATTN_TC_TN,
                                   ATTN_TC_TK, wmma::precision::tf32,
                                   wmma::row_major> aFrag;
                    wmma::fragment<wmma::matrix_b, ATTN_TC_M, ATTN_TC_TN,
                                   ATTN_TC_TK, wmma::precision::tf32,
                                   wmma::row_major> bFrag;
                    wmma::load_matrix_sync(aFrag, &qS[0][ds + kk],
                                           ATTN_TC_MAX_HEADDIM);
                    wmma::load_matrix_sync(bFrag, &stg[kk][0], ATTN_TC_TN);
#pragma unroll
                    for (int t = 0; t < aFrag.num_elements; ++t)
                        aFrag.x[t] = wmma::__float_to_tf32(aFrag.x[t]);
#pragma unroll
                    for (int t = 0; t < bFrag.num_elements; ++t)
                        bFrag.x[t] = wmma::__float_to_tf32(bFrag.x[t]);
                    wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
                }
                __syncthreads();
            }
            wmma::store_matrix_sync(&sS[0][nBase], cFrag, ATTN_TC_KTILE,
                                    wmma::mem_row_major);
            __syncthreads();
        }

        // Zero OOB score columns (all rows) — assignment kills any garbage.
        for (int i = lane; i < ATTN_TC_M * ATTN_TC_KTILE; i += 32) {
            const int c = i % ATTN_TC_KTILE;
            if (c >= tileLen) sS[i / ATTN_TC_KTILE][c] = 0.0f;
        }
        __syncthreads();

        // ---- Online softmax on sS[0..nQPerKv-1][0..tileLen-1] ----------
        if (lane < nQPerKv) {
            const int r = lane;
            float tileMax = -INFINITY;
            for (int c = 0; c < tileLen; ++c) {
                const float s = sS[r][c] * scale;
                sS[r][c] = s;
                if (s > tileMax) tileMax = s;
            }
            const float mPrev = mSh[r];
            const float mNew  = (mPrev > tileMax) ? mPrev : tileMax;
            const float alpha = expf(mPrev - mNew);
            float tileSum = 0.0f;
            for (int c = 0; c < tileLen; ++c) {
                const float e = expf(sS[r][c] - mNew);
                sS[r][c] = e;
                tileSum += e;
            }
            lSh[r] = alpha * lSh[r] + tileSum;
            mSh[r] = mNew;
            aSh[r] = alpha;
        }
        __syncthreads();

        // ---- P.V: oRun[M, D] = alpha * oRun + P[M, .] . V[., D] --------
        for (int dt = 0; dt < nDTiles; ++dt) {
            const int dBase = dt * ATTN_TC_TN;

            wmma::fragment<wmma::accumulator, ATTN_TC_M, ATTN_TC_TN,
                           ATTN_TC_TK, float> cFrag;
            wmma::fill_fragment(cFrag, 0.0f);

            for (int ns = 0; ns < tileLen; ns += ATTN_TC_STAGE) {
                // Stage V slab: stg[nl][d] = V[kStart+ns+nl][dBase+d].
                for (int i = lane; i < ATTN_TC_STAGE * ATTN_TC_TN; i += 32) {
                    const int nl = i / ATTN_TC_TN;
                    const int dd = i % ATTN_TC_TN;
                    stg[nl][dd] = (ns + nl < tileLen)
                        ? v[static_cast<size_t>(kStart + ns + nl) * kvStride
                            + static_cast<size_t>(hkv) * headDim + (dBase + dd)]
                        : 0.0f;
                }
                __syncthreads();

                for (int kk = 0; kk < ATTN_TC_STAGE; kk += ATTN_TC_TK) {
                    wmma::fragment<wmma::matrix_a, ATTN_TC_M, ATTN_TC_TN,
                                   ATTN_TC_TK, wmma::precision::tf32,
                                   wmma::row_major> aFrag;
                    wmma::fragment<wmma::matrix_b, ATTN_TC_M, ATTN_TC_TN,
                                   ATTN_TC_TK, wmma::precision::tf32,
                                   wmma::row_major> bFrag;
                    // P sub-tile [M, TK] from sS (cols >= tileLen are 0).
                    wmma::load_matrix_sync(aFrag, &sS[0][ns + kk], ATTN_TC_KTILE);
                    wmma::load_matrix_sync(bFrag, &stg[kk][0], ATTN_TC_TN);
#pragma unroll
                    for (int t = 0; t < aFrag.num_elements; ++t)
                        aFrag.x[t] = wmma::__float_to_tf32(aFrag.x[t]);
#pragma unroll
                    for (int t = 0; t < bFrag.num_elements; ++t)
                        bFrag.x[t] = wmma::__float_to_tf32(bFrag.x[t]);
                    wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
                }
                __syncthreads();
            }
            // Store O contribution to the scratch, then combine with alpha.
            wmma::store_matrix_sync(&stg[0][0], cFrag, ATTN_TC_TN,
                                    wmma::mem_row_major);
            __syncthreads();
            for (int i = lane; i < ATTN_TC_M * ATTN_TC_TN; i += 32) {
                const int r  = i / ATTN_TC_TN;
                const int dd = i % ATTN_TC_TN;
                oRun[r][dBase + dd] =
                    aSh[r] * oRun[r][dBase + dd] + stg[r][dd];
            }
            __syncthreads();
        }
    }

    for (int i = lane; i < ATTN_TC_M * headDim; i += 32) {
        const int r = i / headDim;
        const int d = i % headDim;
        if (r >= nQPerKv) continue;
        const int hq = hkv * nQPerKv + r;
        const float invL = (lSh[r] > 0.0f) ? (1.0f / lSh[r]) : 0.0f;
        out[static_cast<size_t>(pq) * qStride
            + static_cast<size_t>(hq) * headDim + d] = oRun[r][d] * invL;
    }
}
