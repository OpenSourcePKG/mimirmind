// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Step 3.1 — FP16 tensor-core flash-attention prefill (FA-2 style), the attn
// win the whole FP16-KV enabler was built for. Fixes the failures of the
// shelved single-warp TF32 attempt (attention_prefill_flash_f32_gqa_tc):
//   - M is filled with REAL query positions (a q-tile of ATTN_TC16_BQ rows),
//     not 8 padded GQA heads, so the m16 MMA tile is not half-wasted.
//   - operands are native FP16 (K/V come straight from the fp16 KV cache; Q is
//     cast F32->half once on stage), so there is no F32->TF32 staging cost.
//   - wmma m16n16k16 (K=16 contraction) halves the k-steps vs TF32's k8.
//
// One CTA owns (query-head hq, q-tile). It streams the causal K/V range in
// 16-key tiles, does QK^T and P.V on tensor cores, keeps an online-softmax
// running (m,l,O). Per-ROW causal masking (each of the BQ query positions has
// its own kMax) is applied to the score tile before the softmax.
//
// Single-warp first (block = 32); a multi-warp split is Step 3.2. Runs only
// when the KV cache is fp16 (the fp16-KV enabler); f32 KV keeps the scalar
// head-packed kernel. Bit-NEAR (fp16) — parity-gated with a tolerance.
//
// Layouts: q [T_q, nHeads, headDim] f32; k,v [T_k, nKvHeads, headDim] __half;
//          out [T_q, nHeads, headDim] f32.
// Launch: grid( nHeads, ceil(T_q / BQ), 1 ), block( 32, 1, 1 ).

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <math.h>   // INFINITY

using namespace nvcuda;

#ifndef ATTN_TC16_BQ
#define ATTN_TC16_BQ 16          // query rows per CTA == MMA M
#endif
#ifndef ATTN_TC16_BK
#define ATTN_TC16_BK 16          // keys per K-tile == MMA N
#endif
#ifndef ATTN_TC16_MAXHD
#define ATTN_TC16_MAXHD 256      // headDim bound (Qwen3-Next = 256)
#endif

#define TC16_MMA_K 16            // wmma m16n16k16 contraction

extern "C" __global__ __launch_bounds__(32)
void attention_prefill_flash_fp16_tc(
    const float*  __restrict__ q,       // [T_q, nHeads, headDim] f32
    const __half* __restrict__ k,       // [T_k, nKvHeads, headDim] f16
    const __half* __restrict__ v,       // [T_k, nKvHeads, headDim] f16
          float*  __restrict__ out,     // [T_q, nHeads, headDim] f32
    const int                  T_q,
    const int                  nHeads,
    const int                  nKvHeads,
    const int                  headDim,
    const int*    __restrict__ curLenPtr,
    const float                scale,
    const int                  slidingWindow)
{
    (void)slidingWindow;
    const int hq   = blockIdx.x;
    const int q0   = blockIdx.y * ATTN_TC16_BQ;   // first query position
    const int lane = threadIdx.x;                 // 0..31

    const int qStride        = nHeads   * headDim;
    const int kvStride       = nKvHeads * headDim;
    const int hkv            = (hq * nKvHeads) / nHeads;
    const int positionOffset = curLenPtr[0];

    // Per-CTA causal upper bound = last query row's kMax.
    const int lastPq = q0 + ATTN_TC16_BQ - 1;
    const int kMaxTile = positionOffset + lastPq + 1;   // exclusive
    const int nKTiles  = (kMaxTile + ATTN_TC16_BK - 1) / ATTN_TC16_BK;

    __shared__ __half qS[ATTN_TC16_BQ][ATTN_TC16_MAXHD];   // Q (f16)
    __shared__ __half kS[ATTN_TC16_BK][ATTN_TC16_MAXHD];   // K-tile (f16)
    __shared__ __half pS[ATTN_TC16_BQ][ATTN_TC16_BK];      // softmaxed P (f16)
    __shared__ float  sS[ATTN_TC16_BQ][ATTN_TC16_BK];      // scores (f32)
    __shared__ float  oRun[ATTN_TC16_BQ][ATTN_TC16_MAXHD]; // running O (f32)
    __shared__ float  oT [ATTN_TC16_BQ][ATTN_TC16_BK];     // P.V d-tile
    __shared__ float  mSh[ATTN_TC16_BQ], lSh[ATTN_TC16_BQ], aSh[ATTN_TC16_BQ];

    // Stage Q (cast f32 -> f16), init running state / O.
    for (int i = lane; i < ATTN_TC16_BQ * headDim; i += 32) {
        const int m = i / headDim, d = i % headDim;
        const int pq = q0 + m;
        qS[m][d] = (pq < T_q)
                     ? __float2half(q[(size_t)pq * qStride + (size_t)hq * headDim + d])
                     : __float2half(0.0f);
        oRun[m][d] = 0.0f;
    }
    for (int m = lane; m < ATTN_TC16_BQ; m += 32) {
        mSh[m] = -INFINITY; lSh[m] = 0.0f; aSh[m] = 1.0f;
    }
    __syncthreads();

    const int nDTiles = headDim / ATTN_TC16_BK;   // headDim multiple of 16

    for (int kt = 0; kt < nKTiles; ++kt) {
        const int kBase = kt * ATTN_TC16_BK;

        // Stage K-tile (mask OOB keys -> 0; masked cols get -inf later anyway).
        for (int i = lane; i < ATTN_TC16_BK * headDim; i += 32) {
            const int n = i / headDim, d = i % headDim;
            const int kk = kBase + n;
            kS[n][d] = (kk < kMaxTile)
                         ? k[(size_t)kk * kvStride + (size_t)hkv * headDim + d]
                         : __float2half(0.0f);
        }
        __syncthreads();

        // ---- QK^T: sS[BQ, BK] = Q[BQ, D] . K[BK, D]^T (f16 MMA) ----------
        {
            wmma::fragment<wmma::accumulator, 16, 16, TC16_MMA_K, float> acc;
            wmma::fill_fragment(acc, 0.0f);
            for (int dc = 0; dc < headDim; dc += TC16_MMA_K) {
                wmma::fragment<wmma::matrix_a, 16, 16, TC16_MMA_K, __half,
                               wmma::row_major> aFrag;
                wmma::fragment<wmma::matrix_b, 16, 16, TC16_MMA_K, __half,
                               wmma::col_major> bFrag;   // K^T: [d, n] from [n, d]
                wmma::load_matrix_sync(aFrag, &qS[0][dc], ATTN_TC16_MAXHD);
                wmma::load_matrix_sync(bFrag, &kS[0][dc], ATTN_TC16_MAXHD);
                wmma::mma_sync(acc, aFrag, bFrag, acc);
            }
            wmma::store_matrix_sync(&sS[0][0], acc, ATTN_TC16_BK,
                                    wmma::mem_row_major);
        }
        __syncthreads();

        // ---- per-row causal mask + scale, then online softmax ----------
        // One lane per query row.
        if (lane < ATTN_TC16_BQ) {
            const int m  = lane;
            const int pq = q0 + m;
            const int kMax_m = positionOffset + pq + 1;   // exclusive
            float tileMax = -INFINITY;
            for (int n = 0; n < ATTN_TC16_BK; ++n) {
                const int kk = kBase + n;
                float s = (kk < kMax_m) ? sS[m][n] * scale : -INFINITY;
                sS[m][n] = s;
                if (s > tileMax) tileMax = s;
            }
            const float mPrev = mSh[m];
            const float mNew  = (mPrev > tileMax) ? mPrev : tileMax;
            const float alpha = expf(mPrev - mNew);
            float tileSum = 0.0f;
            for (int n = 0; n < ATTN_TC16_BK; ++n) {
                const float e = (sS[m][n] > -INFINITY) ? expf(sS[m][n] - mNew) : 0.0f;
                pS[m][n] = __float2half(e);
                tileSum += e;
            }
            lSh[m] = alpha * lSh[m] + tileSum;
            mSh[m] = mNew;
            aSh[m] = alpha;
        }
        __syncthreads();

        // ---- P.V: oRun[BQ, D] = alpha*oRun + P[BQ, BK] . V[BK, D] -------
        for (int dt = 0; dt < nDTiles; ++dt) {
            const int dBase = dt * ATTN_TC16_BK;
            // Stage V d-tile [BK, 16] f16 (mask OOB keys).
            wmma::fragment<wmma::accumulator, 16, 16, TC16_MMA_K, float> acc;
            wmma::fill_fragment(acc, 0.0f);
            {
                wmma::fragment<wmma::matrix_a, 16, 16, TC16_MMA_K, __half,
                               wmma::row_major> aFrag;   // P[BQ, BK]
                wmma::fragment<wmma::matrix_b, 16, 16, TC16_MMA_K, __half,
                               wmma::row_major> bFrag;   // V[BK, 16d]
                wmma::load_matrix_sync(aFrag, &pS[0][0], ATTN_TC16_BK);
                // V[n][dBase+d] straight from kS-style read into a frag needs a
                // staged tile; reuse kS as scratch for the V d-tile.
                for (int i = lane; i < ATTN_TC16_BK * ATTN_TC16_BK; i += 32) {
                    const int n = i / ATTN_TC16_BK, d = i % ATTN_TC16_BK;
                    const int kk = kBase + n;
                    kS[n][d] = (kk < kMaxTile)
                        ? v[(size_t)kk * kvStride + (size_t)hkv * headDim + (dBase + d)]
                        : __float2half(0.0f);
                }
                __syncthreads();
                wmma::load_matrix_sync(bFrag, &kS[0][0], ATTN_TC16_MAXHD);
                wmma::mma_sync(acc, aFrag, bFrag, acc);
            }
            wmma::store_matrix_sync(&oT[0][0], acc, ATTN_TC16_BK,
                                    wmma::mem_row_major);
            __syncthreads();
            for (int i = lane; i < ATTN_TC16_BQ * ATTN_TC16_BK; i += 32) {
                const int m = i / ATTN_TC16_BK, d = i % ATTN_TC16_BK;
                oRun[m][dBase + d] = aSh[m] * oRun[m][dBase + d] + oT[m][d];
            }
            __syncthreads();
        }
    }

    // Normalise + write.
    for (int i = lane; i < ATTN_TC16_BQ * headDim; i += 32) {
        const int m = i / headDim, d = i % headDim;
        const int pq = q0 + m;
        if (pq >= T_q) continue;
        const float invL = (lSh[m] > 0.0f) ? (1.0f / lSh[m]) : 0.0f;
        out[(size_t)pq * qStride + (size_t)hq * headDim + d] = oRun[m][d] * invL;
    }
}
