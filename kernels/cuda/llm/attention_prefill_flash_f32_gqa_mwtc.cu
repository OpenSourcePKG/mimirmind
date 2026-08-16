// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Multi-warp TF32 tensor-core flash-attention prefill (FA-2 style) for the
// F32 KV path — the F32 sibling of Step 3.2's fp16 kernel. This is the
// kernel that actually runs on Qwen3-Next's prefill attention (whose K/V
// are freshly-projected F32 scratch, so kvDtype is always F32 — the fp16
// prefill-attn kernels are never reached). It attacks the same two failures
// the shelved single-warp P3.b (attention_prefill_flash_f32_gqa_tc) had:
//   - M is filled with 16 REAL query positions (a q-tile) per warp, not the
//     8 padded GQA heads P3.b used (half the m16 MMA wasted).
//   - the CTA runs MULTIPLE warps so the wmma pipeline latency that
//     serialised P3.b's single warp is hidden.
// while keeping P3.a's GQA-head-packing win (K/V staged once, shared).
//
// Structure: one CTA owns (kv-head hkv, q-tile of BQ=16 positions, head-half
// hh). It runs HPB warps; warp w handles query-head hkv*nQPerKv + hh*HPB + w
// for the same 16 positions. The K/V tile is staged ONCE into shared memory
// and read by all warps (GQA bandwidth win). The GQA group of nQPerKv heads
// is split into ceil(nQPerKv/HPB) head-halves (grid.z) so the per-head
// resident state (Q + running O in shared, F32 = 2x fp16's bytes) fits the
// dynamic-smem budget — K/V is therefore re-read once per head-half
// (nQPerKv/HPB times, still an HPB-way bandwidth reduction vs one WG/head).
//
// QK^T and P.V run on TF32 tensor cores (m16n16k8), operands cast F32->TF32
// on load (10-bit mantissa => bit-NEAR, parity-gated). Online softmax keeps
// (m, l, O) in F32 with PER-ROW causal masking. wmma layout opacity forces
// the per-row alpha rescale of O through shared memory (like P3.b's oRun).
//
// Opt-in MIMIRMIND_ATTN_F32_MWTC=1 (needs kvDtype=F32, nQPerKv in
// [2, MAXHEADS], headDim<=256 and %16==0). Dynamic smem, host opts in.
//
// Layouts: q,out [T_q, nHeads, headDim] f32; k,v [T_k, nKvHeads, headDim] f32.
// Launch: grid( nKvHeads, ceil(T_q/BQ), ceil(nQPerKv/HPB) ),
//         block( 32*warpsPerBlock, 1, 1 ), sharedMemBytes = smem(headDim, HPB).

#include <cuda_runtime.h>
#include <mma.h>

#include <cstddef>  // std::size_t
#include <math.h>   // INFINITY

using namespace nvcuda;

#ifndef ATTN_MW_BQ
#define ATTN_MW_BQ 16            // query rows per CTA == MMA M
#endif
#ifndef ATTN_MW_BK
#define ATTN_MW_BK 16            // keys per K-tile == MMA N
#endif
#ifndef ATTN_MW_MAXHD
#define ATTN_MW_MAXHD 256        // headDim bound (Qwen3-Next = 256)
#endif
#ifndef ATTN_MW_HPB
#define ATTN_MW_HPB 2            // query heads (== warps) per CTA head-half
#endif                          // (bounded by the 99 KiB sm_121 dyn-smem cap:
                                //  qS+oRun = 2*HPB*16*headDim*4)

#define MW_MMA_K 8               // TF32 wmma m16n16k8 contraction

// Dynamic-shared layout; regions padded to 128 bytes. The host computes the
// identical total. Sized by HPB (not nQPerKv) so F32 Q+O stay in budget.
struct MwSmem {
    float* qS;     // [HPB][BQ][headDim]  Q (staged once, cast tf32 on MMA load)
    float* kvS;    // [BK][headDim]       shared K, then re-staged as V
    float* oRun;   // [HPB][BQ][headDim]  running O (per-row rescale)
    float* sS;     // [HPB][BQ][BK]       QK^T scores
    float* pS;     // [HPB][BQ][BK]       softmaxed P (tf32 source)
    float* oT;     // [HPB][BQ][BK]       P.V d-tile result scratch
    float* mSh;    // [HPB][BQ]
    float* lSh;    // [HPB][BQ]
    float* aSh;    // [HPB][BQ]
};

static __device__ __forceinline__ std::size_t mwAlign128(std::size_t n) {
    return (n + 127u) & ~static_cast<std::size_t>(127u);
}

static __device__ __forceinline__ MwSmem mwCarve(char* base, int hpb, int hd) {
    MwSmem s;
    std::size_t off = 0;
    auto take = [&](std::size_t bytes) -> float* {
        float* p = reinterpret_cast<float*>(base + off);
        off += mwAlign128(bytes);
        return p;
    };
    s.qS   = take((std::size_t)hpb * ATTN_MW_BQ * hd * sizeof(float));
    s.kvS  = take((std::size_t)ATTN_MW_BK * hd * sizeof(float));
    s.oRun = take((std::size_t)hpb * ATTN_MW_BQ * hd * sizeof(float));
    s.sS   = take((std::size_t)hpb * ATTN_MW_BQ * ATTN_MW_BK * sizeof(float));
    s.pS   = take((std::size_t)hpb * ATTN_MW_BQ * ATTN_MW_BK * sizeof(float));
    s.oT   = take((std::size_t)hpb * ATTN_MW_BQ * ATTN_MW_BK * sizeof(float));
    s.mSh  = take((std::size_t)hpb * ATTN_MW_BQ * sizeof(float));
    s.lSh  = take((std::size_t)hpb * ATTN_MW_BQ * sizeof(float));
    s.aSh  = take((std::size_t)hpb * ATTN_MW_BQ * sizeof(float));
    return s;
}

extern __shared__ char mw_smem[];

extern "C" __global__ __launch_bounds__(32 * ATTN_MW_HPB)
void attention_prefill_flash_f32_gqa_mwtc(
    const float* __restrict__ q,        // [T_q, nHeads, headDim] f32
    const float* __restrict__ k,        // [T_k, nKvHeads, headDim] f32
    const float* __restrict__ v,        // [T_k, nKvHeads, headDim] f32
          float* __restrict__ out,      // [T_q, nHeads, headDim] f32
    const int                 T_q,
    const int                 nHeads,
    const int                 nKvHeads,
    const int                 headDim,
    const int*   __restrict__ curLenPtr,
    const float               scale,
    const int                 slidingWindow)
{
    (void)slidingWindow;
    const int hkv      = blockIdx.x;                 // KV-head index
    const int q0       = blockIdx.y * ATTN_MW_BQ;    // first query position
    const int hh       = blockIdx.z;                 // head-half index
    const int nQPerKv  = nHeads / nKvHeads;
    const int nWarps   = blockDim.x >> 5;            // warps this CTA (<= HPB)
    const int warp     = threadIdx.x >> 5;
    const int lane     = threadIdx.x & 31;
    const int qhInGroup = hh * ATTN_MW_HPB + warp;    // 0..nQPerKv-1 (maybe OOB)
    const int active   = (qhInGroup < nQPerKv);
    const int hq       = hkv * nQPerKv + qhInGroup;   // this warp's query head

    const int qStride        = nHeads   * headDim;
    const int kvStride       = nKvHeads * headDim;
    const int positionOffset = curLenPtr[0];

    const int lastPq   = q0 + ATTN_MW_BQ - 1;
    const int kMaxTile = positionOffset + lastPq + 1;   // exclusive
    const int nKTiles  = (kMaxTile + ATTN_MW_BK - 1) / ATTN_MW_BK;
    const int nDTiles  = headDim / ATTN_MW_BK;          // d-tiles of 16 (P.V N)

    MwSmem sm = mwCarve(mw_smem, ATTN_MW_HPB, headDim);
    float* qSW   = sm.qS   + (std::size_t)warp * ATTN_MW_BQ * headDim;
    float* oRunW = sm.oRun + (std::size_t)warp * ATTN_MW_BQ * headDim;
    float* sSW   = sm.sS   + (std::size_t)warp * ATTN_MW_BQ * ATTN_MW_BK;
    float* pSW   = sm.pS   + (std::size_t)warp * ATTN_MW_BQ * ATTN_MW_BK;
    float* oTW   = sm.oT   + (std::size_t)warp * ATTN_MW_BQ * ATTN_MW_BK;
    float* mShW  = sm.mSh  + (std::size_t)warp * ATTN_MW_BQ;
    float* lShW  = sm.lSh  + (std::size_t)warp * ATTN_MW_BQ;
    float* aShW  = sm.aSh  + (std::size_t)warp * ATTN_MW_BQ;

    // Stage this warp's Q [BQ][headDim] (masked), init O / softmax state.
    if (active) {
        for (int i = lane; i < ATTN_MW_BQ * headDim; i += 32) {
            const int m = i / headDim, d = i % headDim;
            const int pq = q0 + m;
            qSW[i] = (pq < T_q)
                ? q[(std::size_t)pq * qStride + (std::size_t)hq * headDim + d]
                : 0.0f;
            oRunW[i] = 0.0f;
        }
        for (int m = lane; m < ATTN_MW_BQ; m += 32) {
            mShW[m] = -INFINITY; lShW[m] = 0.0f; aShW[m] = 1.0f;
        }
    }
    __syncthreads();

    for (int kt = 0; kt < nKTiles; ++kt) {
        const int kBase = kt * ATTN_MW_BK;

        // Stage K-tile [BK][headDim] into shared, ONCE for all warps.
        for (int i = threadIdx.x; i < ATTN_MW_BK * headDim; i += blockDim.x) {
            const int n = i / headDim, d = i % headDim;
            const int kk = kBase + n;
            sm.kvS[i] = (kk < kMaxTile)
                ? k[(std::size_t)kk * kvStride + (std::size_t)hkv * headDim + d]
                : 0.0f;
        }
        __syncthreads();

        // ---- QK^T: sSW[BQ][BK] = Q[BQ,D] . K[BK,D]^T (TF32 MMA) ----------
        if (active) {
            wmma::fragment<wmma::accumulator, 16, 16, MW_MMA_K, float> acc;
            wmma::fill_fragment(acc, 0.0f);
            for (int dc = 0; dc < headDim; dc += MW_MMA_K) {
                wmma::fragment<wmma::matrix_a, 16, 16, MW_MMA_K,
                               wmma::precision::tf32, wmma::row_major> aFrag;
                wmma::fragment<wmma::matrix_b, 16, 16, MW_MMA_K,
                               wmma::precision::tf32, wmma::col_major> bFrag;
                wmma::load_matrix_sync(aFrag, qSW + dc, headDim);       // Q[16][8]
                wmma::load_matrix_sync(bFrag, sm.kvS + dc, headDim);    // K^T[8][16]
#pragma unroll
                for (int t = 0; t < aFrag.num_elements; ++t)
                    aFrag.x[t] = wmma::__float_to_tf32(aFrag.x[t]);
#pragma unroll
                for (int t = 0; t < bFrag.num_elements; ++t)
                    bFrag.x[t] = wmma::__float_to_tf32(bFrag.x[t]);
                wmma::mma_sync(acc, aFrag, bFrag, acc);
            }
            wmma::store_matrix_sync(sSW, acc, ATTN_MW_BK, wmma::mem_row_major);
        }
        __syncwarp();

        // ---- per-row causal mask + scale + online softmax ---------------
        if (active && lane < ATTN_MW_BQ) {
            const int m  = lane;
            const int pq = q0 + m;
            const int kMax_m = positionOffset + pq + 1;   // exclusive
            float tileMax = -INFINITY;
            for (int n = 0; n < ATTN_MW_BK; ++n) {
                const int kk = kBase + n;
                float s = (kk < kMax_m) ? sSW[m * ATTN_MW_BK + n] * scale
                                        : -INFINITY;
                sSW[m * ATTN_MW_BK + n] = s;
                if (s > tileMax) tileMax = s;
            }
            const float mPrev = mShW[m];
            const float mNew  = (mPrev > tileMax) ? mPrev : tileMax;
            const float alpha = expf(mPrev - mNew);
            float tileSum = 0.0f;
            for (int n = 0; n < ATTN_MW_BK; ++n) {
                const float sv = sSW[m * ATTN_MW_BK + n];
                const float e  = (sv > -INFINITY) ? expf(sv - mNew) : 0.0f;
                pSW[m * ATTN_MW_BK + n] = e;
                tileSum += e;
            }
            lShW[m] = alpha * lShW[m] + tileSum;
            mShW[m] = mNew;
            aShW[m] = alpha;
        }
        __syncthreads();

        // Re-stage the same key tile as V [BK][headDim] into shared kvS.
        for (int i = threadIdx.x; i < ATTN_MW_BK * headDim; i += blockDim.x) {
            const int n = i / headDim, d = i % headDim;
            const int kk = kBase + n;
            sm.kvS[i] = (kk < kMaxTile)
                ? v[(std::size_t)kk * kvStride + (std::size_t)hkv * headDim + d]
                : 0.0f;
        }
        __syncthreads();

        // ---- P.V: oRun[BQ][D] = alpha*oRun + P[BQ,BK] . V[BK,D] ----------
        if (active) {
            for (int dt = 0; dt < nDTiles; ++dt) {
                const int dBase = dt * ATTN_MW_BK;
                wmma::fragment<wmma::accumulator, 16, 16, MW_MMA_K, float> acc;
                wmma::fill_fragment(acc, 0.0f);
                for (int kk = 0; kk < ATTN_MW_BK; kk += MW_MMA_K) {
                    wmma::fragment<wmma::matrix_a, 16, 16, MW_MMA_K,
                                   wmma::precision::tf32, wmma::row_major> aFrag;
                    wmma::fragment<wmma::matrix_b, 16, 16, MW_MMA_K,
                                   wmma::precision::tf32, wmma::row_major> bFrag;
                    wmma::load_matrix_sync(aFrag, pSW + kk, ATTN_MW_BK);     // P[16][8]
                    wmma::load_matrix_sync(bFrag, sm.kvS + kk * headDim + dBase,
                                           headDim);                         // V[8][16]
#pragma unroll
                    for (int t = 0; t < aFrag.num_elements; ++t)
                        aFrag.x[t] = wmma::__float_to_tf32(aFrag.x[t]);
#pragma unroll
                    for (int t = 0; t < bFrag.num_elements; ++t)
                        bFrag.x[t] = wmma::__float_to_tf32(bFrag.x[t]);
                    wmma::mma_sync(acc, aFrag, bFrag, acc);
                }
                wmma::store_matrix_sync(oTW, acc, ATTN_MW_BK, wmma::mem_row_major);
                __syncwarp();
                for (int i = lane; i < ATTN_MW_BQ * ATTN_MW_BK; i += 32) {
                    const int m = i / ATTN_MW_BK, d = i % ATTN_MW_BK;
                    oRunW[m * headDim + (dBase + d)] =
                        aShW[m] * oRunW[m * headDim + (dBase + d)]
                        + oTW[m * ATTN_MW_BK + d];
                }
                __syncwarp();
            }
        }
        __syncthreads();
    }

    // ---- Normalise + write -----------------------------------------------
    if (active) {
        for (int i = lane; i < ATTN_MW_BQ * headDim; i += 32) {
            const int m = i / headDim, d = i % headDim;
            const int pq = q0 + m;
            if (pq >= T_q) continue;
            const float invL = (lShW[m] > 0.0f) ? (1.0f / lShW[m]) : 0.0f;
            out[(std::size_t)pq * qStride + (std::size_t)hq * headDim + d] =
                oRunW[m * headDim + d] * invL;
        }
    }
}
