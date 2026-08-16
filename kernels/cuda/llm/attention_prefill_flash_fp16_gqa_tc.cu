// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Step 3.2 — GQA-head-packed, multi-warp FP16 tensor-core flash-attention
// prefill (FA-2 style). This is the kernel the whole FP16-KV enabler was
// built for: it fuses the TWO wins the single-warp Step 3.1 kernel
// (attention_prefill_flash_fp16_tc) and the shelved P3.b TF32 kernel
// (attention_prefill_flash_f32_gqa_tc) each got only half of.
//
//   * Step 3.1 filled the MMA M with 16 REAL query positions of ONE head,
//     but re-read K/V per query head — no GQA bandwidth win, single warp.
//   * P3.b packed the whole GQA group into M (K/V read once per group) but
//     M = nQPerKv (=8) was padded to 16 => half the MMA wasted, single warp.
//
// This kernel gets BOTH: one CTA owns (kv-head hkv, q-tile of BQ=16 query
// positions). It runs nQPerKv WARPS; warp w handles query-head
// hq = hkv*nQPerKv + w for the SAME 16 query positions, so M = 16 real
// rows (MMA filled) AND the K/V tile is staged ONCE into shared memory and
// read by every warp (the GQA bandwidth win == read K/V once per group,
// not once per head). The nQPerKv independent warps give the instruction-
// level parallelism that hides the wmma pipeline latency that serialised
// both single-warp predecessors.
//
// Per-ROW causal masking (each of the 16 query positions has its own kMax)
// is applied to the QK^T score tile before the online softmax, exactly as
// in Step 3.1.
//
// wmma layout opacity forces the split: the O accumulator needs a per-ROW
// alpha rescale each k-tile, and wmma fragments do not expose their row
// mapping, so O lives in shared memory (row-indexed rescale, like 3.1's
// oRun). Q is never rescaled, so it is loaded ONCE into persistent
// register a-fragments (qFrag) before the k-loop — that keeps the big
// [nWarps][BQ][headDim] Q array out of shared memory.
//
// Shared-memory budget (dynamic, host opts in via
// CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES; falls back to the scalar
// fp16 kernel if the opt-in is rejected). For headDim=256, nQPerKv=8 the
// dominant term is oRun = nWarps*BQ*headDim*4 = 128 KiB.
//
// Opt-in MIMIRMIND_ATTN_FP16_GQA_TC=1 (needs kvDtype=fp16, nQPerKv in
// [2, MAXWARPS], headDim<=256 and %16==0). Bit-NEAR (fp16) — parity-gated.
//
// Layouts: q [T_q, nHeads, headDim] f32; k,v [T_k, nKvHeads, headDim] __half;
//          out [T_q, nHeads, headDim] f32.
// Launch: grid( nKvHeads, ceil(T_q/BQ), 1 ), block( 32*nQPerKv, 1, 1 ),
//         sharedMemBytes = smem_bytes(headDim, nQPerKv).

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <cstddef>  // std::size_t
#include <math.h>   // INFINITY

using namespace nvcuda;

#ifndef ATTN_G16_BQ
#define ATTN_G16_BQ 16           // query rows per CTA == MMA M
#endif
#ifndef ATTN_G16_BK
#define ATTN_G16_BK 16           // keys per K-tile == MMA N
#endif
#ifndef ATTN_G16_MAXHD
#define ATTN_G16_MAXHD 256       // headDim bound (Qwen3-Next = 256)
#endif
#ifndef ATTN_G16_MAXWARPS
#define ATTN_G16_MAXWARPS 8      // max nQPerKv (== warps per CTA)
#endif

#define G16_MMA_K 16             // wmma m16n16k16 contraction
#define G16_MAXDTILES (ATTN_G16_MAXHD / ATTN_G16_BK)   // <= 16

// Dynamic-shared layout. All regions are carved from one extern block; each
// region is padded to 128 bytes so half/float wmma loads stay aligned. The
// host computes the identical total via smemBytesFp16GqaTc() below.
struct Smem {
    float* oRun;   // [nW][BQ][headDim]  running O (needs per-row rescale)
    __half* kvS;   // [BK][headDim]      shared K, then re-staged as V
    __half* qStg;  // [nW][BQ][BK]       per-warp Q d-tile cast scratch
    float* sS;     // [nW][BQ][BK]       QK^T scores / (unused after softmax)
    __half* pS;    // [nW][BQ][BK]       softmaxed P (fp16 MMA operand)
    float* oT;     // [nW][BQ][BK]       P.V d-tile result scratch
    float* mSh;    // [nW][BQ]
    float* lSh;    // [nW][BQ]
    float* aSh;    // [nW][BQ]
};

static __device__ __forceinline__ std::size_t align128(std::size_t n) {
    return (n + 127u) & ~static_cast<std::size_t>(127u);
}

static __device__ __forceinline__ Smem carveSmem(char* base, int nW, int hd) {
    Smem s;
    std::size_t off = 0;
    auto take = [&](std::size_t bytes) -> char* {
        char* p = base + off;
        off += align128(bytes);
        return p;
    };
    s.oRun = reinterpret_cast<float*>( take((std::size_t)nW * ATTN_G16_BQ * hd * sizeof(float)) );
    s.kvS  = reinterpret_cast<__half*>(take((std::size_t)ATTN_G16_BK * hd * sizeof(__half)) );
    s.qStg = reinterpret_cast<__half*>(take((std::size_t)nW * ATTN_G16_BQ * ATTN_G16_BK * sizeof(__half)) );
    s.sS   = reinterpret_cast<float*>( take((std::size_t)nW * ATTN_G16_BQ * ATTN_G16_BK * sizeof(float)) );
    s.pS   = reinterpret_cast<__half*>(take((std::size_t)nW * ATTN_G16_BQ * ATTN_G16_BK * sizeof(__half)) );
    s.oT   = reinterpret_cast<float*>( take((std::size_t)nW * ATTN_G16_BQ * ATTN_G16_BK * sizeof(float)) );
    s.mSh  = reinterpret_cast<float*>( take((std::size_t)nW * ATTN_G16_BQ * sizeof(float)) );
    s.lSh  = reinterpret_cast<float*>( take((std::size_t)nW * ATTN_G16_BQ * sizeof(float)) );
    s.aSh  = reinterpret_cast<float*>( take((std::size_t)nW * ATTN_G16_BQ * sizeof(float)) );
    return s;
}

extern __shared__ char g16_smem[];

extern "C" __global__ __launch_bounds__(32 * ATTN_G16_MAXWARPS)
void attention_prefill_flash_fp16_gqa_tc(
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
    const int hkv     = blockIdx.x;                 // KV-head index
    const int q0      = blockIdx.y * ATTN_G16_BQ;   // first query position
    const int nQPerKv = nHeads / nKvHeads;          // == warps per CTA
    const int warp    = threadIdx.x >> 5;           // 0..nQPerKv-1
    const int lane    = threadIdx.x & 31;           // 0..31
    const int hq      = hkv * nQPerKv + warp;        // this warp's query head

    const int qStride        = nHeads   * headDim;
    const int kvStride       = nKvHeads * headDim;
    const int positionOffset = curLenPtr[0];

    // Per-CTA causal upper bound = last query row's kMax (exclusive).
    const int lastPq   = q0 + ATTN_G16_BQ - 1;
    const int kMaxTile = positionOffset + lastPq + 1;
    const int nKTiles  = (kMaxTile + ATTN_G16_BK - 1) / ATTN_G16_BK;
    const int nDTiles  = headDim / ATTN_G16_BK;     // headDim multiple of 16

    Smem sm = carveSmem(g16_smem, nQPerKv, headDim);

    // Per-warp views into the packed shared arrays.
    float*  oRunW = sm.oRun + (std::size_t)warp * ATTN_G16_BQ * headDim;
    __half* qStgW = sm.qStg + (std::size_t)warp * ATTN_G16_BQ * ATTN_G16_BK;
    float*  sSW   = sm.sS   + (std::size_t)warp * ATTN_G16_BQ * ATTN_G16_BK;
    __half* pSW   = sm.pS   + (std::size_t)warp * ATTN_G16_BQ * ATTN_G16_BK;
    float*  oTW   = sm.oT   + (std::size_t)warp * ATTN_G16_BQ * ATTN_G16_BK;
    float*  mShW  = sm.mSh  + (std::size_t)warp * ATTN_G16_BQ;
    float*  lShW  = sm.lSh  + (std::size_t)warp * ATTN_G16_BQ;
    float*  aShW  = sm.aSh  + (std::size_t)warp * ATTN_G16_BQ;

    // ---- Init running O / softmax state, load Q into persistent frags -----
    for (int i = lane; i < ATTN_G16_BQ * headDim; i += 32) {
        oRunW[i] = 0.0f;
    }
    for (int m = lane; m < ATTN_G16_BQ; m += 32) {
        mShW[m] = -INFINITY; lShW[m] = 0.0f; aShW[m] = 1.0f;
    }

    // Q a-fragments, resident across the whole k-loop (Q is never rescaled).
    wmma::fragment<wmma::matrix_a, 16, 16, G16_MMA_K, __half, wmma::row_major>
        qFrag[G16_MAXDTILES];
    for (int dt = 0; dt < nDTiles; ++dt) {
        const int dBase = dt * ATTN_G16_BK;
        // Stage this warp's Q d-tile [BQ][16] (cast f32->f16). Warp-private.
        for (int i = lane; i < ATTN_G16_BQ * ATTN_G16_BK; i += 32) {
            const int m = i / ATTN_G16_BK, d = i % ATTN_G16_BK;
            const int pq = q0 + m;
            qStgW[i] = (pq < T_q)
                ? __float2half(q[(std::size_t)pq * qStride
                                 + (std::size_t)hq * headDim + (dBase + d)])
                : __float2half(0.0f);
        }
        __syncwarp();
        wmma::load_matrix_sync(qFrag[dt], qStgW, ATTN_G16_BK);
        __syncwarp();
    }
    __syncthreads();

    // ---- Stream causal K/V range in BK-key tiles --------------------------
    for (int kt = 0; kt < nKTiles; ++kt) {
        const int kBase = kt * ATTN_G16_BK;

        // Stage K-tile [BK][headDim] into shared, ONCE for all warps.
        for (int i = threadIdx.x; i < ATTN_G16_BK * headDim; i += blockDim.x) {
            const int n = i / headDim, d = i % headDim;
            const int kk = kBase + n;
            sm.kvS[i] = (kk < kMaxTile)
                ? k[(std::size_t)kk * kvStride + (std::size_t)hkv * headDim + d]
                : __float2half(0.0f);
        }
        __syncthreads();

        // ---- QK^T: sSW[BQ][BK] = Q[BQ,D] . K[BK,D]^T (fp16 MMA) -----------
        if (warp < nQPerKv) {
            wmma::fragment<wmma::accumulator, 16, 16, G16_MMA_K, float> acc;
            wmma::fill_fragment(acc, 0.0f);
            for (int dt = 0; dt < nDTiles; ++dt) {
                const int dBase = dt * ATTN_G16_BK;
                wmma::fragment<wmma::matrix_b, 16, 16, G16_MMA_K, __half,
                               wmma::col_major> bFrag;   // K^T: col-major over [n,d]
                wmma::load_matrix_sync(bFrag, sm.kvS + dBase, headDim);
                wmma::mma_sync(acc, qFrag[dt], bFrag, acc);
            }
            wmma::store_matrix_sync(sSW, acc, ATTN_G16_BK, wmma::mem_row_major);
        }
        __syncwarp();

        // ---- per-row causal mask + scale + online softmax ----------------
        // One lane per query row within the warp.
        if (warp < nQPerKv && lane < ATTN_G16_BQ) {
            const int m  = lane;
            const int pq = q0 + m;
            const int kMax_m = positionOffset + pq + 1;   // exclusive
            float tileMax = -INFINITY;
            for (int n = 0; n < ATTN_G16_BK; ++n) {
                const int kk = kBase + n;
                float s = (kk < kMax_m) ? sSW[m * ATTN_G16_BK + n] * scale
                                        : -INFINITY;
                sSW[m * ATTN_G16_BK + n] = s;
                if (s > tileMax) tileMax = s;
            }
            const float mPrev = mShW[m];
            const float mNew  = (mPrev > tileMax) ? mPrev : tileMax;
            const float alpha = expf(mPrev - mNew);
            float tileSum = 0.0f;
            for (int n = 0; n < ATTN_G16_BK; ++n) {
                const float sv = sSW[m * ATTN_G16_BK + n];
                const float e  = (sv > -INFINITY) ? expf(sv - mNew) : 0.0f;
                pSW[m * ATTN_G16_BK + n] = __float2half(e);
                tileSum += e;
            }
            lShW[m] = alpha * lShW[m] + tileSum;
            mShW[m] = mNew;
            aShW[m] = alpha;
        }
        __syncthreads();

        // Re-stage the same key tile as V [BK][headDim] into shared kvS.
        for (int i = threadIdx.x; i < ATTN_G16_BK * headDim; i += blockDim.x) {
            const int n = i / headDim, d = i % headDim;
            const int kk = kBase + n;
            sm.kvS[i] = (kk < kMaxTile)
                ? v[(std::size_t)kk * kvStride + (std::size_t)hkv * headDim + d]
                : __float2half(0.0f);
        }
        __syncthreads();

        // ---- P.V: oRun[BQ][D] = alpha*oRun + P[BQ,BK] . V[BK,D] -----------
        if (warp < nQPerKv) {
            wmma::fragment<wmma::matrix_a, 16, 16, G16_MMA_K, __half,
                           wmma::row_major> pFrag;
            wmma::load_matrix_sync(pFrag, pSW, ATTN_G16_BK);   // P[BQ][BK]
            for (int dt = 0; dt < nDTiles; ++dt) {
                const int dBase = dt * ATTN_G16_BK;
                wmma::fragment<wmma::accumulator, 16, 16, G16_MMA_K, float> acc;
                wmma::fill_fragment(acc, 0.0f);
                wmma::fragment<wmma::matrix_b, 16, 16, G16_MMA_K, __half,
                               wmma::row_major> vFrag;         // V[BK][16d]
                wmma::load_matrix_sync(vFrag, sm.kvS + dBase, headDim);
                wmma::mma_sync(acc, pFrag, vFrag, acc);
                wmma::store_matrix_sync(oTW, acc, ATTN_G16_BK, wmma::mem_row_major);
                __syncwarp();
                for (int i = lane; i < ATTN_G16_BQ * ATTN_G16_BK; i += 32) {
                    const int m = i / ATTN_G16_BK, d = i % ATTN_G16_BK;
                    oRunW[m * headDim + (dBase + d)] =
                        aShW[m] * oRunW[m * headDim + (dBase + d)]
                        + oTW[m * ATTN_G16_BK + d];
                }
                __syncwarp();
            }
        }
        __syncthreads();
    }

    // ---- Normalise + write -----------------------------------------------
    if (warp < nQPerKv) {
        for (int i = lane; i < ATTN_G16_BQ * headDim; i += 32) {
            const int m = i / headDim, d = i % headDim;
            const int pq = q0 + m;
            if (pq >= T_q) continue;
            const float invL = (lShW[m] > 0.0f) ? (1.0f / lShW[m]) : 0.0f;
            out[(std::size_t)pq * qStride + (std::size_t)hq * headDim + d] =
                oRunW[m * headDim + d] * invL;
        }
    }
}