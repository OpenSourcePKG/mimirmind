// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Tensor-core chunked GatedDeltaNet prefill stage K2 — 5.21.9 TC variant.
//
// Same contract as deltanet_chunk_forward_batched (worker-pool grid [G,1,1],
// nSeq as an argument, ragged seqT/seqOff/activeMask, COMPACT a0 layout,
// per-worker scratch), with the O(C*S)/O(C*C) phases as BF16 wmma GEMMs
// (FP32 accumulation) instead of 1-thread-per-state-column scalar loops.
//
// v3 (register blocking + materialised coefficient mirrors):
//  * v2 already mirrored the chunk operands to per-worker global BF16 once
//    per chunk (kB, qB=Qs, ktB, s0b) — that removed v1's dominating per-mma
//    lane staging (section 16.7s -> 8.79s, first net win over AR).
//  * v3 assigns each warp an OUTPUT COLUMN stripe and keeps 2-8 accumulator
//    fragments live, so every B fragment is loaded ONCE per k-tile and
//    reused across the stripe's row tiles (4-8x fewer B loads — the load
//    latency was the remaining bottleneck), and materialises the two [C,C]
//    decay-coefficient matrices (M from a0, W from KQ) as BF16 mirrors so
//    no tile staging remains at all.
//
// Math (identical formulas to the scalar kernel, evaluated as GEMMs):
//   step 2   U  = K  @ S0,  UQ = Qs @ S0
//   step 2.5 KQ = Qs @ K^T
//   step 3   RP[m,:] = b_m (V[m,:] - egc_m U[m,:])          (elementwise)
//   step 4   D  = M @ RP,  M[a,m] = a0[a,m] e^{g_a-g_m} (m<=a)
//   step 5   O[a,:] = egc_a UQ[a,:] + (W @ D)[a,:],
//            W[a,m] = e^{g_a-g_m} KQ[a,m] (m<=a)
//   step 6   S' = eLast*S0 + Kt^T @ D,  Kt[m,:] = e^{g_last-g_m} K[m,:]
//
// Numerics: persistent state FP32-resident in smem across the item; GEMM
// operands rounded to BF16 at the per-chunk mirror conversion, FP32
// accumulate. Tolerance-equal to the scalar chunk pipeline; serving quality
// bar = needle + coherence (both green on the v2 math, unchanged here).
//
// Shape restriction: S == 128, C == 64 (prod GDN shape); host falls back to
// the scalar kernel otherwise.
//
// Per-worker scratch = 7*C*S floats (224 KiB) as a byte pool:
//   off   0Ki  uh    fp32 [C,S] (32 KiB)
//   off  32Ki  uqh   fp32 [C,S] (32 KiB)
//   off  64Ki  rphB  bf16 [C,S] (16 KiB)
//   off  80Ki  dB    bf16 [C,S] (16 KiB)
//   off  96Ki  kqS   fp32 [C,C] (16 KiB)
//   off 112Ki  kB    bf16 [C,S] (16 KiB)
//   off 128Ki  qB    bf16 [C,S] (16 KiB)   (Qs = q * 1/sqrt(S))
//   off 144Ki  ktB   bf16 [C,S] (16 KiB)
//   off 160Ki  s0b   bf16 [S,S] (32 KiB)
//   off 192Ki  mB    bf16 [C,C] ( 8 KiB)
//   off 200Ki  wB    bf16 [C,C] ( 8 KiB)
//
// Launch: grid = [G, 1, 1], block = 256, dynamic smem = S*S*4 (fp32 state).

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>

using namespace nvcuda;

#define DCFT_S 128
#define DCFT_C 64
#define DCFT_T16 16
#define DCFT_WARPS 8

extern "C" __global__ __launch_bounds__(256)
void deltanet_chunk_forward_batched_tc(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ gCum,
    const float* __restrict__ beta,
    const float* __restrict__ a0,
    float*       __restrict__ state,
    float*       __restrict__ out,
    float*       __restrict__ scratch,
    const int T, const int H, const int nSeq, const int C,
    const unsigned char* __restrict__ activeMask,
    const int* __restrict__ seqT,
    const int* __restrict__ seqOff)
{
    constexpr int S  = DCFT_S;
    constexpr int TL = DCFT_T16;
    if (C != DCFT_C) return;               // host guards; belt-and-braces

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid / 32;
    const int lane = tid % 32;

    extern __shared__ float s0[];                       // [S, S] fp32 state
    __shared__ float gc[DCFT_C];
    __shared__ float egc[DCFT_C];
    __shared__ int   itemInfo[4];
    __shared__ float cStg[DCFT_WARPS][TL * TL];         // accum store tiles

    const int maxChunks = (T + C - 1) / C;
    const size_t stateStride = (size_t)H * S * S;

    char* wbase = reinterpret_cast<char*>(scratch)
                + (size_t)blockIdx.x * 7 * C * S * sizeof(float);
    float*         uh   = reinterpret_cast<float*>(wbase);
    float*         uqh  = reinterpret_cast<float*>(wbase + 32 * 1024);
    __nv_bfloat16* rphB = reinterpret_cast<__nv_bfloat16*>(wbase + 64 * 1024);
    __nv_bfloat16* dB   = reinterpret_cast<__nv_bfloat16*>(wbase + 80 * 1024);
    float*         kqS  = reinterpret_cast<float*>(wbase + 96 * 1024);
    __nv_bfloat16* kB   = reinterpret_cast<__nv_bfloat16*>(wbase + 112 * 1024);
    __nv_bfloat16* qB   = reinterpret_cast<__nv_bfloat16*>(wbase + 128 * 1024);
    __nv_bfloat16* ktB  = reinterpret_cast<__nv_bfloat16*>(wbase + 144 * 1024);
    __nv_bfloat16* s0b  = reinterpret_cast<__nv_bfloat16*>(wbase + 160 * 1024);
    __nv_bfloat16* mB   = reinterpret_cast<__nv_bfloat16*>(wbase + 192 * 1024);
    __nv_bfloat16* wB   = reinterpret_cast<__nv_bfloat16*>(wbase + 200 * 1024);

    const float qScale = rsqrtf(static_cast<float>(S));
    const int totalItems = nSeq * H;

    using FragA  = wmma::fragment<wmma::matrix_a, TL, TL, TL,
                                  __nv_bfloat16, wmma::row_major>;
    using FragAc = wmma::fragment<wmma::matrix_a, TL, TL, TL,
                                  __nv_bfloat16, wmma::col_major>;
    using FragB  = wmma::fragment<wmma::matrix_b, TL, TL, TL,
                                  __nv_bfloat16, wmma::row_major>;
    using FragBc = wmma::fragment<wmma::matrix_b, TL, TL, TL,
                                  __nv_bfloat16, wmma::col_major>;
    using FragC  = wmma::fragment<wmma::accumulator, TL, TL, TL, float>;

    for (int item = blockIdx.x; item < totalItems; item += gridDim.x) {
        const int seq = item / H;
        const int h   = item % H;

        if (tid == 0) {
            int active = (activeMask == nullptr || activeMask[seq] != 0);
            int Tseq   = (seqT != nullptr) ? seqT[seq] : T;
            long long tokBase = (seqOff != nullptr)
                                    ? (long long)seqOff[seq]
                                    : (long long)seq * T;
            long long chunkBase;
            if (seqT != nullptr) {
                chunkBase = 0;
                for (int s = 0; s < seq; ++s)
                    chunkBase += (seqT[s] + C - 1) / C;
            } else {
                chunkBase = (long long)seq * maxChunks;
            }
            itemInfo[0] = Tseq;
            itemInfo[1] = (int)tokBase;
            itemInfo[2] = (int)chunkBase;
            itemInfo[3] = active;
        }
        __syncthreads();
        const int    Tseq      = itemInfo[0];
        const size_t tokBase   = (size_t)itemInfo[1];
        const size_t chunkBase = (size_t)itemInfo[2];
        const int    active    = itemInfo[3];
        __syncthreads();
        if (!active) continue;

        const float* __restrict__ q_    = q    + tokBase * (size_t)H * S;
        const float* __restrict__ k_    = k    + tokBase * (size_t)H * S;
        const float* __restrict__ v_    = v    + tokBase * (size_t)H * S;
        const float* __restrict__ gCum_ = gCum + tokBase * H;
        const float* __restrict__ beta_ = beta + tokBase * H;
        float*       __restrict__ out_  = out  + tokBase * (size_t)H * S;
        float* st = state + (size_t)seq * stateStride + (size_t)h * S * S;

        for (int i = tid; i < S * S; i += 256) {
            s0[i] = st[i];
        }
        __syncthreads();

        // v4: single-token items (the DECODE slots of a mixed serving
        // forward) skip the whole chunk apparatus — mirrors, coefficient
        // GEMMs and a0 are pure overhead for one token. Run the plain
        // AR delta-rule step instead (same math as the v3 recurrence,
        // consuming the pre-gated gLog/sigmoided beta this pipeline gets).
        if (Tseq == 1) {
            const float g1 = __expf(gCum_[h]);
            const float b1 = beta_[h];
            float* ksh = cStg[0];       // [S] k row     (reuse accum tiles)
            float* qsh = cStg[2];       // [S] scaled q
            if (tid < S) {
                const size_t base = (size_t)h * S + tid;
                ksh[tid] = k_[base];
                qsh[tid] = q_[base] * qScale;
            }
            __syncthreads();
            if (tid < S) {
                const int j = tid;
                float sk = 0.0f;
                for (int i = 0; i < S; ++i) {
                    sk += (s0[(size_t)i * S + j] * g1) * ksh[i];
                }
                const float dj = (v_[(size_t)h * S + j] - sk) * b1;
                float oj = 0.0f;
                for (int i = 0; i < S; ++i) {
                    const float sij = s0[(size_t)i * S + j] * g1
                                    + ksh[i] * dj;
                    s0[(size_t)i * S + j] = sij;
                    oj += sij * qsh[i];
                }
                out_[(size_t)h * S + j] = oj;
            }
            __syncthreads();
            for (int i = tid; i < S * S; i += 256) {
                st[i] = s0[i];
            }
            __syncthreads();
            continue;
        }

        for (int c0 = 0; c0 < Tseq; c0 += C) {
            const int cs  = (C < Tseq - c0) ? C : (Tseq - c0);
            const int cIx = c0 / C;
            const float* a0c = a0 + (chunkBase + cIx) * H * C * C
                                  + (size_t)h * C * C;

            if (tid < C) {
                const float g = (tid < cs)
                    ? gCum_[(size_t)(c0 + tid) * H + h] : 0.0f;
                gc[tid]  = g;
                egc[tid] = __expf(g);
            }
            __syncthreads();
            const float gLast = gc[cs - 1];

            // ---- per-chunk BF16 mirrors (ONE parallel pass) --------------
            for (int i = tid; i < C * S; i += 256) {
                const int m = i / S, j = i % S;
                float kv = 0.0f, qv = 0.0f, ktv = 0.0f;
                if (m < cs) {
                    const size_t base = ((size_t)(c0 + m) * H + h) * S + j;
                    kv  = k_[base];
                    qv  = q_[base] * qScale;
                    ktv = kv * __expf(gLast - gc[m]);
                }
                kB[i]  = __float2bfloat16(kv);
                qB[i]  = __float2bfloat16(qv);
                ktB[i] = __float2bfloat16(ktv);
            }
            // s0b is refreshed by step 6's store loop from chunk to chunk;
            // only the FIRST chunk mirrors the freshly-loaded state here.
            if (c0 == 0) {
                for (int i = tid; i < S * S; i += 256) {
                    s0b[i] = __float2bfloat16(s0[i]);
                }
            }
            for (int i = tid; i < C * C; i += 256) {
                const int a = i / C, m = i % C;
                float x = 0.0f;
                if (a < cs && m <= a) {
                    x = a0c[(size_t)a * C + m] * __expf(gc[a] - gc[m]);
                }
                mB[i] = __float2bfloat16(x);
            }
            __syncthreads();

            // ---- step 2: U = K @ S0, UQ = Qs @ S0 (warp = column stripe) -
            {
                const int tn = warp;                   // 8 warps = 8 columns
                for (int g = 0; g < 2; ++g) {
                    const __nv_bfloat16* A = (g == 0) ? kB : qB;
                    float* dst = (g == 0) ? uh : uqh;
                    FragC acc[4];
                    for (int tm = 0; tm < 4; ++tm) {
                        wmma::fill_fragment(acc[tm], 0.0f);
                    }
                    for (int tk = 0; tk < S / TL; ++tk) {
                        FragB bF;
                        wmma::load_matrix_sync(
                            bF, s0b + (size_t)(tk * TL) * S + tn * TL, S);
                        for (int tm = 0; tm < 4; ++tm) {
                            FragA aF;
                            wmma::load_matrix_sync(
                                aF, A + (size_t)(tm * TL) * S + tk * TL, S);
                            wmma::mma_sync(acc[tm], aF, bF, acc[tm]);
                        }
                    }
                    for (int tm = 0; tm < 4; ++tm) {
                        wmma::store_matrix_sync(
                            dst + (size_t)(tm * TL) * S + tn * TL, acc[tm],
                            S, wmma::mem_row_major);
                    }
                }
            }
            __syncthreads();

            // ---- step 2.5: KQ = Qs @ K^T (4x4 tiles, warp = (half, col)) -
            {
                const int tn = warp % 4;
                const int th = warp / 4;               // tm half: 0 -> 0..1
                FragC acc[2];
                wmma::fill_fragment(acc[0], 0.0f);
                wmma::fill_fragment(acc[1], 0.0f);
                for (int tk = 0; tk < S / TL; ++tk) {
                    FragBc bF;
                    wmma::load_matrix_sync(
                        bF, kB + (size_t)(tn * TL) * S + tk * TL, S);
                    for (int r = 0; r < 2; ++r) {
                        const int tm = th * 2 + r;
                        FragA aF;
                        wmma::load_matrix_sync(
                            aF, qB + (size_t)(tm * TL) * S + tk * TL, S);
                        wmma::mma_sync(acc[r], aF, bF, acc[r]);
                    }
                }
                for (int r = 0; r < 2; ++r) {
                    const int tm = th * 2 + r;
                    wmma::store_matrix_sync(
                        kqS + (size_t)(tm * TL) * C + tn * TL, acc[r], C,
                        wmma::mem_row_major);
                }
            }
            __syncthreads();

            // ---- step 3: RP mirror + W mirror (one parallel pass) --------
            for (int i = tid; i < C * S; i += 256) {
                const int m = i / S, j = i % S;
                float r = 0.0f;
                if (m < cs) {
                    const float bm = beta_[(size_t)(c0 + m) * H + h];
                    r = bm * (v_[((size_t)(c0 + m) * H + h) * S + j]
                              - egc[m] * uh[(size_t)m * S + j]);
                }
                rphB[i] = __float2bfloat16(r);
            }
            for (int i = tid; i < C * C; i += 256) {
                const int a = i / C, m = i % C;
                float x = 0.0f;
                if (a < cs && m <= a) {
                    x = kqS[(size_t)a * C + m] * __expf(gc[a] - gc[m]);
                }
                wB[i] = __float2bfloat16(x);
            }
            __syncthreads();

            // ---- step 4: D = M @ RP (warp = column stripe) ---------------
            {
                const int tn = warp;
                FragC acc[4];
                for (int tm = 0; tm < 4; ++tm) {
                    wmma::fill_fragment(acc[tm], 0.0f);
                }
                for (int tk = 0; tk < C / TL; ++tk) {
                    FragB bF;
                    wmma::load_matrix_sync(
                        bF, rphB + (size_t)(tk * TL) * S + tn * TL, S);
                    for (int tm = 0; tm < 4; ++tm) {
                        FragA aF;
                        wmma::load_matrix_sync(
                            aF, mB + (size_t)(tm * TL) * C + tk * TL, C);
                        wmma::mma_sync(acc[tm], aF, bF, acc[tm]);
                    }
                }
                for (int tm = 0; tm < 4; ++tm) {
                    wmma::store_matrix_sync(cStg[warp], acc[tm], TL,
                                            wmma::mem_row_major);
                    __syncwarp();
                    for (int i = lane; i < TL * TL; i += 32) {
                        dB[(size_t)(tm * TL + i / TL) * S + tn * TL + i % TL]
                            = __float2bfloat16(cStg[warp][i]);
                    }
                    __syncwarp();
                }
            }
            __syncthreads();

            // ---- step 5: O = diag(egc) UQ + W @ D (warp = column) --------
            {
                const int tn = warp;
                FragC acc[4];
                for (int tm = 0; tm < 4; ++tm) {
                    wmma::fill_fragment(acc[tm], 0.0f);
                }
                for (int tk = 0; tk < C / TL; ++tk) {
                    FragB bF;
                    wmma::load_matrix_sync(
                        bF, dB + (size_t)(tk * TL) * S + tn * TL, S);
                    for (int tm = 0; tm < 4; ++tm) {
                        FragA aF;
                        wmma::load_matrix_sync(
                            aF, wB + (size_t)(tm * TL) * C + tk * TL, C);
                        wmma::mma_sync(acc[tm], aF, bF, acc[tm]);
                    }
                }
                for (int tm = 0; tm < 4; ++tm) {
                    wmma::store_matrix_sync(cStg[warp], acc[tm], TL,
                                            wmma::mem_row_major);
                    __syncwarp();
                    for (int i = lane; i < TL * TL; i += 32) {
                        const int a = tm * TL + i / TL;
                        if (a >= cs) continue;
                        const int j = tn * TL + i % TL;
                        out_[((size_t)(c0 + a) * H + h) * S + j] =
                            egc[a] * uqh[(size_t)a * S + j] + cStg[warp][i];
                    }
                    __syncwarp();
                }
            }
            __syncthreads();

            // ---- step 6: S' = eLast*S0 + Kt^T @ D (warp = column) --------
            {
                const float eLast = egc[cs - 1];
                const int tn = warp;
                FragC acc[8];
                for (int tm = 0; tm < 8; ++tm) {
                    for (int i = lane; i < TL * TL; i += 32) {
                        cStg[warp][i] = eLast
                            * s0[(size_t)(tm * TL + i / TL) * S
                                 + tn * TL + i % TL];
                    }
                    __syncwarp();
                    wmma::load_matrix_sync(acc[tm], cStg[warp], TL,
                                           wmma::mem_row_major);
                    __syncwarp();
                }
                for (int tk = 0; tk < C / TL; ++tk) {
                    FragB bF;
                    wmma::load_matrix_sync(
                        bF, dB + (size_t)(tk * TL) * S + tn * TL, S);
                    for (int tm = 0; tm < 8; ++tm) {
                        FragAc aF;
                        wmma::load_matrix_sync(
                            aF, ktB + (size_t)(tk * TL) * S + tm * TL, S);
                        wmma::mma_sync(acc[tm], aF, bF, acc[tm]);
                    }
                }
                for (int tm = 0; tm < 8; ++tm) {
                    wmma::store_matrix_sync(cStg[warp], acc[tm], TL,
                                            wmma::mem_row_major);
                    __syncwarp();
                    for (int i = lane; i < TL * TL; i += 32) {
                        const size_t idx =
                            (size_t)(tm * TL + i / TL) * S + tn * TL + i % TL;
                        const float sv = cStg[warp][i];
                        s0[idx]  = sv;
                        // Refresh the BF16 mirror in the same pass — the next
                        // chunk's step 2 reads it, saving the separate
                        // per-chunk S*S mirror conversion.
                        s0b[idx] = __float2bfloat16(sv);
                    }
                    __syncwarp();
                }
            }
            __syncthreads();
        }

        for (int i = tid; i < S * S; i += 256) {
            st[i] = s0[i];
        }
        __syncthreads();
    }
}
