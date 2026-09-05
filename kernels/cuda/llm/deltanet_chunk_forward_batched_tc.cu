// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Tensor-core chunked GatedDeltaNet prefill stage K2 — 5.21.9 TC variant.
//
// Same contract as deltanet_chunk_forward_batched (worker-pool grid [G,1,1],
// nSeq as an argument, ragged seqT/seqOff/activeMask, COMPACT a0 layout,
// per-worker scratch), but the O(C*S) / O(C*C) inner phases run as BF16
// wmma GEMMs with FP32 accumulation instead of 1-thread-per-state-column
// scalar loops.
//
// v2 (staging redesign): v1 converted every A/B tile lane-cooperatively into
// a warp-private smem staging buffer per mma — that staging dominated the
// runtime (section 16.7s vs the scalar chunk's 19.5s but still well behind
// AR's 11.4s). v2 converts each chunk's operands to BF16 ONCE per chunk into
// per-worker GLOBAL mirrors (kB, qB=Qs, ktB, s0b) and feeds
// wmma::load_matrix_sync straight from global (row_major, or col_major where
// the math needs the transpose). Only the two [C,C] coefficient matrices
// (M from a0*exp-decay, W from kq*exp-decay) are still built tile-wise in a
// warp-private staging buffer.
//
// Math (identical formulas to the scalar kernel, evaluated as GEMMs):
//   step 2   U  = K  @ S0,  UQ = Qs @ S0          (C x S) = (C x S)(S x S)
//   step 2.5 KQ = Qs @ K^T                         (C x C)
//   step 3   RP[m,:] = b_m (V[m,:] - egc_m U[m,:])           (elementwise)
//   step 4   D  = M @ RP,  M[a,m] = a0[a,m] e^{g_a-g_m} (m<=a)
//   step 5   O[a,:] = egc_a UQ[a,:] + (W @ D)[a,:],
//            W[a,m] = e^{g_a-g_m} KQ[a,m] (m<=a)
//   step 6   S' = eLast*S0 + Kt^T @ D,  Kt[m,:] = e^{g_last-g_m} K[m,:]
//
// Numerics: the persistent state S stays FP32-resident in shared memory
// across the whole item; GEMM operands are rounded to BF16 at the per-chunk
// mirror conversion and accumulate in FP32. Tolerance-equal to the scalar
// chunk pipeline (not bit-identical); serving quality bar = needle+coherence.
//
// Shape restriction: S == 128 and C == 64 (the prod GDN shape). The host
// dispatcher falls back to the scalar kernel otherwise.
//
// Per-worker scratch = 6*C*S floats (192 KiB) as a byte pool:
//   off   0Ki  uh   fp32 [C,S] (32 KiB)  step-2 out, step-3 in
//   off  32Ki  uqh  fp32 [C,S] (32 KiB)  step-2 out, step-5 epilogue in
//   off  64Ki  rphB bf16 [C,S] (16 KiB)  step-3 out, step-4 B
//   off  80Ki  dB   bf16 [C,S] (16 KiB)  step-4 out, step-5/6 B
//   off  96Ki  kq   fp32 [C,C] (16 KiB)  step-2.5 out, step-5 W source
//   off 112Ki  kB   bf16 [C,S] (16 KiB)  K chunk mirror
//   off 128Ki  qB   bf16 [C,S] (16 KiB)  Qs chunk mirror (q * 1/sqrt(S))
//   off 144Ki  ktB  bf16 [C,S] (16 KiB)  decay-scaled K mirror (step 6 A)
//   off 160Ki  s0b  bf16 [S,S] (32 KiB)  per-chunk BF16 state mirror
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
    __shared__ __nv_bfloat16 aStg[DCFT_WARPS][TL * TL]; // M/W tile staging
    __shared__ float         cStg[DCFT_WARPS][TL * TL]; // accum store tiles

    const int maxChunks = (T + C - 1) / C;
    const size_t stateStride = (size_t)H * S * S;

    char* wbase = reinterpret_cast<char*>(scratch)
                + (size_t)blockIdx.x * 6 * C * S * sizeof(float);
    float*         uh   = reinterpret_cast<float*>(wbase);
    float*         uqh  = reinterpret_cast<float*>(wbase + 32 * 1024);
    __nv_bfloat16* rphB = reinterpret_cast<__nv_bfloat16*>(wbase + 64 * 1024);
    __nv_bfloat16* dB   = reinterpret_cast<__nv_bfloat16*>(wbase + 80 * 1024);
    float*         kqS  = reinterpret_cast<float*>(wbase + 96 * 1024);
    __nv_bfloat16* kB   = reinterpret_cast<__nv_bfloat16*>(wbase + 112 * 1024);
    __nv_bfloat16* qB   = reinterpret_cast<__nv_bfloat16*>(wbase + 128 * 1024);
    __nv_bfloat16* ktB  = reinterpret_cast<__nv_bfloat16*>(wbase + 144 * 1024);
    __nv_bfloat16* s0b  = reinterpret_cast<__nv_bfloat16*>(wbase + 160 * 1024);

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

            // ---- per-chunk BF16 operand mirrors (ONE parallel pass) ------
            // kB / qB / ktB from global fp32 (zero-padded beyond cs);
            // s0b from the fp32 smem state.
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
            for (int i = tid; i < S * S; i += 256) {
                s0b[i] = __float2bfloat16(s0[i]);
            }
            __syncthreads();

            // ---- step 2: U = K @ S0, UQ = Qs @ S0 ------------------------
            {
                const int mT = C / TL, nT = S / TL;        // 4 x 8 per GEMM
                for (int t = warp; t < 2 * mT * nT; t += DCFT_WARPS) {
                    const bool isQ = t >= mT * nT;
                    const int  tt  = isQ ? t - mT * nT : t;
                    const int  tm  = tt / nT;
                    const int  tn  = tt % nT;
                    const __nv_bfloat16* A = isQ ? qB : kB;
                    float* dst = isQ ? uqh : uh;
                    FragC acc;
                    wmma::fill_fragment(acc, 0.0f);
                    for (int tk = 0; tk < S / TL; ++tk) {
                        FragA aF;
                        FragB bF;
                        wmma::load_matrix_sync(
                            aF, A + (size_t)(tm * TL) * S + tk * TL, S);
                        wmma::load_matrix_sync(
                            bF, s0b + (size_t)(tk * TL) * S + tn * TL, S);
                        wmma::mma_sync(acc, aF, bF, acc);
                    }
                    wmma::store_matrix_sync(
                        dst + (size_t)(tm * TL) * S + tn * TL, acc, S,
                        wmma::mem_row_major);
                }
            }
            __syncthreads();

            // ---- step 2.5: KQ = Qs @ K^T (B = kB col_major) --------------
            {
                const int mT = C / TL, nT = C / TL;        // 4 x 4
                for (int t = warp; t < mT * nT; t += DCFT_WARPS) {
                    const int tm = t / nT;
                    const int tn = t % nT;
                    FragC acc;
                    wmma::fill_fragment(acc, 0.0f);
                    for (int tk = 0; tk < S / TL; ++tk) {
                        FragA  aF;
                        FragBc bF;
                        wmma::load_matrix_sync(
                            aF, qB + (size_t)(tm * TL) * S + tk * TL, S);
                        wmma::load_matrix_sync(
                            bF, kB + (size_t)(tn * TL) * S + tk * TL, S);
                        wmma::mma_sync(acc, aF, bF, acc);
                    }
                    wmma::store_matrix_sync(
                        kqS + (size_t)(tm * TL) * C + tn * TL, acc, C,
                        wmma::mem_row_major);
                }
            }
            __syncthreads();

            // ---- step 3: RP = diag(b) (V - diag(egc) U) — bf16 out -------
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
            __syncthreads();

            // ---- step 4: D = M @ RP (M staged from a0 * exp decay) -------
            {
                const int mT = C / TL, nT = S / TL;
                for (int t = warp; t < mT * nT; t += DCFT_WARPS) {
                    const int tm = t / nT;
                    const int tn = t % nT;
                    FragC acc;
                    wmma::fill_fragment(acc, 0.0f);
                    for (int tk = 0; tk < C / TL; ++tk) {
                        for (int i = lane; i < TL * TL; i += 32) {
                            const int a = tm * TL + i / TL;
                            const int m = tk * TL + i % TL;
                            float x = 0.0f;
                            if (a < cs && m <= a) {
                                x = a0c[(size_t)a * C + m]
                                  * __expf(gc[a] - gc[m]);
                            }
                            aStg[warp][i] = __float2bfloat16(x);
                        }
                        __syncwarp();
                        FragA aF;
                        FragB bF;
                        wmma::load_matrix_sync(aF, aStg[warp], TL);
                        wmma::load_matrix_sync(
                            bF, rphB + (size_t)(tk * TL) * S + tn * TL, S);
                        wmma::mma_sync(acc, aF, bF, acc);
                        __syncwarp();
                    }
                    wmma::store_matrix_sync(cStg[warp], acc, TL,
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

            // ---- step 5: O = diag(egc) UQ + W @ D ------------------------
            {
                const int mT = C / TL, nT = S / TL;
                for (int t = warp; t < mT * nT; t += DCFT_WARPS) {
                    const int tm = t / nT;
                    const int tn = t % nT;
                    FragC acc;
                    wmma::fill_fragment(acc, 0.0f);
                    for (int tk = 0; tk < C / TL; ++tk) {
                        for (int i = lane; i < TL * TL; i += 32) {
                            const int a = tm * TL + i / TL;
                            const int m = tk * TL + i % TL;
                            float x = 0.0f;
                            if (a < cs && m <= a) {
                                x = kqS[(size_t)a * C + m]
                                  * __expf(gc[a] - gc[m]);
                            }
                            aStg[warp][i] = __float2bfloat16(x);
                        }
                        __syncwarp();
                        FragA aF;
                        FragB bF;
                        wmma::load_matrix_sync(aF, aStg[warp], TL);
                        wmma::load_matrix_sync(
                            bF, dB + (size_t)(tk * TL) * S + tn * TL, S);
                        wmma::mma_sync(acc, aF, bF, acc);
                        __syncwarp();
                    }
                    wmma::store_matrix_sync(cStg[warp], acc, TL,
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

            // ---- step 6: S' = eLast*S0 + Kt^T @ D (in-place s0 tiles) ----
            {
                const float eLast = egc[cs - 1];
                const int mT = S / TL, nT = S / TL;        // 8 x 8
                for (int t = warp; t < mT * nT; t += DCFT_WARPS) {
                    const int tm = t / nT;
                    const int tn = t % nT;
                    for (int i = lane; i < TL * TL; i += 32) {
                        cStg[warp][i] = eLast
                            * s0[(size_t)(tm * TL + i / TL) * S
                                 + tn * TL + i % TL];
                    }
                    __syncwarp();
                    FragC acc;
                    wmma::load_matrix_sync(acc, cStg[warp], TL,
                                           wmma::mem_row_major);
                    for (int tk = 0; tk < C / TL; ++tk) {
                        FragAc aF;
                        FragB  bF;
                        wmma::load_matrix_sync(
                            aF, ktB + (size_t)(tk * TL) * S + tm * TL, S);
                        wmma::load_matrix_sync(
                            bF, dB + (size_t)(tk * TL) * S + tn * TL, S);
                        wmma::mma_sync(acc, aF, bF, acc);
                    }
                    wmma::store_matrix_sync(cStg[warp], acc, TL,
                                            wmma::mem_row_major);
                    __syncwarp();
                    for (int i = lane; i < TL * TL; i += 32) {
                        s0[(size_t)(tm * TL + i / TL) * S + tn * TL + i % TL]
                            = cStg[warp][i];
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
