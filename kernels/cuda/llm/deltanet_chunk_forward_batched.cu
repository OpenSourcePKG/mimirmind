// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Batched chunked GatedDeltaNet prefill stage K2 — 5.21.9 serving variant.
// Runs the chunk forward (readout + state carry) for nSeq sequences, each
// with its own q/k/v/gCum/beta/a0 slabs and carried state.
//
// v2 (5.21.9) restructure vs the original uniform-T kernel — motivated by
// the serving ragged forward (mixed decode+prefill slots, nSeq up to 64):
//
//  * WORKER-POOL grid: gridDim.x fixed worker blocks iterate over the
//    (seq, head) items (item = w, w+G, ...). Global scratch is per WORKER
//    (5 chunk tensors [C,S]), so its size is G*5*C*S floats — independent
//    of nSeq — instead of nSeq*H tensors (which reaches hundreds of MiB at
//    maxBatch).
//  * The per-chunk state snapshot s0 [S,S] lives in dynamic SHARED memory
//    (needs the >48 KiB opt-in; measured faster than a global-scratch s0
//    despite the 1-block/SM cap — see the perf note in the kernel body).
//  * kq[a,m] dots are computed thread-parallel over (a,m) pairs (serial
//    S-dot per thread) instead of C^2/2 whole-block tree reductions — this
//    changes the reduction order, so results are tolerance-equal (not
//    bit-identical) to the single-sequence kernel; the math is unchanged.
//  * COMPACT a0 layout (matches deltanet_kkt_solve_batched): sequence seq's
//    chunk c sits at block chunkBase(seq) + c with chunkBase = sum over
//    s<seq of ceil(seqT[s]/C), computed in-kernel from seqT.
//
// Ragged layout (seqT/seqOff set): activations token-major [nRow, H, S]
// (gates [nRow, H]); sequence seq's tokens start at token seqOff[seq] and
// run for seqT[seq] tokens. T (= maxSeqT) only sizes the uniform fallback.
// seqT/seqOff/activeMask all nullptr => uniform T at stride seq*T,
// chunkBase = seq*maxChunks. Frozen slots (activeMask 0) are skipped with
// state untouched.
//
// Launch: grid = dim3(G, 1, 1) workers, block = S threads, dynamic smem =
// S*S*4 bytes. Scratch = G * 5 * C * S floats, caller-owned.

#include <cuda_runtime.h>

#ifndef DELTANET_CHUNK_FWD_MAX_S
#define DELTANET_CHUNK_FWD_MAX_S 128
#endif
#ifndef DELTANET_CHUNK_FWD_MAX_C
#define DELTANET_CHUNK_FWD_MAX_C 64
#endif

extern "C" __global__ __launch_bounds__(DELTANET_CHUNK_FWD_MAX_S)
void deltanet_chunk_forward_batched(
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
    // S rides blockDim.x (the launch always uses S threads), freeing the
    // argument slot for nSeq — kMaxArgs is exhausted at 16. A first version
    // carried nSeq in gridDim.z instead; that DUPLICATED the whole item walk
    // per z-slice (every block loops item = blockIdx.x), so two blocks raced
    // on the same state column — grid must be [G, 1, 1].
    const int S = blockDim.x;
    const int j = threadIdx.x;

    // Perf note (2026-09-04 A/B at conc16 serving prefill, vs the AR
    // recurrence's 14.5s TTFT): s0-in-smem = 22.1s, s0-in-global-scratch =
    // 23.3s. NEITHER scalar form beats AR — the chunk math costs ~2-3x the
    // FLOPs in the same 1-thread-per-column pattern, and the smem variant is
    // additionally capped at 1 block/SM by the 64 KiB opt-in. The smem form
    // is kept (it measured faster); a real win needs the matmul phases on
    // tensor cores (BF16 MMA tiles) — a separate kernel track.
    extern __shared__ float s0[];                     // [S, S] state snapshot
    __shared__ float egc[DELTANET_CHUNK_FWD_MAX_C];   // exp(gCum) (<=1, stable)
    __shared__ float gc[DELTANET_CHUNK_FWD_MAX_C];    // raw gCum (<=0)
    __shared__ float kqs[DELTANET_CHUNK_FWD_MAX_C * DELTANET_CHUNK_FWD_MAX_C];
    __shared__ int   itemInfo[4];                     // Tseq, tokBase, chunkBase, active

    const int maxChunks = (T + C - 1) / C;
    const size_t stateStride = (size_t)H * S * S;

    // Per-WORKER scratch: 5 chunk tensors [C, S] each.
    float* wbase = scratch + (size_t)blockIdx.x * 5 * C * S;
    float* uh  = wbase;
    float* uqh = wbase + (size_t)C * S;
    float* qsh = wbase + 2 * (size_t)C * S;
    float* rph = wbase + 3 * (size_t)C * S;
    float* dh  = wbase + 4 * (size_t)C * S;

    const float qScale = rsqrtf(static_cast<float>(S));
    const int totalItems = nSeq * H;

    for (int item = blockIdx.x; item < totalItems; item += gridDim.x) {
        const int seq = item / H;
        const int h   = item % H;

        // One thread derives the per-item geometry (avoids nSeq-wide loops in
        // every thread); the block syncs on the shared copy.
        if (j == 0) {
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
        const int Tseq      = itemInfo[0];
        const size_t tokBase = (size_t)itemInfo[1];
        const size_t chunkBase = (size_t)itemInfo[2];
        const int active    = itemInfo[3];
        __syncthreads();
        if (!active) continue;

        const float* __restrict__ q_    = q    + tokBase * (size_t)H * S;
        const float* __restrict__ k_    = k    + tokBase * (size_t)H * S;
        const float* __restrict__ v_    = v    + tokBase * (size_t)H * S;
        const float* __restrict__ gCum_ = gCum + tokBase * H;
        const float* __restrict__ beta_ = beta + tokBase * H;
        float*       __restrict__ out_  = out  + tokBase * (size_t)H * S;
        float* st = state + (size_t)seq * stateStride + (size_t)h * S * S;

        for (int c0 = 0; c0 < Tseq; c0 += C) {
            const int cs  = (C < Tseq - c0) ? C : (Tseq - c0);
            const int cIx = c0 / C;
            const float* a0c = a0 + (chunkBase + cIx) * H * C * C
                                  + (size_t)h * C * C;

            for (int i = 0; i < S; ++i) s0[i * S + j] = st[(size_t)i * S + j];
            if (j < cs) {
                const float g = gCum_[(size_t)(c0 + j) * H + h];
                gc[j]  = g;
                egc[j] = __expf(g);
            }
            __syncthreads();

            // Step 2: qs_a = qScale*q_a; u_a = S0^T k_a; uq_a = S0^T qs_a.
            for (int a = 0; a < cs; ++a) {
                const float* qa = q_ + ((size_t)(c0 + a) * H + h) * S;
                const float* ka = k_ + ((size_t)(c0 + a) * H + h) * S;
                qsh[a * S + j] = qa[j] * qScale;
                float uj = 0.0f, uqj = 0.0f;
                for (int i = 0; i < S; ++i) {
                    const float sij = s0[i * S + j];
                    uj  += sij * ka[i];
                    uqj += sij * (qa[i] * qScale);
                }
                uh[a * S + j]  = uj;
                uqh[a * S + j] = uqj;
            }
            __syncthreads();

            // Step 2.5: kq[a,m] = k_m . qs_a for m<=a — thread-parallel over
            // the (a,m) pairs, serial S-dot per thread (v2: replaces C^2/2
            // whole-block tree reductions; reduction order differs from the
            // single-seq kernel => tolerance-equal, not bit-identical).
            for (int p = j; p < cs * cs; p += S) {
                const int a = p / cs;
                const int m = p % cs;
                if (m <= a) {
                    const float* km = k_ + ((size_t)(c0 + m) * H + h) * S;
                    float kk = 0.0f;
                    for (int i = 0; i < S; ++i) kk += km[i] * qsh[a * S + i];
                    kqs[a * C + m] = kk;
                }
            }

            // Step 3: r'_m = beta_m (v_m - exp(G_m) u_m).
            for (int m = 0; m < cs; ++m) {
                const float* vm = v_ + ((size_t)(c0 + m) * H + h) * S;
                const float  bm = beta_[(size_t)(c0 + m) * H + h];
                rph[m * S + j] = bm * (vm[j] - egc[m] * uh[m * S + j]);
            }
            __syncthreads();

            // Step 4: d_a = sum_{m<=a} a0[a,m] exp(G_a - G_m) r'_m.
            for (int a = 0; a < cs; ++a) {
                float dj = 0.0f;
                for (int m = 0; m <= a; ++m)
                    dj += a0c[a * C + m] * __expf(gc[a] - gc[m]) * rph[m * S + j];
                dh[a * S + j] = dj;
            }
            __syncthreads();

            // Step 5: o_a = exp(G_a) uq_a + sum_{m<=a} exp(G_a-G_m)(k_m.qs_a) d_m.
            for (int a = 0; a < cs; ++a) {
                float* oa = out_ + ((size_t)(c0 + a) * H + h) * S;
                float oj = egc[a] * uqh[a * S + j];
                for (int m = 0; m <= a; ++m) {
                    const float w = __expf(gc[a] - gc[m]) * kqs[a * C + m];
                    oj += w * dh[m * S + j];
                }
                oa[j] = oj;
            }
            __syncthreads();

            // Step 6: state carry S' = exp(G_last) S0
            //         + sum_m exp(G_last - G_m) k_m d_m^T.
            const float eLast = egc[cs - 1];
            for (int i = 0; i < S; ++i) {
                float sij = eLast * s0[i * S + j];
                for (int m = 0; m < cs; ++m) {
                    const float* km = k_ + ((size_t)(c0 + m) * H + h) * S;
                    sij += (__expf(gc[cs - 1] - gc[m]) * km[i]) * dh[m * S + j];
                }
                st[(size_t)i * S + j] = sij;
            }
            __syncthreads();
        }
        __syncthreads();
    }
}
