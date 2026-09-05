// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Chunked GatedDeltaNet prefill — stage K1: per-chunk ungated triangular
// inverse A0. Direct port of compute::deltanetKktSolveInverse.
//
//   lt[a,m] = beta_a (k_a . k_m)   for m < a   (strict-lower gated-free Gram)
//   L = I + strictLower(lt);  A0 = L^-1 is unit lower-triangular:
//     A0[a,a] = 1;  A0[a,m>a] = 0;
//     A0[a,m] = -sum_{p=m..a-1} lt[a,p] A0[p,m]   (a > m)
//
//   k [T,H,S]; beta [T,H]; a0 [nChunks,H,C,C] row-major.
//
// One block per (chunk, head): grid = nChunks*H (bid = c*H + h). Block = C
// threads. lt is built into shared (thread = row a, dot over S), then the
// unit-lower inverse is solved with thread = column m: each column is an
// independent serial forward-substitution over rows a (A0[a,m] needs
// A0[p<a,m], same thread → no cross-thread hazard). Correctness-first; the
// per-row Gram dot and the serial substitution are the O(C^2) work.
//
// Assumes C <= MAX_C. Block dim = C (<= MAX_C = __launch_bounds__).

#include <cuda_runtime.h>

#ifndef DELTANET_KKT_MAX_C
#define DELTANET_KKT_MAX_C 64
#endif

extern "C" __global__ __launch_bounds__(DELTANET_KKT_MAX_C)
void deltanet_kkt_solve(
    const float* __restrict__ k,
    const float* __restrict__ beta,
    float*       __restrict__ a0,
    const int T, const int H, const int S, const int C)
{
    const int nChunks = (T + C - 1) / C;
    const int bid = blockIdx.x;                 // c*H + h
    if (bid >= nChunks * H) return;
    const int c = bid / H;
    const int h = bid % H;
    const int c0 = c * C;
    int cs = C;
    if (c0 + cs > T) cs = T - c0;

    const int t = threadIdx.x;                  // row a (phase 1) / col m (phase 2)
    float* a0c = a0 + (static_cast<size_t>(c) * H + h) * C * C;

    __shared__ float lt[DELTANET_KKT_MAX_C * DELTANET_KKT_MAX_C];

    // Zero this chunk's a0 block and lt.
    for (int idx = t; idx < C * C; idx += blockDim.x) {
        a0c[idx] = 0.0f;
        lt[idx]  = 0.0f;
    }
    __syncthreads();

    // Phase 1: strict-lower Gram, thread = row a. Non-FMA (__fmul_rn/__fadd_rn)
    // so the accumulation rounds bit-for-bit like the CPU reference — the
    // triangular inverse below amplifies tiny rounding differences into the
    // large A0 entries, and a fused-vs-separate mul-add there breaks the
    // absolute 2e-3 parity tolerance at S=128.
    if (t < cs) {
        const float* ka = k + (static_cast<size_t>(c0 + t) * H + h) * S;
        const float  ba = beta[(c0 + t) * H + h];
        for (int m = 0; m < t; ++m) {
            const float* km = k + (static_cast<size_t>(c0 + m) * H + h) * S;
            float kk = 0.0f;
            for (int i = 0; i < S; ++i) kk = __fadd_rn(kk, __fmul_rn(ka[i], km[i]));
            lt[t * C + m] = __fmul_rn(ba, kk);
        }
    }
    __syncthreads();

    // Phase 2: unit-lower inverse, thread = column m (independent per column).
    if (t < cs) {
        a0c[t * C + t] = 1.0f;
        for (int a = t + 1; a < cs; ++a) {
            float acc = 0.0f;
            for (int p = t; p < a; ++p)
                acc = __fadd_rn(acc, __fmul_rn(lt[a * C + p], a0c[p * C + t]));
            a0c[a * C + t] = -acc;
        }
    }
}

// 5.21.9 — batched/ragged K1 for the serving prefill path. grid =
// dim3(maxChunks*H, nSeq) with maxChunks = ceil(T/C), T = maxSeqT. a0 uses a
// COMPACT chunk layout [totalChunks, H, C, C]: sequence seq's chunk c lives
// at block index chunkBase(seq) + c, where chunkBase = sum over s<seq of
// ceil(seqT[s]/C) (computed in-kernel from seqT — an O(nSeq) scalar loop,
// nSeq <= 64). This keeps the a0 allocation proportional to the ACTUAL chunk
// count (sum(ceil(seqT/C))) instead of nSeq*maxChunks, which explodes on
// mixed decode+prefill forwards where most slots carry one token.
// seqT/seqOff/activeMask all nullptr => uniform T at k/beta stride seq*T and
// chunkBase = seq*maxChunks, math per (seq, chunk, head) bit-identical to
// nSeq single deltanet_kkt_solve calls. Blocks for chunks beyond a
// sequence's own ceil(Tseq/C) return (their a0 blocks are never read by
// K2); frozen slots (activeMask 0) return untouched.
extern "C" __global__ __launch_bounds__(256)
void deltanet_kkt_solve_batched(
    const float* __restrict__ kIn,
    const float* __restrict__ betaIn,
    float*       __restrict__ a0,
    const int T, const int H, const int S, const int C,
    const unsigned char* __restrict__ activeMask,
    const int* __restrict__ seqT,
    const int* __restrict__ seqOff)
{
    const int seq = blockIdx.y;
    if (activeMask != nullptr && activeMask[seq] == 0) return;
    const int maxChunks = (T + C - 1) / C;
    const int bid = blockIdx.x;                 // c*H + h (grid-side numbering)
    if (bid >= maxChunks * H) return;
    const int c = bid / H;
    const int h = bid % H;

    const int Tseq = (seqT != nullptr) ? seqT[seq] : T;
    const int nChunks = (Tseq + C - 1) / C;
    if (c >= nChunks) return;
    const int c0 = c * C;
    int cs = C;
    if (c0 + cs > Tseq) cs = Tseq - c0;

    // v4 fast path: a 1-token chunk's inverse is the 1x1 identity, and K2
    // only ever reads a0c[0] for it — skip the C*C zeroing (the DECODE slots
    // of a mixed serving forward all hit this).
    if (cs == 1) {
        if (threadIdx.x == 0) {
            size_t cb;
            if (seqT != nullptr) {
                cb = 0;
                for (int s = 0; s < seq; ++s) cb += (size_t)(seqT[s] + C - 1) / C;
            } else {
                cb = (size_t)seq * maxChunks;
            }
            a0[(cb + c) * H * C * C + (size_t)h * C * C] = 1.0f;
        }
        return;
    }

    const size_t tokBase = (seqOff != nullptr) ? (size_t)seqOff[seq]
                                               : (size_t)seq * (size_t)T;
    const float* __restrict__ k    = kIn    + tokBase * (size_t)H * S;
    const float* __restrict__ beta = betaIn + tokBase * H;

    size_t chunkBase;
    if (seqT != nullptr) {
        chunkBase = 0;
        for (int s = 0; s < seq; ++s) chunkBase += (size_t)(seqT[s] + C - 1) / C;
    } else {
        chunkBase = (size_t)seq * maxChunks;
    }

    const int t = threadIdx.x;
    float* a0c = a0 + (chunkBase + c) * H * C * C + (size_t)h * C * C;

    __shared__ float lt[DELTANET_KKT_MAX_C * DELTANET_KKT_MAX_C];

    for (int idx = t; idx < C * C; idx += blockDim.x) {
        a0c[idx] = 0.0f;
        lt[idx]  = 0.0f;
    }
    __syncthreads();

    // Phase 1: strict-lower Gram — v4 (5.21.9): the (a, m) entries are
    // independent, so distribute the PAIRS across the whole block instead of
    // one serial row per thread (the old form left thread a computing a dots
    // serially — the 2026-09-05 sub-split profile showed K1 at 9% of the
    // whole prefill, 2x the entire TC chunk forward). Each dot keeps its
    // serial ascending-i non-FMA order, so every entry is BIT-IDENTICAL to
    // the previous form — only the thread assignment changes.
    for (int p = t; p < C * C; p += blockDim.x) {
        const int a = p / C;
        const int m = p % C;
        if (a >= cs || m >= a) continue;
        const float* ka = k + (static_cast<size_t>(c0 + a) * H + h) * S;
        const float* km = k + (static_cast<size_t>(c0 + m) * H + h) * S;
        const float  ba = beta[(c0 + a) * H + h];
        float kk = 0.0f;
        for (int i = 0; i < S; ++i) kk = __fadd_rn(kk, __fmul_rn(ka[i], km[i]));
        lt[a * C + m] = __fmul_rn(ba, kk);
    }
    __syncthreads();

    // Phase 2: unit-lower inverse, thread = column m (unchanged).
    if (t < cs) {
        a0c[t * C + t] = 1.0f;
        for (int a = t + 1; a < cs; ++a) {
            float acc = 0.0f;
            for (int p = t; p < a; ++p)
                acc = __fadd_rn(acc, __fmul_rn(lt[a * C + p], a0c[p * C + t]));
            a0c[a * C + t] = -acc;
        }
    }
}