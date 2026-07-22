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