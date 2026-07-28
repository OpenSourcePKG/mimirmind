// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Chunked GatedDeltaNet prefill — stage K2: chunk forward (readout + state
// carry). Consumes G (K0, gCum) and A0 (K1) and produces `out` + the carried
// `state`. Direct port of compute::deltanetChunkForward (GatedDeltaNet.cpp).
//
// Layout: q,k,v,out [T,H,S]; gCum,beta [T,H]; state [H,S,S] (s[i*S+j]);
//         a0 [nChunks,H,C,C]. qScale = 1/sqrt(S).
//
// Grid = H (one block per head); chunks are processed sequentially inside the
// block because the state carry (step 6) feeds the next chunk. Block = S
// threads, thread j owns state column j (mirrors the AR kernel's column-per-
// lane idea). Per-head scratch (s0/u/uq/qs/rp/d) lives in global memory (the
// 128×128 S0 snapshot exceeds the shared budget); egc/ieg and the per-(a,m)
// kq reduction use shared. Correctness-first (not perf-tuned): the O(C^2) kq
// block-reductions are cheap at the test width and can be optimised later.
//
// Assumes S is a power of two (head_dim; 16 in tests, 128 in prod) for the
// tree reduction, S <= MAX_S, and C <= MAX_C.

#include <hip/hip_runtime.h>

#ifndef DELTANET_CHUNK_FWD_MAX_S
#define DELTANET_CHUNK_FWD_MAX_S 128
#endif
#ifndef DELTANET_CHUNK_FWD_MAX_C
#define DELTANET_CHUNK_FWD_MAX_C 64
#endif

extern "C" __global__ __launch_bounds__(DELTANET_CHUNK_FWD_MAX_S)
void deltanet_chunk_forward(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ gCum,
    const float* __restrict__ beta,
    const float* __restrict__ a0,
    float*       __restrict__ state,
    float*       __restrict__ out,
    // One scratch arg (kMaxArgs=16): s0[H,S,S] then u|uq|qs|rp|d ([H,C,S] each).
    float*       __restrict__ scratch,
    const int T, const int H, const int S, const int C)
{
    const int h = blockIdx.x;
    const int j = threadIdx.x;
    if (h >= H) return;

    const float qScale = rsqrtf(static_cast<float>(S));

    const size_t hss = static_cast<size_t>(H) * S * S;   // s0 region size
    const size_t hcs = static_cast<size_t>(H) * C * S;   // one [H,C,S] region
    float* st  = state   + static_cast<size_t>(h) * S * S;
    float* s0  = scratch             + static_cast<size_t>(h) * S * S;
    float* uh  = scratch + hss       + static_cast<size_t>(h) * C * S;
    float* uqh = scratch + hss + hcs + static_cast<size_t>(h) * C * S;
    float* qsh = scratch + hss + 2 * hcs + static_cast<size_t>(h) * C * S;
    float* rph = scratch + hss + 3 * hcs + static_cast<size_t>(h) * C * S;
    float* dh  = scratch + hss + 4 * hcs + static_cast<size_t>(h) * C * S;

    __shared__ float egc[DELTANET_CHUNK_FWD_MAX_C];  // exp(gCum) (<=1, stable)
    __shared__ float gc[DELTANET_CHUNK_FWD_MAX_C];   // raw gCum (<=0)
    __shared__ float kqs[DELTANET_CHUNK_FWD_MAX_C * DELTANET_CHUNK_FWD_MAX_C];
    __shared__ float red[DELTANET_CHUNK_FWD_MAX_S];

    for (int c0 = 0; c0 < T; c0 += C) {
        const int cs  = (C < T - c0) ? C : (T - c0);
        const int cIx = c0 / C;
        const float* a0c = a0 + (static_cast<size_t>(cIx) * H + h) * C * C;

        // Snapshot chunk-start state (column j across all rows).
        for (int i = 0; i < S; ++i) s0[i * S + j] = st[i * S + j];
        if (j < cs) {
            // Keep raw gCum (<=0); exp(-gCum) alone overflows to +inf for
            // large gate logs and egc*ieg then gives 0*inf=nan. All decays
            // below are formed as exp(gc[a]-gc[m]) (a>=m -> exponent <=0).
            const float g = gCum[(c0 + j) * H + h];
            gc[j]  = g;
            egc[j] = __expf(g);
        }
        __syncthreads();

        // Step 2: qs_a = qScale*q_a; u_a = S0^T k_a; uq_a = S0^T qs_a.
        for (int a = 0; a < cs; ++a) {
            const float* qa = q + (static_cast<size_t>(c0 + a) * H + h) * S;
            const float* ka = k + (static_cast<size_t>(c0 + a) * H + h) * S;
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

        // Step 3: r'_m = exp(-G_m) beta_m (v_m - exp(G_m) u_m).
        for (int m = 0; m < cs; ++m) {
            const float* vm = v + (static_cast<size_t>(c0 + m) * H + h) * S;
            const float  bm = beta[(c0 + m) * H + h];
            rph[m * S + j] = bm * (vm[j] - egc[m] * uh[m * S + j]);
        }
        __syncthreads();

        // Step 4: d_a = exp(G_a) * sum_{m<=a} a0[a,m] r'_m.
        for (int a = 0; a < cs; ++a) {
            float dj = 0.0f;
            for (int m = 0; m <= a; ++m)
                dj += a0c[a * C + m] * __expf(gc[a] - gc[m]) * rph[m * S + j];
            dh[a * S + j] = dj;
        }
        __syncthreads();

        // Step 4.5: kq[a,m] = k_m . qs_a  (reduction over i == thread j).
        for (int a = 0; a < cs; ++a) {
            for (int m = 0; m <= a; ++m) {
                const float* km = k + (static_cast<size_t>(c0 + m) * H + h) * S;
                red[j] = km[j] * qsh[a * S + j];
                __syncthreads();
                for (int off = S >> 1; off > 0; off >>= 1) {
                    if (j < off) red[j] += red[j + off];
                    __syncthreads();
                }
                if (j == 0) kqs[a * C + m] = red[0];
                __syncthreads();
            }
        }

        // Step 5: o_a = exp(G_a) uq_a + sum_{m<=a} exp(G_a-G_m)(k_m.qs_a) d_m.
        for (int a = 0; a < cs; ++a) {
            float* oa = out + (static_cast<size_t>(c0 + a) * H + h) * S;
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
                const float* km = k + (static_cast<size_t>(c0 + m) * H + h) * S;
                sij += (__expf(gc[cs - 1] - gc[m]) * km[i]) * dh[m * S + j];
            }
            st[i * S + j] = sij;
        }
        __syncthreads();
    }
}