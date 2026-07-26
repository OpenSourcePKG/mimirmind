// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Batched chunked GatedDeltaNet prefill stage K2 — M-Cuda.Batch variant of
// deltanet_chunk_forward. Runs the chunk forward (readout + state carry)
// for nSeq sequences in ONE launch, each with its own q/k/v/gCum/beta/a0,
// its own carried state and its own global scratch region. Math per
// (seq, head) is byte-identical to the single-sequence kernel; a
// per-sequence offset (blockIdx.y) is added to every base pointer.
//
// Per-sequence strides (all derived from T,H,S,C — no extra param):
//   q,k,v,out : [nSeq, T, H, S]        stride = T*H*S
//   gCum,beta : [nSeq, T, H]           stride = T*H
//   state     : [nSeq, H, S, S]        stride = H*S*S
//   a0        : [nSeq, nChunks, H, C, C]  stride = nChunks*H*C*C
//   scratch   : [nSeq, (H*S*S + 5*H*C*S)] stride = H*S*S + 5*H*C*S
// Launch: grid = dim3(H, nSeq, 1), block = S threads.

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
    const int T, const int H, const int S, const int C)
{
    const int seq = blockIdx.y;
    const int h   = blockIdx.x;
    const int j   = threadIdx.x;
    if (h >= H) return;

    const int nChunks = (T + C - 1) / C;
    const size_t actStride   = (size_t)T * H * S;               // q,k,v,out
    const size_t gateStride  = (size_t)T * H;                   // gCum,beta
    const size_t stateStride = (size_t)H * S * S;               // state
    const size_t a0Stride    = (size_t)nChunks * H * C * C;     // a0
    const size_t scratchStride = (size_t)H * S * S + 5 * (size_t)H * C * S;

    const float* __restrict__ q_    = q     + (size_t)seq * actStride;
    const float* __restrict__ k_    = k     + (size_t)seq * actStride;
    const float* __restrict__ v_    = v     + (size_t)seq * actStride;
    const float* __restrict__ gCum_ = gCum  + (size_t)seq * gateStride;
    const float* __restrict__ beta_ = beta  + (size_t)seq * gateStride;
    const float* __restrict__ a0_   = a0    + (size_t)seq * a0Stride;
    float*       __restrict__ out_  = out   + (size_t)seq * actStride;
    float*       __restrict__ state_   = state   + (size_t)seq * stateStride;
    float*       __restrict__ scratch_ = scratch + (size_t)seq * scratchStride;

    const float qScale = rsqrtf(static_cast<float>(S));

    const size_t hss = static_cast<size_t>(H) * S * S;   // s0 region size
    const size_t hcs = static_cast<size_t>(H) * C * S;   // one [H,C,S] region
    float* st  = state_   + static_cast<size_t>(h) * S * S;
    float* s0  = scratch_             + static_cast<size_t>(h) * S * S;
    float* uh  = scratch_ + hss       + static_cast<size_t>(h) * C * S;
    float* uqh = scratch_ + hss + hcs + static_cast<size_t>(h) * C * S;
    float* qsh = scratch_ + hss + 2 * hcs + static_cast<size_t>(h) * C * S;
    float* rph = scratch_ + hss + 3 * hcs + static_cast<size_t>(h) * C * S;
    float* dh  = scratch_ + hss + 4 * hcs + static_cast<size_t>(h) * C * S;

    __shared__ float egc[DELTANET_CHUNK_FWD_MAX_C];  // exp(gCum) (<=1, stable)
    __shared__ float gc[DELTANET_CHUNK_FWD_MAX_C];   // raw gCum (<=0)
    __shared__ float kqs[DELTANET_CHUNK_FWD_MAX_C * DELTANET_CHUNK_FWD_MAX_C];
    __shared__ float red[DELTANET_CHUNK_FWD_MAX_S];

    for (int c0 = 0; c0 < T; c0 += C) {
        const int cs  = (C < T - c0) ? C : (T - c0);
        const int cIx = c0 / C;
        const float* a0c = a0_ + (static_cast<size_t>(cIx) * H + h) * C * C;

        for (int i = 0; i < S; ++i) s0[i * S + j] = st[i * S + j];
        if (j < cs) {
            const float g = gCum_[(c0 + j) * H + h];
            gc[j]  = g;
            egc[j] = __expf(g);
        }
        __syncthreads();

        // Step 2: qs_a = qScale*q_a; u_a = S0^T k_a; uq_a = S0^T qs_a.
        for (int a = 0; a < cs; ++a) {
            const float* qa = q_ + (static_cast<size_t>(c0 + a) * H + h) * S;
            const float* ka = k_ + (static_cast<size_t>(c0 + a) * H + h) * S;
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
            const float* vm = v_ + (static_cast<size_t>(c0 + m) * H + h) * S;
            const float  bm = beta_[(c0 + m) * H + h];
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
                const float* km = k_ + (static_cast<size_t>(c0 + m) * H + h) * S;
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
            float* oa = out_ + (static_cast<size_t>(c0 + a) * H + h) * S;
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
                const float* km = k_ + (static_cast<size_t>(c0 + m) * H + h) * S;
                sij += (__expf(gc[cs - 1] - gc[m]) * km[i]) * dh[m * S + j];
            }
            st[i * S + j] = sij;
        }
        __syncthreads();
    }
}
