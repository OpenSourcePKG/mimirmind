// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// OEA — Opportunistic Expert Activation (Bragi 5.22). Training-free, batch-aware
// MoE routing that reduces the number of UNIQUE experts activated across a decode
// batch, so the grouped-MoE GEMM loads fewer expert weight matrices (moe.gemm is
// the dominant decode-bandwidth term at batch). Source: arxiv 2511.02237
// (Oncescu/Dao) — measured -39% MoE-decode @batch16 on Qwen3-30B, no significant
// accuracy loss. LOSSY (routing change) → quality-gated; env-gated + default OFF.
//
// Two device kernels, launched back-to-back per MoE layer (after moe_topk,
// before moe_group_build), rewriting the [T,K] top-K routing in place:
//
//   oea_active_mask  — decide which experts stay "active" for this batch:
//       (a) every token's TOP-1 expert is mandatory (never drop a token's best),
//       (b) any expert shared by >= minShare tokens across the batch's top-K
//           (popular enough to piggyback on). Singleton lower-rank experts
//           (pop < minShare, not a top-1) are dropped. A top-up guarantees
//           |active| >= K so every token can re-fill K slots.
//
//   oea_reroute      — re-select each token's top-K among ONLY the active set
//       (using the original router logits), renormalise the kept weights. Tokens
//       whose original K were all active are unchanged (lossless for them);
//       others piggyback onto the best active experts. Union after this <= |active|.
//
// No-op at T==1 (single-token decode has no batch to piggyback on) — the host
// gate skips the call, so single-user latency is untouched.

#include <cuda_runtime.h>

#ifndef OEA_MAX_EXPERTS
#define OEA_MAX_EXPERTS 256
#endif
#ifndef OEA_MAX_K
#define OEA_MAX_K 16
#endif
#define OEA_PER ((OEA_MAX_EXPERTS + 31) / 32)

// ---------------------------------------------------------------------------
// Kernel 1 — build the active-expert mask. One block, block.x threads,
// dynamic shared memory = nExperts * sizeof(int) for the popularity histogram.
// ---------------------------------------------------------------------------
extern "C" __global__ void oea_active_mask(
    const int* __restrict__ expIdx,    // [R = T*K] top-K expert ids (desc/token)
          int* __restrict__ active,    // [nExperts] out: 1 active, 0 dropped
    const int                R,
    const int                nExperts,
    const int                K,
    const int                minShare)
{
    extern __shared__ int pop[];       // [nExperts]
    const int tid = static_cast<int>(threadIdx.x);
    const int nt  = static_cast<int>(blockDim.x);

    for (int e = tid; e < nExperts; e += nt) {
        pop[e] = 0;
    }
    __syncthreads();

    // Popularity histogram over all T*K assignments.
    for (int i = tid; i < R; i += nt) {
        const int e = expIdx[i];
        if (e >= 0 && e < nExperts) {
            atomicAdd(&pop[e], 1);
        }
    }
    __syncthreads();

    // (b) shared experts (pop >= minShare) start active.
    for (int e = tid; e < nExperts; e += nt) {
        active[e] = (pop[e] >= minShare) ? 1 : 0;
    }
    __syncthreads();

    // (a) mandatory: every token's top-1 (assignment t*K + 0) stays active.
    const int T = (K > 0) ? (R / K) : 0;
    for (int t = tid; t < T; t += nt) {
        const int e = expIdx[t * K];
        if (e >= 0 && e < nExperts) {
            active[e] = 1;                 // benign concurrent write of the same 1
        }
    }
    __syncthreads();

    // Guarantee |active| >= K (top up by descending popularity) so oea_reroute
    // can always re-fill K slots. Rare at serving batch (|active| >> K), but
    // makes the reroute total. Thread 0 — the set is tiny (<= nExperts).
    if (tid == 0) {
        int cnt = 0;
        for (int e = 0; e < nExperts; ++e) {
            cnt += active[e];
        }
        while (cnt < K && cnt < nExperts) {
            int best = -1;
            int bestPop = -1;
            for (int e = 0; e < nExperts; ++e) {
                if (!active[e] && pop[e] > bestPop) {
                    bestPop = pop[e];
                    best    = e;
                }
            }
            if (best < 0) {
                break;
            }
            active[best] = 1;
            ++cnt;
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 2 — re-select each token's top-K among the active set. One warp per
// token (grid.x = T, block.x = 32). Structure mirrors moe_topk exactly (same
// warp-parallel argmax, same renormalisation); the ONLY difference is that an
// inactive expert's logit is masked to -FLT_MAX so it never wins a pass.
// ---------------------------------------------------------------------------
extern "C" __global__ void oea_reroute(
    const float* __restrict__ logits,     // [T, nExperts] router scores (F32)
    const int*   __restrict__ active,     // [nExperts] 1 if active
          int*   __restrict__ outIdx,     // [T, K] rewritten in place
          float* __restrict__ outWeight,  // [T, K] rewritten in place
    const int                 nExperts,
    const int                 K,
    const float               wScale)
{
    const unsigned FULL = 0xffffffffu;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int t    = static_cast<int>(blockIdx.x);
    const int kk   = (K < OEA_MAX_K) ? K : OEA_MAX_K;

    const float* __restrict__ row    = logits    + static_cast<size_t>(t) * nExperts;
    int*         __restrict__ idxOut = outIdx    + static_cast<size_t>(t) * K;
    float*       __restrict__ wOut   = outWeight + static_cast<size_t>(t) * K;

    const float NEG_BIG = -3.402823466e+38f;

    float lg[OEA_PER];
    int   eid[OEA_PER];
#pragma unroll
    for (int j = 0; j < OEA_PER; ++j) {
        const int e = lane + j * 32;
        eid[j] = e;
        // Masked load: only active experts compete; the rest sink to -FLT_MAX.
        lg[j]  = (e < nExperts && active[e]) ? row[e] : NEG_BIG;
    }

    // global max over active experts (warp butterfly)
    float m = NEG_BIG;
#pragma unroll
    for (int j = 0; j < OEA_PER; ++j) {
        m = fmaxf(m, lg[j]);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(FULL, m, o));
    }

    float ex[OEA_PER];
#pragma unroll
    for (int j = 0; j < OEA_PER; ++j) {
        ex[j] = (eid[j] < nExperts && active[eid[j]]) ? __expf(lg[j] - m) : 0.0f;
    }
    bool taken[OEA_PER];
#pragma unroll
    for (int j = 0; j < OEA_PER; ++j) {
        taken[j] = false;
    }

    float keptSum = 0.0f;
    float selEx[OEA_MAX_K];
    for (int k = 0; k < kk; ++k) {
        float lb  = -1.0f;
        int   lbi = 0x7fffffff;
#pragma unroll
        for (int j = 0; j < OEA_PER; ++j) {
            if (!taken[j] && eid[j] < nExperts && active[eid[j]] && ex[j] > lb) {
                lb  = ex[j];
                lbi = eid[j];
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(FULL, lb,  o);
            const int   oi = __shfl_xor_sync(FULL, lbi, o);
            if (ov > lb || (ov == lb && oi < lbi)) {
                lb  = ov;
                lbi = oi;
            }
        }
#pragma unroll
        for (int j = 0; j < OEA_PER; ++j) {
            if (eid[j] == lbi) {
                taken[j] = true;
            }
        }
        keptSum += lb;
        if (lane == 0) {
            idxOut[k] = lbi;
            selEx[k]  = lb;
        }
    }

    if (lane == 0) {
        const float invKept = (keptSum > 0.0f) ? (1.0f / keptSum) : 1.0f;
        for (int k = 0; k < kk; ++k) {
            wOut[k] = selEx[k] * invKept * wScale;
        }
    }
}
