// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Device-side MoE top-K router for Kolibri-1 (DeepSeek-V3-style sigmoid gate).
//
// Variant of moe_topk.cu (M-Q3N.5) for the Kolibri-1 routing math, so the MoE
// block (router matmul -> top-K -> FP4-TC fused-K gate/up/down) runs as one
// uninterrupted device stream with NO per-layer host sync — the serialization
// that pins the batched decode trunk.
//
// Math per token t (mirrors the host loop in Kolibri1Backend::runKolibriMoe):
//   select = indices of the top-K largest (logit[e] + bias[e])   (DeepSeek-V3
//            select-on-bias; ties: lowest expert index)
//   w[k]   = sigmoid(logit[select[k]])        (UNBIASED logit, NO softmax, NO
//            renormalisation — norm_topk_prob=false) * wScale
//
// Warp-parallel like moe_topk.cu: one warp (block.x=32) per token, lane l owns
// experts l, l+32, …; the K-pass argmax is a warp butterfly reduction. The
// selection carries the raw (unbiased) logit of the winner alongside its score
// so the sigmoid weight uses the unbiased value.

#include <cuda_runtime.h>

#ifndef MOE_TOPK_MAX_EXPERTS
#define MOE_TOPK_MAX_EXPERTS 512   // Kolibri-1 has 384 routed experts
#endif
#ifndef MOE_TOPK_MAX_K
#define MOE_TOPK_MAX_K 16
#endif
#define MOE_TOPK_PER ((MOE_TOPK_MAX_EXPERTS + 31) / 32)

// One block (one warp) per token. Grid.x = T, block.x = 32.
extern "C" __global__ void moe_topk_sigmoid_bias(
    const float* __restrict__ logits,      // [T, nExperts]  router scores (F32)
    const float* __restrict__ bias,        // [nExperts]     selection bias (F32)
          int*   __restrict__ outIdx,      // [T, K]         expert indices (desc)
          float* __restrict__ outWeight,   // [T, K]         sigmoid weights * wScale
    const int                 nExperts,
    const int                 K,
    const float               wScale)
{
    const unsigned FULL = 0xffffffffu;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int t    = static_cast<int>(blockIdx.x);
    const int kk   = (K < MOE_TOPK_MAX_K) ? K : MOE_TOPK_MAX_K;

    const float* __restrict__ row    = logits    + static_cast<size_t>(t) * nExperts;
    int*         __restrict__ idxOut = outIdx    + static_cast<size_t>(t) * K;
    float*       __restrict__ wOut   = outWeight + static_cast<size_t>(t) * K;

    const float NEG_BIG = -3.402823466e+38f;   // invalid experts sink below any score

    // Each lane holds its experts' raw logit + selection score (logit + bias).
    float lg[MOE_TOPK_PER];
    float sc[MOE_TOPK_PER];
    int   eid[MOE_TOPK_PER];
    bool  taken[MOE_TOPK_PER];
#pragma unroll
    for (int j = 0; j < MOE_TOPK_PER; ++j) {
        const int e = lane + j * 32;
        eid[j]   = e;
        lg[j]    = (e < nExperts) ? row[e] : NEG_BIG;
        sc[j]    = (e < nExperts) ? (row[e] + bias[e]) : NEG_BIG;
        taken[j] = false;
    }

    // K-pass argmax on the biased score, descending; tie-break lowest index.
    for (int k = 0; k < kk; ++k) {
        float lb   = NEG_BIG;        // best biased score this pass
        int   lbi  = 0x7fffffff;     // its expert index
        float lblg = 0.0f;           // its UNBIASED logit (for the sigmoid weight)
#pragma unroll
        for (int j = 0; j < MOE_TOPK_PER; ++j) {
            if (!taken[j] && eid[j] < nExperts && sc[j] > lb) {
                lb = sc[j]; lbi = eid[j]; lblg = lg[j];
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov  = __shfl_xor_sync(FULL, lb,   o);
            const int   oi  = __shfl_xor_sync(FULL, lbi,  o);
            const float olg = __shfl_xor_sync(FULL, lblg, o);
            if (ov > lb || (ov == lb && oi < lbi)) {
                lb = ov; lbi = oi; lblg = olg;
            }
        }
#pragma unroll
        for (int j = 0; j < MOE_TOPK_PER; ++j) {
            if (eid[j] == lbi) { taken[j] = true; }
        }
        if (lane == 0) {
            idxOut[k] = lbi;
            // sigmoid(unbiased logit), no renormalisation.
            wOut[k]   = (1.0f / (1.0f + __expf(-lblg))) * wScale;
        }
    }
}
