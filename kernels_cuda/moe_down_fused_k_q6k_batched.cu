// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Batched fused MoE down-projection (Q6_K) — M-Cuda.Batch batched variant of
// moe_down_fused_k_q6k. Processes nSeq decode tokens, each with its own gate
// activations, routed-expert list, router weights and output accumulator, in
// ONE launch. The Q6_K expert bank is shared. Math per (seq, n) is
// byte-identical to the single-token kernel (same K-loop, Q6_K dequant, Kahan
// folds, warp16 reduce, read-modify-write); only a per-sequence offset
// (blockIdx.y) is added to X / expIdx / kw / accum.
//
// The Q4_K_XL "unsloth-dynamic" GGUF quantizes a few layers' ffn_down_exps as
// Q6_K instead of Q5_K (blk 34/38/39 for Qwen3.6-35B). Without this batched
// variant those layers fell back to the per-token runMoeFfn on the serving
// path — costly at high nSeq (nSeq sequential MoE evals vs one batched).
//
// Layout (per-sequence strides derive from ffPer,dModel,kActive):
//   X (gateAct) : [nSeq, kActive, ffPer]  seqStride = kActive*ffPer
//   expIdx      : [nSeq, kActive]         seqStride = kActive
//   kw          : [nSeq, kActive]         seqStride = kActive
//   accum       : [nSeq, dModel]          seqStride = dModel (read-modify-write)
//   W           : Q6_K expert bank, shared across sequences
// Launch: grid = dim3(ceil(dModel / OUTPUTS_PER_GROUP), nSeq, 1),
//         block = MOE_DOWN_LOCAL.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifndef MOE_DOWN_LOCAL
#define MOE_DOWN_LOCAL 64
#endif

#ifndef MOE_DOWN_SG
#define MOE_DOWN_SG 16
#endif

#define MOE_DOWN_OUTPUTS_PER_GROUP (MOE_DOWN_LOCAL / MOE_DOWN_SG)

#define Q6K_BLOCK_ELEMENTS 256
#define Q6K_BLOCK_BYTES    210
#define X_TILE_ELEMENTS    1024

static __device__ __forceinline__ float warp16_reduce_sum(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 8, 16);
    v += __shfl_xor_sync(0xffffffffu, v, 4, 16);
    v += __shfl_xor_sync(0xffffffffu, v, 2, 16);
    v += __shfl_xor_sync(0xffffffffu, v, 1, 16);
    return v;
}

extern "C" __global__ __launch_bounds__(MOE_DOWN_LOCAL)
void moe_down_fused_k_q6k_batched(
    const float*         __restrict__ X,          // [nSeq, K, ffPer]
    const unsigned char* __restrict__ W,          // Q6_K expert bank (shared)
    const int*           __restrict__ expIdx,     // [nSeq, K]
    const float*         __restrict__ kw,         // [nSeq, K]
          float*         __restrict__ accum,      // [nSeq, dModel] RMW
    const int                         ffPer,
    const int                         dModel,
    const int                         kActive,
    const int                         expertBytes)
{
    __shared__ float xTile[X_TILE_ELEMENTS];

    const int seq = blockIdx.y;
    const float* __restrict__ Xb =
        X + static_cast<size_t>(seq) * kActive * ffPer;
    const int*   __restrict__ expIdxB = expIdx + static_cast<size_t>(seq) * kActive;
    const float* __restrict__ kwB     = kw     + static_cast<size_t>(seq) * kActive;
    float*       __restrict__ accumB  = accum  + static_cast<size_t>(seq) * dModel;

    const int  wg       = blockIdx.x;
    const int  tid      = threadIdx.x;
    const int  lsize    = blockDim.x;
    const int  sgInWg   = tid / MOE_DOWN_SG;    // 0..3
    const int  sgLocal  = tid % MOE_DOWN_SG;    // 0..15
    const int  n        = wg * MOE_DOWN_OUTPUTS_PER_GROUP + sgInWg;
    const bool active   = (n < dModel);
    const int  nSuper   = ffPer / Q6K_BLOCK_ELEMENTS;
    const int  rowBytes = nSuper * Q6K_BLOCK_BYTES;

    float          accumSum = 0.0f;
    volatile float accumKc  = 0.0f;

    for (int k = 0; k < kActive; ++k) {
        const int   e   = expIdxB[k];
        const float ekw = kwB[k];

        const unsigned char* __restrict__ Wexpert =
            W + static_cast<size_t>(e) * static_cast<size_t>(expertBytes);
        const float* __restrict__ Xk =
            Xb + static_cast<size_t>(k) * static_cast<size_t>(ffPer);

        float          sum = 0.0f;
        volatile float kc  = 0.0f;

        for (int tile = 0; tile < ffPer; tile += X_TILE_ELEMENTS) {
            const int tileK = (X_TILE_ELEMENTS < ffPer - tile)
                                ? X_TILE_ELEMENTS : (ffPer - tile);
            for (int i = tid; i < tileK; i += lsize) {
                xTile[i] = Xk[tile + i];
            }
            __syncthreads();

            if (active) {
                const unsigned char* __restrict__ row =
                    Wexpert + static_cast<size_t>(n)
                            * static_cast<size_t>(rowBytes);

                const int sbStart  = tile / Q6K_BLOCK_ELEMENTS;
                const int sbInTile = X_TILE_ELEMENTS / Q6K_BLOCK_ELEMENTS;
                const int sbEnd    = (sbStart + sbInTile < nSuper)
                                       ? (sbStart + sbInTile)
                                       : nSuper;

                for (int sb = sbStart; sb < sbEnd; ++sb) {
                    const unsigned char* __restrict__ block =
                        row + sb * Q6K_BLOCK_BYTES;

                    const unsigned char* ql = block;
                    const unsigned char* qh = block + 128;
                    const signed char*   sc =
                        reinterpret_cast<const signed char*>(block + 192);
                    const __half*        d_ptr =
                        reinterpret_cast<const __half*>(block + 208);
                    const float d = __half2float(d_ptr[0]);

                    const int xLocalBase = (sb - sbStart) * Q6K_BLOCK_ELEMENTS;

                    for (int hIdx = 0; hIdx < 2; ++hIdx) {
                        const int xHalfBase = xLocalBase + hIdx * 128;
                        const unsigned char* qlp = ql + hIdx * 64;
                        const unsigned char* qhp = qh + hIdx * 32;
                        const signed char*   scp = sc + hIdx * 8;

                        for (int l = sgLocal; l < 32; l += MOE_DOWN_SG) {
                            const int is = l / 16;

                            const unsigned int ql0  = qlp[l +  0];
                            const unsigned int ql32 = qlp[l + 32];
                            const unsigned int qhv  = qhp[l];

                            const int q1 = static_cast<int>(
                                (ql0  & 0x0Fu) | (((qhv >> 0) & 0x03u) << 4)) - 32;
                            const int q2 = static_cast<int>(
                                (ql32 & 0x0Fu) | (((qhv >> 2) & 0x03u) << 4)) - 32;
                            const int q3 = static_cast<int>(
                                (ql0  >> 4)    | (((qhv >> 4) & 0x03u) << 4)) - 32;
                            const int q4 = static_cast<int>(
                                (ql32 >> 4)    | (((qhv >> 6) & 0x03u) << 4)) - 32;

                            const float s0 = d * static_cast<float>(scp[is + 0]);
                            const float s2 = d * static_cast<float>(scp[is + 2]);
                            const float s4 = d * static_cast<float>(scp[is + 4]);
                            const float s6 = d * static_cast<float>(scp[is + 6]);

                            #define KAHAN_ADD(dest, comp, term)                 \
                                do {                                             \
                                    const float _y = (term) - (comp);            \
                                    const float _t = (dest) + _y;                \
                                    (comp) = (_t - (dest)) - _y;                 \
                                    (dest) = _t;                                 \
                                } while (0)

                            KAHAN_ADD(sum, kc, xTile[xHalfBase + l +  0] * (s0 * static_cast<float>(q1)));
                            KAHAN_ADD(sum, kc, xTile[xHalfBase + l + 32] * (s2 * static_cast<float>(q2)));
                            KAHAN_ADD(sum, kc, xTile[xHalfBase + l + 64] * (s4 * static_cast<float>(q3)));
                            KAHAN_ADD(sum, kc, xTile[xHalfBase + l + 96] * (s6 * static_cast<float>(q4)));

                            #undef KAHAN_ADD
                        }
                    }
                }
            }

            __syncthreads();
        }

        const float lanePartial = (sum + kc) * ekw;
        const float _y = lanePartial - accumKc;
        const float _t = accumSum + _y;
        accumKc  = (_t - accumSum) - _y;
        accumSum = _t;
    }

    accumSum += accumKc;
    accumSum = warp16_reduce_sum(accumSum);

    if (active && sgLocal == 0) {
        accumB[n] += accumSum;   // read-modify-write
    }
}
