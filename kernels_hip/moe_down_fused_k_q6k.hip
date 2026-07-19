// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Fused MoE down-projection for T=1 decode — Q6_K variant.
//
// Q6_K = 210-byte super-blocks (128 B ql + 64 B qh + 16 B sc + 2 B d),
// 256 elements each. Same read-modify-write semantics as the Q8_0
// sibling kernel; same launch geometry (WgSize=64, 4 sub-groups of 16
// lanes) so the dispatcher can share its group formula.
//
//   accum[n] += sum_{k=0..K-1} kw[k] * sum_{l=0..ffPer-1}
//                                dequant_q6k(W[expIdx[k]], n, l)
//                                * gateAct[k, l]
//
// Dequant math mirrors kernels_hip/matmul_q6k_vec.hip: each super-block
// splits into two 128-element halves; per lane 4 elements at positions
// l, l+32, l+64, l+96 within a half get reconstructed from `ql` +
// 2-bit-pairs of `qh` and scaled by `d * sc[is + {0,2,4,6}]`.
//
// FP32 accumulation across K experts adds up to ~K*ffPer/16 fmas per
// lane per output — well past the cancellation threshold at K=8,
// ffPer=2048. Match the L0 kernel and keep Kahan compensation both
// inside the K-loop (per-expert sum) and across K experts.
//
// WAVE32 (gfx1101) — `warp16_reduce_sum` uses `__shfl_xor` with a mask
// of 16 to fold within the sub-group only. Reuses the exact idiom from
// moe_down_fused_k_q8_0.hip.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#ifndef MOE_DOWN_LOCAL
#define MOE_DOWN_LOCAL 64
#endif

#ifndef MOE_DOWN_SG
#define MOE_DOWN_SG 16
#endif

#define MOE_DOWN_OUTPUTS_PER_GROUP (MOE_DOWN_LOCAL / MOE_DOWN_SG)

#define Q6K_BLOCK_ELEMENTS 256
#define Q6K_BLOCK_BYTES    210

// 1024 elements = 4 super-blocks = 4 KiB SLM per workgroup — matches
// matmul_q6k_vec.hip so occupancy characteristics stay comparable.
#define X_TILE_ELEMENTS 1024

static __device__ __forceinline__ float warp16_reduce_sum(float v) {
    v += __shfl_xor(v, 8, 16);
    v += __shfl_xor(v, 4, 16);
    v += __shfl_xor(v, 2, 16);
    v += __shfl_xor(v, 1, 16);
    return v;
}

extern "C" __global__ __launch_bounds__(MOE_DOWN_LOCAL)
void moe_down_fused_k_q6k(
    const float*         __restrict__ X,          // [K, ffPer]
    const unsigned char* __restrict__ W,          // Q6_K expert bank base
    const int*           __restrict__ expIdx,     // [K]
    const float*         __restrict__ kw,         // [K] router × down-scale
          float*         __restrict__ accum,      // [dModel] read-modify-write
    const int                         ffPer,
    const int                         dModel,
    const int                         kActive,
    const int                         expertBytes)
{
    __shared__ float xTile[X_TILE_ELEMENTS];

    const int  wg       = blockIdx.x;
    const int  tid      = threadIdx.x;
    const int  lsize    = blockDim.x;
    const int  sgInWg   = tid / MOE_DOWN_SG;    // 0..3
    const int  sgLocal  = tid % MOE_DOWN_SG;    // 0..15
    const int  n        = wg * MOE_DOWN_OUTPUTS_PER_GROUP + sgInWg;
    const bool active   = (n < dModel);
    const int  nSuper   = ffPer / Q6K_BLOCK_ELEMENTS;
    const int  rowBytes = nSuper * Q6K_BLOCK_BYTES;

    // Kahan-compensated outer accumulator across K active experts. Each
    // expert's per-lane partial `sum` sits inside the K-loop; we fold
    // its (Kahan-corrected) result into `accumSum` scaled by `kw[k]`.
    float          accumSum = 0.0f;
    volatile float accumKc  = 0.0f;

    for (int k = 0; k < kActive; ++k) {
        const int   e   = expIdx[k];
        const float ekw = kw[k];

        const unsigned char* __restrict__ Wexpert =
            W + static_cast<size_t>(e) * static_cast<size_t>(expertBytes);
        const float* __restrict__ Xk =
            X + static_cast<size_t>(k) * static_cast<size_t>(ffPer);

        // Per-expert per-row dot product accumulator (Kahan). Reset
        // each K iteration; the outer fold scales by kw[k] and moves
        // the partial into the persistent `accumSum`.
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

                    // Two 128-element halves; per lane iterates l in
                    // [sgLocal, sgLocal+16) with step MOE_DOWN_SG=16,
                    // i.e. l takes values sgLocal and sgLocal+16 (2
                    // iterations per lane per half). Each l processes
                    // 4 elements at offsets 0/32/64/96 inside the half.
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

        // Fold this expert's Kahan-corrected partial into the outer
        // accumulator scaled by kw[k]. Kahan applied here too — the
        // outer sum has K terms typically 8 experts, per-lane already
        // holds a ~ffPer/16-element sum whose magnitude dominates.
        const float lanePartial = (sum + kc) * ekw;
        const float _y = lanePartial - accumKc;
        const float _t = accumSum + _y;
        accumKc  = (_t - accumSum) - _y;
        accumSum = _t;
    }

    accumSum += accumKc;
    accumSum = warp16_reduce_sum(accumSum);

    if (active && sgLocal == 0) {
        accum[n] += accumSum;   // read-modify-write
    }
}
