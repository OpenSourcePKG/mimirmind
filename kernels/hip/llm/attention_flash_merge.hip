// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// FlashAttention merge kernel — combine per-tile partials into the
// final attention output for decode mode (T_q == 1).
//
// HIP port of kernels/attention_flash_merge.cl. Reads
// (m_t, l_t, o_t_unnorm) per tile, computes the global softmax
// normaliser, and folds the partials into out[hq, d]:
//
//   m_final = max over t of m_t
//   beta_t  = exp(m_t - m_final)      (0 if m_t == -INFINITY)
//   l_final = sum_t (beta_t * l_t)
//   out[d]  = sum_t (beta_t * o_t_unnorm[d]) / l_final
//
// The partial kernel emits o_t_unnorm = sum_kk exp(score_kk - m_t) *
// v[kk][d], i.e. it already used m_t as the local subtractive bias.
// Multiplying by beta_t re-aligns to the global bias m_final, so the
// merged output equals the monolithic softmax attention exactly (up
// to fp32 reduction order).
//
// Layouts:
//   partialMlo [nHeads, nKTiles, (2 + headDim)]
//   out        [1, nHeads, headDim]
//
// Launch geometry:
//   dim3 grid ( nHeads, 1, 1 )
//   dim3 block( ATTN_FLASH_LOCAL, 1, 1 )
//
// WAVE32 note (RDNA3 gfx1101): ATTN_FLASH_LOCAL=16 means the workgroup
// occupies HALF a wave — same rationale as attention_flash_partial.
// No warp-shuffle reduction is needed here; the prelude runs on lid=0
// only, and the per-d loop is strided across the 16 active threads.

#include <hip/hip_runtime.h>

#include <math.h>   // for INFINITY

#ifndef ATTN_FLASH_LOCAL
#define ATTN_FLASH_LOCAL 16
#endif

// Keep the max-K-tiles envelope in sync with the L0 kernel + the
// host-side scratch allocator (kFlashMaxKTiles in GpuOps.hpp).
// At kFlashKTileSize=64, 512 tiles → 32768-token compile-time ceiling.
#ifndef ATTN_FLASH_MAX_KTILES
#define ATTN_FLASH_MAX_KTILES 512
#endif

#ifndef ATTN_FLASH_KTILE
#define ATTN_FLASH_KTILE 64
#endif

extern "C" __global__ __launch_bounds__(ATTN_FLASH_LOCAL)
void attention_flash_merge(
    const float* __restrict__ partialMlo,
          float* __restrict__ out,
    const int                 nHeads,
    const int                 headDim,
    const int* __restrict__   curLenPtr)
{
    const int hq  = blockIdx.x;
    const int lid = threadIdx.x;

    const int positionOffset = curLenPtr[0];
    const int kMax           = positionOffset + 1;
    const int nKTiles        = (kMax + ATTN_FLASH_KTILE - 1) / ATTN_FLASH_KTILE;

    const int stride = 2 + headDim;
    const float* __restrict__ baseMlo =
        partialMlo + static_cast<size_t>(hq)
                   * static_cast<size_t>(nKTiles)
                   * static_cast<size_t>(stride);

    __shared__ float alphas[ATTN_FLASH_MAX_KTILES];
    __shared__ float mFinalSlm;
    __shared__ float lFinalSlm;

    if (lid == 0) {
        float mFinal = -INFINITY;
        for (int t = 0; t < nKTiles; ++t) {
            const float mt = baseMlo[static_cast<size_t>(t)
                                   * static_cast<size_t>(stride) + 0];
            if (mt > mFinal) mFinal = mt;
        }
        // mFinal can still be -INFINITY if every tile is past kMax.
        // That only happens when there's no valid K position at all,
        // which the wrapper rejects. Treat as 0 here to avoid NaN.
        if (mFinal == -INFINITY) mFinal = 0.0f;

        float lFinal = 0.0f;
        for (int t = 0; t < nKTiles; ++t) {
            const float mt = baseMlo[static_cast<size_t>(t)
                                   * static_cast<size_t>(stride) + 0];
            const float lt = baseMlo[static_cast<size_t>(t)
                                   * static_cast<size_t>(stride) + 1];
            const float beta = (mt == -INFINITY) ? 0.0f : expf(mt - mFinal);
            alphas[t] = beta;
            lFinal  += beta * lt;
        }

        mFinalSlm = mFinal;
        lFinalSlm = lFinal;
    }
    __syncthreads();

    // lFinal == 0 only in the all-empty edge case — guard against
    // division by zero. Output stays at 0 which is the reasonable
    // "no information" answer.
    const float lFinal = lFinalSlm;
    const float invL   = (lFinal > 0.0f) ? (1.0f / lFinal) : 0.0f;

    float* __restrict__ oOut =
        out + static_cast<size_t>(hq) * static_cast<size_t>(headDim);

    for (int d = lid; d < headDim; d += ATTN_FLASH_LOCAL) {
        float acc = 0.0f;
        for (int t = 0; t < nKTiles; ++t) {
            const float ot_d =
                baseMlo[static_cast<size_t>(t)
                      * static_cast<size_t>(stride) + 2 + d];
            acc += alphas[t] * ot_d;
        }
        oOut[d] = acc * invL;
    }
}