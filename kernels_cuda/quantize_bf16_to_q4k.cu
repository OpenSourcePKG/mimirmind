// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// BF16 -> Q4_K weight quantizer (load-time, one-shot).
//
//   src: [rows, K]        BF16 dense weights, row-major (K % 256 == 0)
//   dst: [rows, K/256]     Q4_K super-blocks (144 bytes each), row-major
//
// Used by the NVFP4 loader to re-quantise the materialised BF16 MoE experts
// to 4-bit Q4_K so the memory-bound decode reads ~3.5x less expert traffic.
// Unlike a flat linear Q8_0 (which crushes the small log-distributed weights
// and breaks coherence), Q4_K keeps per-32 sub-block scale+min (asymmetric).
//
// The per-sub-block scale/min use llama.cpp's make_qkx2_quants (least-squares
// refinement over 21 candidate scales + importance weights w = av|x| + |x|) —
// a naive min/max fit is coherent for short outputs but degenerates on long
// ones. This matches the GGUF Q4_K the model was validated against.
//
// Q4_K super-block layout (bit-identical to llama.cpp / Q4K.cpp:63-104):
//   fp16  d        (bytes 0..1)    scale-of-scales
//   fp16  dmin     (bytes 2..3)    scale-of-mins
//   u8    scales[12](bytes 4..15)  8 x (6-bit scale, 6-bit min), get_scale_min_k4
//   u8    qs[128]  (bytes 16..143) 256 nibbles
//   element[i in sub-block s] = d*sc_s*q - dmin*m_s
//
// One thread block per super-block, blockDim = 256 (one thread per element).
// Launch: grid( totalElems/256, 1, 1 ), block( 256, 1, 1 ).

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#ifndef QUANTIZE_Q4K_LOCAL
#define QUANTIZE_Q4K_LOCAL 256
#endif

#define Q4K_BLOCK_ELEMENTS 256
#define Q4K_BLOCK_BYTES    144

namespace {

// llama.cpp make_qkx2_quants (nmax=15, rmin=-1, rdelta=0.1, nstep=20,
// use_mad=false), reduced to return scale + the_min (the 4-bit L is
// recomputed later from the block-quantised scales). x/w are the 32
// sub-block values + importance weights. Reconstruction is scale*q - the_min.
__device__ void makeQkx2(const float* __restrict__ x, const float* __restrict__ w,
                         float& outScale, float& outMin) {
    float mn = x[0], mx = x[0];
    float sum_w = w[0], sum_x = w[0] * x[0];
    for (int i = 1; i < 32; ++i) {
        mn = fminf(mn, x[i]);
        mx = fmaxf(mx, x[i]);
        sum_w += w[i];
        sum_x += w[i] * x[i];
    }
    if (mn > 0.0f) mn = 0.0f;
    if (mx == mn) { outScale = 0.0f; outMin = -mn; return; }

    float iscale = 15.0f / (mx - mn);
    float scale  = 1.0f / iscale;
    float best_mad = 0.0f;
    for (int i = 0; i < 32; ++i) {
        int l = static_cast<int>(nearbyintf(iscale * (x[i] - mn)));
        l = max(0, min(15, l));
        const float diff = scale * static_cast<float>(l) + mn - x[i];
        best_mad += w[i] * diff * diff;
    }
    float bestScale = scale, bestMin = mn;
    for (int is = 0; is <= 20; ++is) {
        iscale = (-1.0f + 0.1f * static_cast<float>(is) + 15.0f) / (mx - mn);
        float sum_l = 0.0f, sum_l2 = 0.0f, sum_xl = 0.0f;
        for (int i = 0; i < 32; ++i) {
            int l = static_cast<int>(nearbyintf(iscale * (x[i] - mn)));
            l = max(0, min(15, l));
            const float wi = w[i];
            const float lf = static_cast<float>(l);
            sum_l  += wi * lf;
            sum_l2 += wi * lf * lf;
            sum_xl += wi * lf * x[i];
        }
        const float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0.0f) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l) / D;
            float this_min   = (sum_l2 * sum_x - sum_l * sum_xl) / D;
            if (this_min > 0.0f) {
                this_min   = 0.0f;
                this_scale = (sum_l2 > 0.0f) ? sum_xl / sum_l2 : 0.0f;
            }
            float mad = 0.0f;
            for (int i = 0; i < 32; ++i) {
                int l = static_cast<int>(nearbyintf(iscale * (x[i] - mn)));
                l = max(0, min(15, l));
                const float diff = this_scale * static_cast<float>(l) + this_min - x[i];
                mad += w[i] * diff * diff;
            }
            if (mad < best_mad) {
                best_mad  = mad;
                bestScale = this_scale;
                bestMin   = this_min;
            }
        }
    }
    outScale = bestScale;
    outMin   = -bestMin;   // >= 0
}

} // namespace

extern "C" __global__ __launch_bounds__(QUANTIZE_Q4K_LOCAL)
void quantize_bf16_to_q4k(
    const __nv_bfloat16* __restrict__ src,   // [totalElems]
          unsigned char* __restrict__ dst)   // [totalBlocks * 144]
{
    __shared__ float        xs[Q4K_BLOCK_ELEMENTS];
    __shared__ float        scaleSub[8];
    __shared__ float        minSub[8];
    __shared__ unsigned char qnib[Q4K_BLOCK_ELEMENTS];
    __shared__ float        dShared;
    __shared__ float        dminShared;
    __shared__ unsigned char scQ[8];
    __shared__ unsigned char mQ[8];

    const int blk = blockIdx.x;
    const int tid = threadIdx.x;

    const size_t srcIdx = static_cast<size_t>(blk) * Q4K_BLOCK_ELEMENTS + tid;
    xs[tid] = __bfloat162float(src[srcIdx]);
    __syncthreads();

    // --- Per-sub-block make_qkx2, one leader thread per 32-element sub-block.
    if (tid < 8) {
        const int base = tid * 32;
        float sumx2 = 0.0f;
        for (int l = 0; l < 32; ++l) sumx2 += xs[base + l] * xs[base + l];
        const float avx = sqrtf(sumx2 / 32.0f);
        float w[32], xv[32];
        for (int l = 0; l < 32; ++l) {
            xv[l] = xs[base + l];
            w[l]  = avx + fabsf(xv[l]);
        }
        float sc, mn;
        makeQkx2(xv, w, sc, mn);
        scaleSub[tid] = sc;
        minSub[tid]   = mn;
    }
    __syncthreads();

    // --- Super-block d / dmin + 6-bit sub-block scale/min, computed by tid 0.
    if (tid == 0) {
        float maxScale = 0.0f, maxMin = 0.0f;
        for (int j = 0; j < 8; ++j) {
            maxScale = fmaxf(maxScale, scaleSub[j]);
            maxMin   = fmaxf(maxMin,   minSub[j]);
        }
        const float d    = maxScale / 63.0f;
        const float dmin = maxMin   / 63.0f;
        dShared    = d;
        dminShared = dmin;
        const float invScale = (maxScale > 0.0f) ? 63.0f / maxScale : 0.0f;
        const float invMin   = (maxMin   > 0.0f) ? 63.0f / maxMin   : 0.0f;
        for (int j = 0; j < 8; ++j) {
            int s = static_cast<int>(nearbyintf(scaleSub[j] * invScale));
            int m = static_cast<int>(nearbyintf(minSub[j]   * invMin));
            scQ[j] = static_cast<unsigned char>(max(0, min(63, s)));
            mQ[j]  = static_cast<unsigned char>(max(0, min(63, m)));
        }

        unsigned char* block = dst + static_cast<size_t>(blk) * Q4K_BLOCK_BYTES;
        *reinterpret_cast<__half*>(block)     = __float2half(d);
        *reinterpret_cast<__half*>(block + 2) = __float2half(dmin);

        unsigned char* sc = block + 4;
        for (int j = 0; j < 4; ++j) {
            sc[j]     = scQ[j];
            sc[j + 4] = mQ[j];
        }
        for (int j = 0; j < 4; ++j) {
            sc[j]     |= static_cast<unsigned char>((scQ[j + 4] >> 4) << 6);
            sc[j + 4] |= static_cast<unsigned char>((mQ[j + 4]  >> 4) << 6);
            sc[j + 8]  = static_cast<unsigned char>((scQ[j + 4] & 0x0F)
                                                  | ((mQ[j + 4] & 0x0F) << 4));
        }
    }
    __syncthreads();

    // --- Every thread re-quantises its own element to a 4-bit nibble using the
    //     block-quantised scale/min (matches llama.cpp's final requant step).
    const int sub = tid >> 5;
    const float scaleEff = dShared    * static_cast<float>(scQ[sub]);
    const float minEff   = dminShared * static_cast<float>(mQ[sub]);
    int q = 0;
    if (scaleEff > 0.0f) {
        q = static_cast<int>(nearbyintf((xs[tid] + minEff) / scaleEff));
        q = max(0, min(15, q));
    }
    qnib[tid] = static_cast<unsigned char>(q);
    __syncthreads();

    // --- Combine nibble pairs into qs[128].
    if (tid < 128) {
        const int b    = tid;
        const int lowE = (b >> 5) * 64 + (b & 31);
        const int hiE  = lowE + 32;
        unsigned char* qs = dst + static_cast<size_t>(blk) * Q4K_BLOCK_BYTES + 16;
        qs[b] = static_cast<unsigned char>(qnib[lowE] | (qnib[hiE] << 4));
    }
}
