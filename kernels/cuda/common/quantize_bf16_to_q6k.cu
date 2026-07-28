// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// BF16 -> Q6_K weight quantizer (load-time, one-shot).
//
//   src: [rows, K]        BF16 dense weights, row-major (K % 256 == 0)
//   dst: [rows, K/256]     Q6_K super-blocks (210 bytes each), row-major
//
// Re-quantises the materialised BF16 MoE down-expert banks to 6-bit Q6_K so
// the memory-bound decode reads ~2.4x less down-expert traffic. Q6_K is the
// higher-precision K-quant (used for the sensitive down projection), matching
// the format llama.cpp uses for these experts. Feeds the existing
// moe_down_fused_k_q6k kernels unchanged.
//
// Q6_K super-block layout (bit-identical to llama.cpp / Q6K.cpp:48-98):
//   u8   ql[128] (bytes 0..127)   low 4 bits of each 6-bit quant
//   u8   qh[64]  (bytes 128..191) high 2 bits, 4 quants per byte
//   i8   sc[16]  (bytes 192..207) per-16-element sub-block scales
//   fp16 d       (bytes 208..209) super-scale
//   element[e in sub-block e/16] = d * sc[e/16] * (q - 32)   (q in [0,63])
//
// One thread block per super-block, blockDim = 256 (one thread per element).
// Launch: grid( totalElems/256, 1, 1 ), block( 256, 1, 1 ).

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#ifndef QUANTIZE_Q6K_LOCAL
#define QUANTIZE_Q6K_LOCAL 256
#endif

#define Q6K_BLOCK_ELEMENTS 256
#define Q6K_BLOCK_BYTES    210

namespace {

// llama.cpp make_qx_quants (nmax=32, rmse_type=1, weights = x*x), reduced to
// return the per-16-element sub-block scale. Symmetric (signed) 6-bit quant.
__device__ float makeQxQuants(const float* __restrict__ x) {
    float amax = 0.0f, mx = 0.0f;
    for (int i = 0; i < 16; ++i) {
        const float ax = fabsf(x[i]);
        if (ax > amax) { amax = ax; mx = x[i]; }
    }
    if (amax < 1e-30f) return 0.0f;

    float iscale = -32.0f / mx;
    float sumlx = 0.0f, suml2 = 0.0f;
    for (int i = 0; i < 16; ++i) {
        int l = static_cast<int>(nearbyintf(iscale * x[i]));
        l = max(-32, min(31, l));
        const float w = x[i] * x[i];
        sumlx += w * x[i] * static_cast<float>(l);
        suml2 += w * static_cast<float>(l) * static_cast<float>(l);
    }
    float scale = (suml2 > 0.0f) ? sumlx / suml2 : 0.0f;
    float best = scale * sumlx;
    for (int is = -9; is <= 9; ++is) {
        if (is == 0) continue;
        const float isc = -(32.0f + 0.1f * static_cast<float>(is)) / mx;
        float slx = 0.0f, sl2 = 0.0f;
        for (int i = 0; i < 16; ++i) {
            int l = static_cast<int>(nearbyintf(isc * x[i]));
            l = max(-32, min(31, l));
            const float w = x[i] * x[i];
            slx += w * x[i] * static_cast<float>(l);
            sl2 += w * static_cast<float>(l) * static_cast<float>(l);
        }
        if (sl2 > 0.0f && slx * slx > best * sl2) {
            scale = slx / sl2;
            best  = scale * slx;
        }
    }
    return scale;
}

} // namespace

extern "C" __global__ __launch_bounds__(QUANTIZE_Q6K_LOCAL)
void quantize_bf16_to_q6k(
    const __nv_bfloat16* __restrict__ src,   // [totalElems]
          unsigned char* __restrict__ dst)   // [totalBlocks * 210]
{
    __shared__ float         xs[Q6K_BLOCK_ELEMENTS];
    __shared__ float         scaleSub[16];
    __shared__ signed char   scInt8[16];
    __shared__ float         dShared;
    __shared__ unsigned char L[Q6K_BLOCK_ELEMENTS];   // 6-bit quants (q+32)

    const int blk = blockIdx.x;
    const int tid = threadIdx.x;

    const size_t srcIdx = static_cast<size_t>(blk) * Q6K_BLOCK_ELEMENTS + tid;
    xs[tid] = __bfloat162float(src[srcIdx]);
    __syncthreads();

    // --- Per-16-element sub-block scale, one leader per sub-block (16 total).
    if (tid < 16) {
        float xv[16];
        for (int l = 0; l < 16; ++l) xv[l] = xs[tid * 16 + l];
        scaleSub[tid] = makeQxQuants(xv);
    }
    __syncthreads();

    // --- Super-scale d + int8 sub-block scales, computed by tid 0.
    if (tid == 0) {
        float maxScale = 0.0f, maxAbs = 0.0f;
        for (int j = 0; j < 16; ++j) {
            const float a = fabsf(scaleSub[j]);
            if (a > maxAbs) { maxAbs = a; maxScale = scaleSub[j]; }
        }
        float d = 0.0f;
        if (maxAbs >= 1e-30f) {
            const float iscale = -128.0f / maxScale;
            d = 1.0f / iscale;                       // = -maxScale/128
            for (int j = 0; j < 16; ++j) {
                int s = static_cast<int>(nearbyintf(iscale * scaleSub[j]));
                s = max(-128, min(127, s));
                scInt8[j] = static_cast<signed char>(s);
            }
        } else {
            for (int j = 0; j < 16; ++j) scInt8[j] = 0;
        }
        dShared = d;

        unsigned char* block = dst + static_cast<size_t>(blk) * Q6K_BLOCK_BYTES;
        // sc[16] at bytes 192..207, d (fp16) at 208..209.
        signed char* scDst = reinterpret_cast<signed char*>(block + 192);
        for (int j = 0; j < 16; ++j) scDst[j] = scInt8[j];
        *reinterpret_cast<__half*>(block + 208) = __float2half(d);
    }
    __syncthreads();

    // --- Every thread requantises its own element to a 6-bit value (q+32).
    const int sub = tid >> 4;                        // 0..15
    const float effScale = dShared * static_cast<float>(scInt8[sub]);
    int q = 0;
    if (effScale != 0.0f) {
        q = static_cast<int>(nearbyintf(xs[tid] / effScale));
        q = max(-32, min(31, q));
    }
    L[tid] = static_cast<unsigned char>(q + 32);     // [0,63]
    __syncthreads();

    // --- Pack L into ql[128] (low 4 bits) + qh[64] (high 2 bits).
    // Element layout (inverse of Q6K.cpp dequant), per 128-element half:
    //   ql[half*64 + l]      low nib = elem[half*128+l], high nib = elem[+64]
    //   ql[half*64 + l+32]   low nib = elem[half*128+l+32], high nib = elem[+96]
    //   qh[half*32 + l]      bits 0-1/2-3/4-5/6-7 = elem[+0]/[+32]/[+64]/[+96]
    unsigned char* block = dst + static_cast<size_t>(blk) * Q6K_BLOCK_BYTES;
    if (tid < 128) {
        const int bi     = tid;                      // ql byte index 0..127
        const int half   = bi >> 6;                  // 0..1
        const int within = bi & 63;                  // 0..63
        const int l      = within & 31;              // 0..31
        const int e0     = half * 128 + l + (within < 32 ? 0  : 32);
        const int e1     = half * 128 + l + (within < 32 ? 64 : 96);
        block[bi] = static_cast<unsigned char>((L[e0] & 0x0F) | ((L[e1] & 0x0F) << 4));
    }
    if (tid < 64) {
        const int qi   = tid;                        // qh byte index 0..63
        const int half = qi >> 5;                    // 0..1
        const int l    = qi & 31;                    // 0..31
        const int base = half * 128 + l;
        const unsigned h1 = (L[base +  0] >> 4) & 0x3;
        const unsigned h2 = (L[base + 32] >> 4) & 0x3;
        const unsigned h3 = (L[base + 64] >> 4) & 0x3;
        const unsigned h4 = (L[base + 96] >> 4) & 0x3;
        block[128 + qi] = static_cast<unsigned char>(h1 | (h2 << 2) | (h3 << 4) | (h4 << 6));
    }
}
