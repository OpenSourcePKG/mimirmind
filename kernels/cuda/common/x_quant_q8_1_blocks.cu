// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Per-32-element-block symmetric int8 activation quantisation — the q8_1
// block granularity llama.cpp's ggml-cuda MMVQ kernels expect, as opposed
// to x_quant_i8's single scale for the whole row. Q6_K/Q8_0 weight blocks
// are themselves scaled per 32 (Q8_0) or per 16-within-256 (Q6_K, but its
// MMVQ dot still consumes the activation in 32-wide q8_1 chunks); matching
// that granularity on the activation side avoids one row-wide outlier
// crushing precision for the rest of the row.
//
//   scale[b]     = max_k(|X[b*32+k]|) / 127
//   Y[b*32+k]    = round(X[b*32+k] / scale[b])  clamped to [-127, 127]
//
//   X:     [K]        F32   (K must be a multiple of 32)
//   Y:     [K]         int8
//   scale: [K/32]      F32
//
// One warp (32 lanes) per block: each lane owns exactly one element,
// so the row max is a plain warp32 shuffle-xor reduction — no shared
// memory needed. `roundf` (round-to-nearest, ties-away-from-zero)
// matches ggml's quantize_row_q8_1_reference / x_quant_i8's convention.
//
// The reference-quality "s = d * sum(qs)" companion field llama.cpp's
// block_q8_1 carries is deliberately NOT produced here: it exists only to
// cheapen the offset-correction term of asymmetric formats (Q4_K/Q5_K).
// Q6_K and Q8_0 have no such term (see vec_dot_q6_K_q8_1_impl_mmvq /
// vec_dot_q8_0_q8_1_impl upstream — both consume only the block scale),
// so this quantiser only needs to emit scale+quants.

#include <cuda_runtime.h>

#define X_QUANT_Q81_BLOCK 32

static __device__ __forceinline__ float warp32_reduce_max(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 16, 32));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 8, 32));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 4, 32));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2, 32));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1, 32));
    return v;
}

extern "C" __global__ __launch_bounds__(X_QUANT_Q81_BLOCK)
void x_quant_q8_1_blocks(
    const float* __restrict__       X,     // [K]
          signed char* __restrict__ Y,     // [K]
          float* __restrict__       scale, // [K/32]
    const int                       K)
{
    const int b    = blockIdx.x;              // block index (0..K/32-1)
    const int lane = threadIdx.x;              // 0..31, one element each
    const int base = b * X_QUANT_Q81_BLOCK;

    const float x    = X[base + lane];
    const float amax = warp32_reduce_max(fabsf(x));

    const float s    = amax * (1.0f / 127.0f);
    const float invS = (amax > 0.0f) ? (127.0f / amax) : 0.0f;

    if (lane == 0) {
        scale[b] = s;
    }

    const float q  = roundf(x * invS);
    const float qc = fminf(fmaxf(q, -127.0f), 127.0f);
    Y[base + lane] = static_cast<signed char>(static_cast<int>(qc));
}
