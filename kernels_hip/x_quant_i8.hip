// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Per-row symmetric int8 quantisation of an activation matrix.
//
//   scale[m] = max_k(|X[m, k]|) / 127
//   Y[m, k]  = round(X[m, k] / scale[m])  clamped to [-127, 127]
//
//   X:     [M, K]  F32 dense (M=1 in decode, M=T in prefill)
//   Y:     [M, K]  int8
//   scale: [M]     F32
//
// HIP port of kernels/x_quant_i8.cl. Same launch shape as rmsnorm:
// one workgroup per row, tree reduction over max(|x|) in LDS, then
// per-thread strided quantise pass. Uses `roundf` (round-to-nearest,
// ties-away-from-zero) to match ggml's `quantize_row_q8_0_ref` — the
// same choice that made the L0 kernel byte-parity with ggml (see
// lesson `roundf` = ggml Q8_0 in the HIP-port-progress note).
//
// Zero-input row (max=0): scale=0 stored, all quants=0. Matches L0
// semantics — the DP4A matmul multiplies by scale at the end so
// row m's output is exactly 0, which is correct.
//
// Launch:
//   dim3 grid (M, 1, 1),
//   dim3 block(X_QUANT_I8_LOCAL, 1, 1)
//
// Default 128 == 4 waves of 32 on gfx1101, matches the L0 default.

#include <hip/hip_runtime.h>

#ifndef X_QUANT_I8_LOCAL
#define X_QUANT_I8_LOCAL 128
#endif

extern "C" __global__ __launch_bounds__(X_QUANT_I8_LOCAL)
void x_quant_i8(
    const float* __restrict__       X,     // [M, K]
          signed char* __restrict__ Y,     // [M, K]
          float* __restrict__       scale, // [M]
    const int                       K)
{
    __shared__ float scratch[X_QUANT_I8_LOCAL];

    const int m     = blockIdx.x;
    const int tid   = threadIdx.x;
    const int lsize = blockDim.x;

    const float*       __restrict__ xr = X + static_cast<size_t>(m) * static_cast<size_t>(K);
          signed char* __restrict__ yr = Y + static_cast<size_t>(m) * static_cast<size_t>(K);

    // Per-thread max(|x|) over strided slice.
    float maxAbs = 0.0f;
    for (int k = tid; k < K; k += lsize) {
        maxAbs = fmaxf(maxAbs, fabsf(xr[k]));
    }
    scratch[tid] = maxAbs;
    __syncthreads();

    // Power-of-two tree reduction over max.
    for (int stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
        }
        __syncthreads();
    }

    const float amax = scratch[0];
    const float s    = amax * (1.0f / 127.0f);
    // invS is 0 for an all-zero row — the multiply below produces 0
    // quants, and scale[m]=0 zeroes the matmul output for row m.
    const float invS = (amax > 0.0f) ? (127.0f / amax) : 0.0f;

    if (tid == 0) {
        scale[m] = s;
    }

    for (int k = tid; k < K; k += lsize) {
        // `roundf` = round-to-nearest, ties-away-from-zero — same as
        // OpenCL `round()`, matches ggml's quantize_row_q8_0_ref.
        // Byte-parity with the L0 kernel + with llama.cpp reference
        // for Q8_0-encoded activations.
        const float q  = roundf(xr[k] * invS);
        const float qc = fminf(fmaxf(q, -127.0f), 127.0f);
        yr[k] = static_cast<signed char>(static_cast<int>(qc));
    }
}