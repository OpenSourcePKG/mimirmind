// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

// Per-row symmetric int8 quantisation of an activation matrix.
//
//   scale[m] = max_k(|X[m, k]|) / 127
//   Y[m, k]  = round(X[m, k] / scale[m])  clamped to [-127, 127]
//
//   X:     [M, K]  F32 dense (M=1 in decode, M=T in prefill)
//   Y:     [M, K]  int8
//   scale: [M]     F32
//
// Feeds the DP4A Q8_0 matmul (M8.H.1): the matmul reconstructs
//   Y_out[m, n] = block_scale × scale[m] × sum(dot(char4, char4))
// so the per-row scale never enters the inner loop.
//
// Zero-input row (max=0): scale=0 stored, all quants=0. The DP4A
// matmul multiplies by scale at the end and produces exactly 0,
// which is the correct answer for a zero input row.
//
// The [-127, 127] clamp (rather than [-128, 127]) mirrors ggml's
// quantize_row_q8_0_ref so weights and activations share the same
// symmetric range. Rounding is round-to-nearest, ties away from
// zero — matches OpenCL round().
//
// Launch geometry:
//   local_size_x  = X_QUANT_I8_LOCAL   (128)
//   global_size_x = M × X_QUANT_I8_LOCAL   (one workgroup per row)

#ifndef X_QUANT_I8_LOCAL
#define X_QUANT_I8_LOCAL 128
#endif

__attribute__((reqd_work_group_size(X_QUANT_I8_LOCAL, 1, 1)))
__kernel void x_quant_i8(
    __global const float* X,
    __global       char*  Y,
    __global       float* scale,
    const int             K)
{
    __local float scratch[X_QUANT_I8_LOCAL];

    const int m     = (int)get_group_id(0);
    const int tid   = (int)get_local_id(0);
    const int lsize = (int)get_local_size(0);

    __global const float* xr = X + (size_t)m * (size_t)K;
    __global       char*  yr = Y + (size_t)m * (size_t)K;

    // Per-thread max(|x|) over strided slice.
    float maxAbs = 0.0f;
    for (int k = tid; k < K; k += lsize) {
        maxAbs = fmax(maxAbs, fabs(xr[k]));
    }
    scratch[tid] = maxAbs;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Power-of-two tree reduction over max.
    for (int stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = fmax(scratch[tid], scratch[tid + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const float amax = scratch[0];
    const float s    = amax * (1.0f / 127.0f);
    // invS is 0 for an all-zero row — the multiply below produces 0
    // quants, and scale[m]=0 will zero the matmul output for row m.
    const float invS = (amax > 0.0f) ? (127.0f / amax) : 0.0f;

    if (tid == 0) {
        scale[m] = s;
    }

    for (int k = tid; k < K; k += lsize) {
        const float q  = round(xr[k] * invS);
        const float qc = clamp(q, -127.0f, 127.0f);
        yr[k] = (char)((int)qc);
    }
}
