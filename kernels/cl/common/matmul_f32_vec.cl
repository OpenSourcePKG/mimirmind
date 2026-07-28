// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

// Matrix-vector multiply with dense F32 weights (no dequant).
//
//   Y[n] = sum_{k=0..K-1} X[k] * W[n, k]
//
//   X:  [K]     F32 dense vector (single token / M=1 row)
//   W:  [N, K]  F32 row-major
//   Y:  [N]     F32 dense vector
//
// Same launch geometry + subgroup-reduce structure as matmul_q8_0_vec.cl,
// only the per-element load is a plain F32 read instead of a block dequant.
//
// Motivation (M-CLR.MoE Increment 3): F32 weights had no GPU matmul kernel,
// so GpuMatmul::matmulAsync fell back to a host `compute::matmul`. Inside a
// recorded command list that host call runs ONCE at record time — and its
// flush() is a no-op while recording, so it reads a STALE input buffer (the
// producing kernels are only appended, not yet executed) and bakes a wrong
// result that every replay reuses. The Gemma 4 MoE router
// (`ffn_gate_inp.weight`, F32) hit exactly this, corrupting expert routing
// under Command-List-Replay from the first block. Giving F32 a real GPU
// kernel keeps the router matmul on the recorded queue so it re-materialises
// from fresh inputs on every replay.
//
// Launch geometry (mirrors matmul_q8_0_vec.cl):
//   local_size_x          = MATMUL_F32_LOCAL   (64)
//   sub_group_size        = MATMUL_F32_SG      (16)
//   outputs per workgroup = MATMUL_F32_LOCAL / MATMUL_F32_SG  (= 4)
//   global_size_x         = ceil(N / 4) * 64

#pragma OPENCL EXTENSION cl_intel_subgroups : enable

#ifndef MATMUL_F32_LOCAL
#define MATMUL_F32_LOCAL 64
#endif

#ifndef MATMUL_F32_SG
#define MATMUL_F32_SG 16
#endif

#define MATMUL_F32_OUTPUTS_PER_GROUP (MATMUL_F32_LOCAL / MATMUL_F32_SG)

// 1024 elements = 4 KiB SLM per workgroup — matches matmul_q8_0_vec.cl.
#define X_TILE_ELEMENTS 1024

__attribute__((reqd_work_group_size(MATMUL_F32_LOCAL, 1, 1)))
__attribute__((intel_reqd_sub_group_size(MATMUL_F32_SG)))
__kernel void matmul_f32_vec(
    __global const float* X,
    __global const float* W,
    __global       float* Y,
    const int             K,
    const int             N)
{
    __local float xTile[X_TILE_ELEMENTS];

    const int  wg      = (int)get_group_id(0);
    const int  sgInWg  = (int)get_sub_group_id();           // 0..3
    const int  sgLocal = (int)get_sub_group_local_id();     // 0..15
    const int  tid     = (int)get_local_id(0);
    const int  lsize   = (int)get_local_size(0);
    const int  n       = wg * MATMUL_F32_OUTPUTS_PER_GROUP + sgInWg;
    const bool active  = (n < N);

    float sum = 0.0f;

    for (int tile = 0; tile < K; tile += X_TILE_ELEMENTS) {
        const int tileK = min(X_TILE_ELEMENTS, K - tile);
        for (int i = tid; i < tileK; i += lsize) {
            xTile[i] = X[tile + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (active) {
            __global const float* row = W + (size_t)n * (size_t)K + tile;
            // 16 sub-group lanes stride over this tile's K span.
            for (int l = sgLocal; l < tileK; l += MATMUL_F32_SG) {
                sum = mad(xTile[l], row[l], sum);
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    sum = sub_group_reduce_add(sum);

    if (active && sgLocal == 0) {
        Y[n] = sum;
    }
}
