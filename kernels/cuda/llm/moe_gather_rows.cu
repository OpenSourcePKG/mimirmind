// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Device-side MoE row gather (M-Cuda.MoeGroup, Sub-Step B).
//
// Materialises the per-expert-grouped activation buffer the grouped GEMM
// consumes: for each compacted row r produced by moe_group_build, copy the
// source token's activation vector into row r.
//
//   xCompact[r, :] = x[rowSrcTok[r], :]        r in [0, R), width d_model
//
// A token that routes to K experts is copied K times (into K different
// compacted rows, one per expert group) — that is intended: each expert
// then GEMMs over its own contiguous block of rows, reading its weight once.
//
// Launch: grid.x = R (one block per compacted row), block.x = min(d_model,
// 256) threads striding the row. No shared memory.

#include <cuda_runtime.h>

extern "C" __global__ void moe_gather_rows(
    const float* __restrict__ x,          // [T, d_model]  source activations
    const int*   __restrict__ rowSrcTok,  // [R]           row -> source token
          float* __restrict__ xCompact,   // [R, d_model]  out
    const int                 d_model,
    const int                 R)
{
    const int r = static_cast<int>(blockIdx.x);
    if (r >= R) {
        return;
    }
    const int t = rowSrcTok[r];
    const float* __restrict__ src = x        + static_cast<size_t>(t) * d_model;
    float*       __restrict__ dst = xCompact + static_cast<size_t>(r) * d_model;
    for (int j = static_cast<int>(threadIdx.x); j < d_model;
         j += static_cast<int>(blockDim.x)) {
        dst[j] = src[j];
    }
}
