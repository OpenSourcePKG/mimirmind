// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Device-side MoE expert-output scatter (M-Cuda.MoeGroup, Sub-Step C).
//
// Un-permutes the grouped-GEMM output back to token order and applies the
// router weights, producing the routed-expert contribution to the MoE
// accumulator:
//
//   accum[t, :] = sum_{k=0..K-1} kw[t*K+k] * y[asnToRow[t*K+k], :]
//
// where y[R, d_model] holds each compacted expert-row's down-projection and
// asnToRow (from moe_group_build) maps assignment (t,k) -> its compacted row.
//
// Token-major by construction: one block owns token t and folds its K
// contributions in a FIXED k order, so the result is deterministic and
// run-to-run reproducible — no atomics, no cross-block contention (each
// accum row is written by exactly one block). This overwrites accum with the
// routed-expert sum (it does NOT read-modify-write); the always-on shared
// expert is added afterwards by the caller, exactly as the fused-K path does.
//
// A dropped assignment (asnToRow == -1, malformed routing) contributes 0.
//
// Launch: grid.x = T (one block per token), block.x = min(d_model, 256).

#include <cuda_runtime.h>

extern "C" __global__ void moe_scatter_expert_out(
    const float* __restrict__ y,          // [R, d_model]  per-expert-row output
    const int*   __restrict__ asnToRow,   // [T*K]         assignment -> row
    const float* __restrict__ kw,         // [T*K]         router weight (post-wScale)
          float* __restrict__ accum,      // [T, d_model]  out (overwritten)
    const int                 d_model,
    const int                 T,
    const int                 K)
{
    const int t = static_cast<int>(blockIdx.x);
    if (t >= T) {
        return;
    }
    float* __restrict__ dst = accum + static_cast<size_t>(t) * d_model;
    const int base = t * K;
    for (int j = static_cast<int>(threadIdx.x); j < d_model;
         j += static_cast<int>(blockDim.x)) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            const int row = asnToRow[base + k];
            if (row < 0) {
                continue;
            }
            acc += kw[base + k] * y[static_cast<size_t>(row) * d_model + j];
        }
        dst[j] = acc;
    }
}
