// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// 7.11 Rune projection / conditional steer (Futhark steer_project / steer_if).
//
// rune_row_dot:
//   out[r] = <x[r,:], w>            (normalize == 0)
//   out[r] = <x[r,:], w> / ||x[r,:]|| (normalize != 0; 0 for a zero row)
//   Launch: grid (rows, 1, 1), block (RUNE_DOT_LOCAL, 1, 1)
//
// rune_steer_rows:
//   x[r,k] += alpha * g[r] * c[r] * v[k]
//   g[r] = gate ? (gate[r] > threshold) : 1, c[r] = coef ? coef[r] : 1
//   Launch: grid (ceil(rows*d / RUNE_STEER_LOCAL), 1, 1), block (RUNE_STEER_LOCAL, 1, 1)
//
// Rows are independent, so a per-row gate is valid on every batched / ragged
// layout. No host sync, graph-capturable.

#include <cuda_runtime.h>

#ifndef RUNE_DOT_LOCAL
#define RUNE_DOT_LOCAL 256
#endif
#ifndef RUNE_STEER_LOCAL
#define RUNE_STEER_LOCAL 256
#endif

extern "C" __global__ __launch_bounds__(RUNE_DOT_LOCAL)
void rune_row_dot(
    const float* __restrict__ x,     // [rows, d]
    const float* __restrict__ w,     // [d]
          float* __restrict__ out,   // [rows]
    const int                 d,
    const int                 normalize)
{
    __shared__ float sDot[RUNE_DOT_LOCAL];
    __shared__ float sSq [RUNE_DOT_LOCAL];
    const int r = blockIdx.x;
    const float* row = x + static_cast<long long>(r) * d;
    float dot = 0.0f;
    float sq  = 0.0f;
    for (int k = threadIdx.x; k < d; k += RUNE_DOT_LOCAL) {
        const float a = row[k];
        dot = fmaf(a, w[k], dot);
        sq  = fmaf(a, a, sq);
    }
    sDot[threadIdx.x] = dot;
    sSq [threadIdx.x] = sq;
    __syncthreads();
    for (int s = RUNE_DOT_LOCAL / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sDot[threadIdx.x] += sDot[threadIdx.x + s];
            sSq [threadIdx.x] += sSq [threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float v = sDot[0];
        if (normalize != 0) {
            v = sSq[0] > 0.0f ? v * rsqrtf(sSq[0]) : 0.0f;
        }
        out[r] = v;
    }
}

extern "C" __global__ __launch_bounds__(RUNE_STEER_LOCAL)
void rune_steer_rows(
          float* __restrict__ x,       // [rows, d], in + out
    const float* __restrict__ v,       // [d]
    const float* __restrict__ gate,    // [rows] or nullptr
    const float* __restrict__ coef,    // [rows] or nullptr
    const float               alpha,
    const float               threshold,
    const int                 rows,
    const int                 d)
{
    const long long gid   = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long total = static_cast<long long>(rows) * d;
    if (gid >= total) {
        return;
    }
    const int r = static_cast<int>(gid / d);
    const int k = static_cast<int>(gid - static_cast<long long>(r) * d);
    if (gate != nullptr && !(gate[r] > threshold)) {
        return;
    }
    const float s = alpha * (coef != nullptr ? coef[r] : 1.0f);
    x[gid] = fmaf(s, v[k], x[gid]);
}
