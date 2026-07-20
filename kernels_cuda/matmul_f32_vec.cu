// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
// Ported from kernels_hip/matmul_f32_vec.hip — Track 4 mechanical port, no functional change intended.

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Matrix-vector multiply with F32 weights (no quantisation).
//
//   Y[n] = sum_{k=0..K-1} X[k] * W[n, k]
//
//   X:  [K]     F32 dense vector (single token / M=1)
//   W:  [N, K]  F32 dense matrix, row-major
//   Y:  [N]     F32 dense vector
//
// Same warp layout as the K-quant sibling kernels (RDNA3 warpSize=32,
// LOCAL=128 = 4 warps × 32 lanes, one warp per output row,
// OUTPUTS_PER_GROUP=4). No dequant — the per-lane loop is a straight
// fp32 FMA over the tile stride.
//
// Motivation: Gemma 4 E4B keeps `inp_gate.weight` and `proj.weight` in
// F32 (~2.6 MiB each, 84 dispatches per decode-step). Without a native
// F32 matmul-vec these fell through to the CPU-fallback path which
// stages W/X D→H, computes on host, then H→D — roughly 2× the decode
// cost even though the tensors are tiny. This kernel closes the last
// non-K-quant fallback for Gemma 4.
//
// Launch:
//   dim3 grid ( ceil(N / OUTPUTS_PER_GROUP), 1, 1 )
//   dim3 block( MATMUL_F32_LOCAL, 1, 1 )

#include <cuda_runtime.h>

#ifndef MATMUL_F32_LOCAL
#define MATMUL_F32_LOCAL 128
#endif

#define MATMUL_F32_WARPS             (MATMUL_F32_LOCAL / 32)
#define MATMUL_F32_OUTPUTS_PER_GROUP MATMUL_F32_WARPS

#define X_TILE_ELEMENTS 1024

namespace {

__device__ __forceinline__ float warpReduceSum(float v) {
    v += __shfl_down(v, 16);
    v += __shfl_down(v,  8);
    v += __shfl_down(v,  4);
    v += __shfl_down(v,  2);
    v += __shfl_down(v,  1);
    return v;
}

} // namespace

extern "C" __global__ __launch_bounds__(MATMUL_F32_LOCAL)
void matmul_f32_vec(
    const float* __restrict__ X,
    const float* __restrict__ W,
          float* __restrict__ Y,
    const int                 K,
    const int                 N)
{
    __shared__ float xTile[X_TILE_ELEMENTS];

    const int wg      = blockIdx.x;
    const int tid     = threadIdx.x;
    const int lsize   = blockDim.x;
    const int warpId  = tid / 32;
    const int laneId  = tid % 32;
    const int n       = wg * MATMUL_F32_OUTPUTS_PER_GROUP + warpId;
    const bool active = (n < N);

    float sum = 0.0f;

    for (int tile = 0; tile < K; tile += X_TILE_ELEMENTS) {
        const int tileK = min(X_TILE_ELEMENTS, K - tile);

        for (int i = tid; i < tileK; i += lsize) {
            xTile[i] = X[tile + i];
        }
        __syncthreads();

        if (active) {
            const float* __restrict__ wRow =
                W + static_cast<size_t>(n) * static_cast<size_t>(K) + tile;

            // Stride the tile over the 32 lanes of this warp.
            for (int i = laneId; i < tileK; i += 32) {
                sum = __fmaf_rn(xTile[i], wRow[i], sum);
            }
        }

        __syncthreads();
    }

    sum = warpReduceSum(sum);

    if (active && laneId == 0) {
        Y[n] = sum;
    }
}