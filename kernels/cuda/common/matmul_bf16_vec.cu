// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Matrix-vector multiply with BF16 weights (no quantisation).
//
//   Y[n] = sum_{k=0..K-1} X[k] * bf16_to_f32(W[n, k])
//
//   X:  [K]     F32 dense vector (single token / M=1)
//   W:  [N, K]  BF16 dense matrix, row-major (2 bytes / element)
//   Y:  [N]     F32 dense vector
//
// Mirrors matmul_f32_vec.cu (same warp layout: LOCAL=128 = 4 warps × 32
// lanes, one warp per output row, OUTPUTS_PER_GROUP=4) but reads BF16
// weights and widens each to fp32 in the per-lane FMA. Reading 2-byte
// BF16 halves the weight memory traffic vs the F32 kernel.
//
// Motivation: the NVFP4 checkpoints materialise their (unquantised /
// dequantised-NVFP4) weights to BF16 — notably every MoE expert bank.
// Without a native BF16 matmul-vec these fall through to the CPU-fallback
// in GpuMatmul::matmulAsync (stage W/X D->H, matmul on host, H->D), which
// makes the whole qwen35moe decode host-bound (~seconds per token). This
// kernel keeps those matmuls on the GPU.
//
// Launch:
//   dim3 grid ( ceil(N / OUTPUTS_PER_GROUP), 1, 1 )
//   dim3 block( MATMUL_BF16_LOCAL, 1, 1 )

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#ifndef MATMUL_BF16_LOCAL
#define MATMUL_BF16_LOCAL 128
#endif

#define MATMUL_BF16_WARPS             (MATMUL_BF16_LOCAL / 32)
#define MATMUL_BF16_OUTPUTS_PER_GROUP MATMUL_BF16_WARPS

#define X_TILE_ELEMENTS 1024

namespace {

__device__ __forceinline__ float warpReduceSum(float v) {
    v += __shfl_down_sync(0xffffffffu, v, 16);
    v += __shfl_down_sync(0xffffffffu, v,  8);
    v += __shfl_down_sync(0xffffffffu, v,  4);
    v += __shfl_down_sync(0xffffffffu, v,  2);
    v += __shfl_down_sync(0xffffffffu, v,  1);
    return v;
}

} // namespace

extern "C" __global__ __launch_bounds__(MATMUL_BF16_LOCAL)
void matmul_bf16_vec(
    const float*          __restrict__ X,
    const __nv_bfloat16*  __restrict__ W,
          float*          __restrict__ Y,
    const int                          K,
    const int                          N)
{
    __shared__ float xTile[X_TILE_ELEMENTS];

    const int wg      = blockIdx.x;
    const int tid     = threadIdx.x;
    const int lsize   = blockDim.x;
    const int warpId  = tid / 32;
    const int laneId  = tid % 32;
    const int n       = wg * MATMUL_BF16_OUTPUTS_PER_GROUP + warpId;
    const bool active = (n < N);

    float sum = 0.0f;

    for (int tile = 0; tile < K; tile += X_TILE_ELEMENTS) {
        const int tileK = min(X_TILE_ELEMENTS, K - tile);

        for (int i = tid; i < tileK; i += lsize) {
            xTile[i] = X[tile + i];
        }
        __syncthreads();

        if (active) {
            const __nv_bfloat16* __restrict__ wRow =
                W + static_cast<size_t>(n) * static_cast<size_t>(K) + tile;

            // Stride the tile over the 32 lanes of this warp.
            for (int i = laneId; i < tileK; i += 32) {
                sum = __fmaf_rn(xTile[i], __bfloat162float(wRow[i]), sum);
            }
        }

        __syncthreads();
    }

    sum = warpReduceSum(sum);

    if (active && laneId == 0) {
        Y[n] = sum;
    }
}
