// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Fused Q + K + V RMSNorm — one dispatch instead of three.
//
// HIP port of kernels/rmsnorm_qkv.cl. Q and K normalize with
// per-head_dim weights (plain rmsnorm semantics); V normalizes without
// a learned weight (rmsnorm_no_weight semantics). All three streams
// share head_dim = K and are routed by workgroup index:
//
//   groups [0,                            qRows)              → Q rows
//   groups [qRows,                        qRows + kRows)      → K rows
//   groups [qRows + kRows,                qRows + 2*kRows)    → V rows
//
// Total grid = qRows + 2*kRows = T*nHeads + 2*T*nKvHeads. Each
// workgroup runs the same tree reduction as the vanilla rmsnorm.
//
// `curLenPtr` is a device pointer to a single int (the RoPE-style
// runtime-int slot). K and V rows are written at
// `curLen * kvDim` bytes offset into their respective cache buffers;
// Q rows go to the scratch buffer at their natural offset.
//
// Launch:
//   dim3 grid (qRows + 2*kRows, 1, 1),
//   dim3 block(RMSNORM_QKV_LOCAL_SIZE, 1, 1)

#include <hip/hip_runtime.h>

#ifndef RMSNORM_QKV_LOCAL_SIZE
#define RMSNORM_QKV_LOCAL_SIZE 128
#endif

extern "C" __global__ __launch_bounds__(RMSNORM_QKV_LOCAL_SIZE)
void rmsnorm_qkv(
    const float* __restrict__ q_x,        // [qRows, K]
    const float* __restrict__ q_w,        // [K]
          float* __restrict__ q_y,        // in-place OK (== q_x)
    const float* __restrict__ k_x,        // K cache base for this layer
    const float* __restrict__ k_w,        // [K]
          float* __restrict__ k_y,        // K cache base for this layer
    const float* __restrict__ v_x,        // V cache base for this layer
          float* __restrict__ v_y,        // V cache base for this layer
    const int                 qRows,      // T * nHeads
    const int                 kRows,      // T * nKvHeads   (K and V share this)
    const int                 K,          // head_dim
    const float               eps,
    const int* __restrict__   curLenPtr,  // device int slot (RoPE-style)
    const int                 kvDim)      // nKvHeads * head_dim
{
    __shared__ float scratch[RMSNORM_QKV_LOCAL_SIZE];

    const int gid   = blockIdx.x;
    const int tid   = threadIdx.x;
    const int lsize = blockDim.x;

    // Route this workgroup to Q / K / V. The three branches diverge
    // across workgroups, not threads — no lane divergence.
    const float* __restrict__ xr;
    const float* __restrict__ wr;
          float* __restrict__ yr;
    const size_t kvBase = static_cast<size_t>(curLenPtr[0])
                        * static_cast<size_t>(kvDim);
    if (gid < qRows) {
        const int row = gid;
        xr = q_x + static_cast<size_t>(row) * static_cast<size_t>(K);
        wr = q_w;
        yr = q_y + static_cast<size_t>(row) * static_cast<size_t>(K);
    } else if (gid < qRows + kRows) {
        const int row = gid - qRows;
        xr = k_x + kvBase + static_cast<size_t>(row) * static_cast<size_t>(K);
        wr = k_w;
        yr = k_y + kvBase + static_cast<size_t>(row) * static_cast<size_t>(K);
    } else {
        const int row = gid - qRows - kRows;
        xr = v_x + kvBase + static_cast<size_t>(row) * static_cast<size_t>(K);
        wr = nullptr;   // V: no learned weight
        yr = v_y + kvBase + static_cast<size_t>(row) * static_cast<size_t>(K);
    }

    // Per-thread partial sum of squares.
    float acc = 0.0f;
    for (int k = tid; k < K; k += lsize) {
        const float v = xr[k];
        acc = __fmaf_rn(v, v, acc);
    }
    scratch[tid] = acc;
    __syncthreads();

    // Power-of-two tree reduction in shared memory.
    for (int stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }

    const float mean   = scratch[0] / static_cast<float>(K);
    const float invRms = rsqrtf(mean + eps);

    if (wr != nullptr) {
        for (int k = tid; k < K; k += lsize) {
            yr[k] = xr[k] * wr[k] * invRms;
        }
    } else {
        for (int k = tid; k < K; k += lsize) {
            yr[k] = xr[k] * invRms;
        }
    }
}