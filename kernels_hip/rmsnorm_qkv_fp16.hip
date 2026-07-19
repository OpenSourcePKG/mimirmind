// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// FP16-KV variant of rmsnorm_qkv.hip. Q stays fp32 (workspace); K and
// V live in the fp16 KV cache. Weights (`k_w`) stay fp32. The sum of
// squares and the invRms scalar are computed in fp32 regardless of
// storage dtype (__half2float promotes each read to fp32 before the
// mad). Writes use __float2half so the fp16 cache gets round-trip
// normalised values.
//
// The Q, K and V branches diverge across workgroups only — every
// thread inside a workgroup takes the same branch and hits the same
// __syncthreads(), which is legal under HIP workgroup semantics.
//
// Invoked only when the KV cache is fp16-typed; f32 caches continue
// to use rmsnorm_qkv.hip.
//
// Launch:
//   dim3 grid (qRows + 2*kRows, 1, 1),
//   dim3 block(RMSNORM_QKV_FP16_LOCAL_SIZE, 1, 1)

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#ifndef RMSNORM_QKV_FP16_LOCAL_SIZE
#define RMSNORM_QKV_FP16_LOCAL_SIZE 128
#endif

extern "C" __global__ __launch_bounds__(RMSNORM_QKV_FP16_LOCAL_SIZE)
void rmsnorm_qkv_fp16(
    const float*  __restrict__ q_x,        // [qRows, K] fp32 workspace
    const float*  __restrict__ q_w,        // [K] fp32
          float*  __restrict__ q_y,        // in-place OK (== q_x)
    const __half* __restrict__ k_x,        // K cache base for this layer (fp16)
    const float*  __restrict__ k_w,        // [K] fp32
          __half* __restrict__ k_y,        // K cache base for this layer (fp16)
    const __half* __restrict__ v_x,        // V cache base for this layer (fp16)
          __half* __restrict__ v_y,        // V cache base for this layer (fp16)
    const int                  qRows,      // T * nHeads
    const int                  kRows,      // T * nKvHeads
    const int                  K,          // head_dim
    const float                eps,
    const int*    __restrict__ curLenPtr,  // device int slot
    const int                  kvDim)      // nKvHeads * head_dim
{
    __shared__ float scratch[RMSNORM_QKV_FP16_LOCAL_SIZE];

    const int gid   = blockIdx.x;
    const int tid   = threadIdx.x;
    const int lsize = blockDim.x;
    const size_t kvBase = static_cast<size_t>(curLenPtr[0])
                        * static_cast<size_t>(kvDim);

    // Branch on stream (Q / K / V). Uniform across the workgroup.
    if (gid < qRows) {
        // ---- Q branch: fp32 workspace, identical to rmsnorm_qkv.hip. ---
        const int row = gid;
        const float* __restrict__ xr = q_x + static_cast<size_t>(row) * static_cast<size_t>(K);
        const float* __restrict__ wr = q_w;
              float* __restrict__ yr = q_y + static_cast<size_t>(row) * static_cast<size_t>(K);

        float acc = 0.0f;
        for (int k = tid; k < K; k += lsize) {
            const float v = xr[k];
            acc = __fmaf_rn(v, v, acc);
        }
        scratch[tid] = acc;
        __syncthreads();
        for (int stride = lsize >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) scratch[tid] += scratch[tid + stride];
            __syncthreads();
        }
        const float invRms = rsqrtf(scratch[0] / static_cast<float>(K) + eps);
        for (int k = tid; k < K; k += lsize) {
            yr[k] = xr[k] * wr[k] * invRms;
        }
    } else if (gid < qRows + kRows) {
        // ---- K branch: fp16 cache, weighted rmsnorm. ------------------
        const int row = gid - qRows;
        const __half* __restrict__ xr =
            k_x + kvBase + static_cast<size_t>(row) * static_cast<size_t>(K);
        const float*  __restrict__ wr = k_w;
              __half* __restrict__ yr =
            k_y + kvBase + static_cast<size_t>(row) * static_cast<size_t>(K);

        float acc = 0.0f;
        for (int k = tid; k < K; k += lsize) {
            const float v = __half2float(xr[k]);
            acc = __fmaf_rn(v, v, acc);
        }
        scratch[tid] = acc;
        __syncthreads();
        for (int stride = lsize >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) scratch[tid] += scratch[tid + stride];
            __syncthreads();
        }
        const float invRms = rsqrtf(scratch[0] / static_cast<float>(K) + eps);
        for (int k = tid; k < K; k += lsize) {
            const float xv = __half2float(xr[k]);
            yr[k] = __float2half(xv * wr[k] * invRms);
        }
    } else {
        // ---- V branch: fp16 cache, no learned weight. ----------------
        const int row = gid - qRows - kRows;
        const __half* __restrict__ xr =
            v_x + kvBase + static_cast<size_t>(row) * static_cast<size_t>(K);
              __half* __restrict__ yr =
            v_y + kvBase + static_cast<size_t>(row) * static_cast<size_t>(K);

        float acc = 0.0f;
        for (int k = tid; k < K; k += lsize) {
            const float v = __half2float(xr[k]);
            acc = __fmaf_rn(v, v, acc);
        }
        scratch[tid] = acc;
        __syncthreads();
        for (int stride = lsize >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) scratch[tid] += scratch[tid + stride];
            __syncthreads();
        }
        const float invRms = rsqrtf(scratch[0] / static_cast<float>(K) + eps);
        for (int k = tid; k < K; k += lsize) {
            const float xv = __half2float(xr[k]);
            yr[k] = __float2half(xv * invRms);
        }
    }
}