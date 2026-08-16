// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Staging-redirect rmsnorm_qkv for FP16 KV on models whose K/V projection is
// NOT fused-QKV (e.g. Qwen3-Next: Q is fused with the per-head gate, K/V are
// separate F32 matmuls). Such a model cannot write straight into an fp16 slot
// (a raw fp32 matmul would corrupt it), so K/V are projected into an fp32
// scratch and this kernel folds each row: RMS-norm in fp32, then cast+store
// to the fp16 cache — the fp16 analogue of kv_quant_commit_q8_0's staging.
//
// Differs from rmsnorm_qkv_fp16 in the K/V *read*: there the K/V input aliases
// the fp16 cache in-place (offset by curLen); here the input is a per-forward
// fp32 scratch indexed at row*K (no curLen offset), while the fp16 *write*
// still lands at the cache's curLen write offset (kvBase + row*K). Q is
// unchanged (fp32 workspace, in-place).
//
// The sum-of-squares and invRms are fp32 regardless; writes use __float2half.
//
// Launch:
//   dim3 grid (qRows + 2*kRows, 1, 1),
//   dim3 block(RMSNORM_QKV_FP16_LOCAL_SIZE, 1, 1)

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifndef RMSNORM_QKV_FP16_LOCAL_SIZE
#define RMSNORM_QKV_FP16_LOCAL_SIZE 128
#endif

extern "C" __global__ __launch_bounds__(RMSNORM_QKV_FP16_LOCAL_SIZE)
void rmsnorm_qkv_f32in_fp16out(
    const float*  __restrict__ q_x,        // [qRows, K] fp32 workspace
    const float*  __restrict__ q_w,        // [K] fp32
          float*  __restrict__ q_y,        // in-place OK (== q_x)
    const float*  __restrict__ k_x,        // [kRows, K] fp32 K scratch (this forward)
    const float*  __restrict__ k_w,        // [K] fp32
          __half* __restrict__ k_y,        // K cache base for this layer (fp16)
    const float*  __restrict__ v_x,        // [kRows, K] fp32 V scratch (this forward)
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

    if (gid < qRows) {
        // ---- Q branch: fp32 workspace, in-place (identical to rmsnorm_qkv). --
        const int row = gid;
        const float* __restrict__ xr = q_x + static_cast<size_t>(row) * K;
        const float* __restrict__ wr = q_w;
              float* __restrict__ yr = q_y + static_cast<size_t>(row) * K;

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
        // ---- K branch: fp32 scratch in, fp16 cache out, weighted rmsnorm. ---
        const int row = gid - qRows;
        const float*  __restrict__ xr = k_x + static_cast<size_t>(row) * K;
        const float*  __restrict__ wr = k_w;
              __half* __restrict__ yr =
            k_y + kvBase + static_cast<size_t>(row) * static_cast<size_t>(K);

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
            yr[k] = __float2half(xr[k] * wr[k] * invRms);
        }
    } else {
        // ---- V branch: fp32 scratch in, fp16 cache out, no learned weight. --
        const int row = gid - qRows - kRows;
        const float*  __restrict__ xr = v_x + static_cast<size_t>(row) * K;
              __half* __restrict__ yr =
            v_y + kvBase + static_cast<size_t>(row) * static_cast<size_t>(K);

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
            yr[k] = __float2half(xr[k] * invRms);
        }
    }
}
