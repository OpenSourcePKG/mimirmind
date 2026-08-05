// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Per-tensor E4M3 quantisation helpers for the cuBLASLt FP8 dense-matmul path
// (GpuMatmul::cublasFp8Matmul). cuBLASLt FP8 wants both operands in E4M3 with a
// per-tensor FP32 scale (CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F):
//   D = (a_scale * Aq) @ (b_scale * Bq)
// so we quantise as  q = round_e4m3(x / scale),  scale = amax / 448.
//
// Four kernels: amax over F32 / BF16 (device reduction -> single float), and
// cast-with-inv-scale F32 -> E4M3 / BF16 -> E4M3. amax_* atomically max a
// non-negative float via its int bit-pattern (monotonic for x >= 0); the out
// slot MUST be zero-initialised by the caller.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#ifndef CUBLAS_FP8_LOCAL
#define CUBLAS_FP8_LOCAL 256
#endif

extern "C" __global__ __launch_bounds__(CUBLAS_FP8_LOCAL)
void amax_f32(const float* __restrict__ src, float* __restrict__ outMax,
              const long n) {
    __shared__ float sm[CUBLAS_FP8_LOCAL];
    const long gid = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    float v = 0.0f;
    if (gid < n) { v = fabsf(src[gid]); }
    sm[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sm[threadIdx.x] = fmaxf(sm[threadIdx.x], sm[threadIdx.x + s]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicMax(reinterpret_cast<int*>(outMax), __float_as_int(sm[0]));
    }
}

extern "C" __global__ __launch_bounds__(CUBLAS_FP8_LOCAL)
void amax_bf16(const __nv_bfloat16* __restrict__ src, float* __restrict__ outMax,
               const long n) {
    __shared__ float sm[CUBLAS_FP8_LOCAL];
    const long gid = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    float v = 0.0f;
    if (gid < n) { v = fabsf(__bfloat162float(src[gid])); }
    sm[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sm[threadIdx.x] = fmaxf(sm[threadIdx.x], sm[threadIdx.x + s]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicMax(reinterpret_cast<int*>(outMax), __float_as_int(sm[0]));
    }
}

// From a device amax, derive the cuBLAS per-tensor scale (amax/448, the
// dequant multiplier cuBLAS applies) and its inverse (the quant multiplier).
// Kept device-side so the decode hot path never D2H-syncs for the X scale.
extern "C" __global__ void scale_from_amax(const float* __restrict__ amax,
                                           float* __restrict__ scale,
                                           float* __restrict__ invScale) {
    const float a = *amax;
    const float s = a / 448.0f;
    *scale    = s;
    *invScale = (a > 0.0f) ? (448.0f / a) : 0.0f;
}

extern "C" __global__ __launch_bounds__(CUBLAS_FP8_LOCAL)
void cast_f32_to_e4m3(const float* __restrict__ src,
                      __nv_fp8_storage_t* __restrict__ out,
                      const float* __restrict__ invScale, const long n) {
    const long gid = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= n) { return; }
    out[gid] = __nv_cvt_float_to_fp8(src[gid] * (*invScale),
                                     __NV_SATFINITE, __NV_E4M3);
}

extern "C" __global__ __launch_bounds__(CUBLAS_FP8_LOCAL)
void cast_bf16_to_e4m3(const __nv_bfloat16* __restrict__ src,
                       __nv_fp8_storage_t* __restrict__ out,
                       const float* __restrict__ invScale, const long n) {
    const long gid = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= n) { return; }
    out[gid] = __nv_cvt_float_to_fp8(__bfloat162float(src[gid]) * (*invScale),
                                     __NV_SATFINITE, __NV_E4M3);
}
