// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
// Ported from kernels_hip/qkv_split_fp16.hip — Track 4 mechanical port, no functional change intended.

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// FP16-KV variant of qkv_split.hip. Fused matmul output is fp32; Q
// goes to an fp32 workspace (`Yq`) unchanged, K and V go to the fp16
// KV cache via __float2half.
//
// Layout / dispatch geometry identical to the f32 variant — host
// wiring differs only in the pointer types of `Yk` / `Yv`. Invoked
// only when the KV cache is fp16-typed; f32 caches continue to use
// qkv_split.hip.
//
// Launch:
//   dim3 grid ( ceil(M*Nfused / QKV_SPLIT_FP16_LOCAL), 1, 1 )
//   dim3 block( QKV_SPLIT_FP16_LOCAL, 1, 1 )

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifndef QKV_SPLIT_FP16_LOCAL
#define QKV_SPLIT_FP16_LOCAL 256
#endif

extern "C" __global__ __launch_bounds__(QKV_SPLIT_FP16_LOCAL)
void qkv_split_fp16(
    const float*   __restrict__ fused,
          float*   __restrict__ Yq,
          __half*  __restrict__ Yk,
          __half*  __restrict__ Yv,
    const int                   M,
    const int                   Nq,
    const int                   Nkv,
    const int                   hasV,
    const int                   Nfused,
    const int*     __restrict__ curLenPtr)
{
    const int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = M * Nfused;
    if (idx >= total) return;

    const int m      = idx / Nfused;
    const int n      = idx - m * Nfused;
    const int curLen = curLenPtr[0];

    const float v = fused[idx];

    if (n < Nq) {
        Yq[m * Nq + n] = v;
    } else if (n < Nq + Nkv) {
        Yk[(curLen + m) * Nkv + (n - Nq)] = __float2half(v);
    } else if (hasV != 0) {
        Yv[(curLen + m) * Nkv + (n - Nq - Nkv)] = __float2half(v);
    }
}