// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// FP8 (E4M3) weight-only dequantisation to BF16. The attention projections
// on the W4A16 NVFP4 checkpoint are stored FP8; this unpacks them to BF16 so
// the existing BF16 matmul consumes them (weight-only path — activations
// stay BF16). Per-tensor F32 scale; the input_scale (activation) is unused.
//
// Reconstruction: value = weight_scale * e4m3(weight_byte). The e4m3 decoder
// matches core::modelopt::NvFp4Reference (verified bit-exact vs CUTLASS).
//
// Launch:
//   dim3 grid ( ceil(n / DEQUANT_FP8_LOCAL), 1, 1 )
//   dim3 block( DEQUANT_FP8_LOCAL, 1, 1 )

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#ifndef DEQUANT_FP8_LOCAL
#define DEQUANT_FP8_LOCAL 256
#endif

__device__ __forceinline__ float dqf8_e4m3(unsigned b)
{
    const unsigned s = (b >> 7) & 0x1u;
    const unsigned e = (b >> 3) & 0xFu;
    const unsigned m = b & 0x7u;
    const float sign = s ? -1.0f : 1.0f;
    if (e == 0u) {
        return sign * ldexpf(static_cast<float>(m) / 8.0f, -6);
    }
    if (e == 0xFu && m == 0x7u) {
        return __int_as_float(0x7fffffff); // NaN
    }
    return sign * ldexpf(1.0f + static_cast<float>(m) / 8.0f, static_cast<int>(e) - 7);
}

extern "C" __global__ __launch_bounds__(DEQUANT_FP8_LOCAL)
void dequant_fp8(
    const unsigned char* __restrict__ weight,       // (n,) E4M3
    const float                       weight_scale, // per-tensor F32
    __nv_bfloat16*       __restrict__ out,          // (n,)
    const long                        n)
{
    const long gid = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= n) {
        return;
    }
    out[gid] = __float2bfloat16(weight_scale * dqf8_e4m3(weight[gid]));
}