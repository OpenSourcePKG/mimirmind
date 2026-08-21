// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// FP8 (E4M3) weight-only dequantisation with a PER-CHANNEL scale to BF16.
// This is the compressed-tensors (llm-compressor / vLLM) FP8 layout used by
// the dense qwen3_5 checkpoint (Qwen3.8-27B): each output row (channel) has
// its own scale, stored as a `[out, 1]` BF16 vector `weight_scale` — unlike
// the ModelOpt FP8 path (dequant_fp8.cu) which uses ONE per-tensor F32 scalar.
//
// Reconstruction: value(r,c) = weight_scale[r] * e4m3(weight_byte(r,c)),
// where r = gid / in is the output-row (channel) index for a row-major
// `[out, in]` weight. The e4m3 decoder matches dequant_fp8.cu /
// core::modelopt::NvFp4Reference (verified bit-exact vs CUTLASS).
//
// The scale vector is BF16 (the compressed-tensors convention); the kernel
// widens each channel scale to F32 before applying it.
//
// Launch:
//   dim3 grid ( ceil(rows*in / DEQUANT_FP8_PC_LOCAL), 1, 1 )
//   dim3 block( DEQUANT_FP8_PC_LOCAL, 1, 1 )

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#ifndef DEQUANT_FP8_PC_LOCAL
#define DEQUANT_FP8_PC_LOCAL 256
#endif

__device__ __forceinline__ float dqf8pc_e4m3(unsigned b)
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

extern "C" __global__ __launch_bounds__(DEQUANT_FP8_PC_LOCAL)
void dequant_fp8_perchannel(
    const unsigned char* __restrict__ weight,      // (rows, in) E4M3, row-major
    const __nv_bfloat16* __restrict__ weight_scale, // (rows,) per-channel BF16
    __nv_bfloat16*       __restrict__ out,          // (rows*in,)
    const long                        rows,
    const long                        in)
{
    const long gid = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long n   = rows * in;
    if (gid >= n) {
        return;
    }
    const long  row   = gid / in;                          // output channel
    const float scale = __bfloat162float(weight_scale[row]);
    out[gid] = __float2bfloat16(scale * dqf8pc_e4m3(weight[gid]));
}
