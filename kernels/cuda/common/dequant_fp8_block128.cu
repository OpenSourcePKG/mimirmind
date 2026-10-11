// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// FP8 (E4M3) 128x128 block-scaled weight dequantisation to BF16 — the
// DeepSeek-V3 / Kolibri-1 producer format: each weight byte is E4M3, and one
// F32 scale covers a 128x128 tile of the [rows, in] weight matrix, stored in a
// companion `*.weight_scale_inv` tensor of shape [ceil(rows/128), ceil(in/128)]
// (row-major). Unpacks to BF16 so the existing BF16 path consumes it (the
// Kolibri loader then re-quantises to the 32-block GgmlType::FP8_E4M3 the
// matmul_fp8 kernels expect — roadmap 5.34.4).
//
// Reconstruction: value = scale_inv[r/128][c/128] * e4m3(weight_byte). The
// e4m3 decoder matches dequant_fp8.cu / core::modelopt::NvFp4Reference.
//
// Launch:
//   dim3 grid ( ceil(rows*in / DEQUANT_FP8_BLK128_LOCAL), 1, 1 )
//   dim3 block( DEQUANT_FP8_BLK128_LOCAL, 1, 1 )

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#ifndef DEQUANT_FP8_BLK128_LOCAL
#define DEQUANT_FP8_BLK128_LOCAL 256
#endif

__device__ __forceinline__ float dqf8blk_e4m3(unsigned b)
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

extern "C" __global__ __launch_bounds__(DEQUANT_FP8_BLK128_LOCAL)
void dequant_fp8_block128(
    const unsigned char* __restrict__ weight,    // [rows*in] E4M3, row-major
    const float*         __restrict__ scale_inv, // [scale_rows*scale_cols] F32
    __nv_bfloat16*       __restrict__ out,        // [rows*in]
    const long                        rows,
    const long                        in,
    const long                        scale_cols) // = ceil(in / 128)
{
    const long gid = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long n   = rows * in;
    if (gid >= n) {
        return;
    }
    const long r = gid / in;
    const long c = gid - r * in;
    const float scale = scale_inv[(r >> 7) * scale_cols + (c >> 7)];
    out[gid] = __float2bfloat16(scale * dqf8blk_e4m3(weight[gid]));
}
