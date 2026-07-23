// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// NVFP4 weight-only dequantisation: unpack an NVFP4 weight to BF16 so the
// existing BF16 matmul can consume it. This is the W4A16 path — activations
// stay BF16, only the weight is 4-bit in memory. No fp4 tensor-core; the
// win is load-time bandwidth (4-bit weights) with full BF16 compute quality.
//
// Reconstruction per element (matches core::modelopt::NvFp4Reference, whose
// e2m1/e4m3 decoders are verified bit-exact against CUTLASS):
//   value = global_scale * e4m3(block_scale) * e2m1(nibble)
// Layout: packed U8 [rows, in/2] (element 2j low nibble, 2j+1 high),
//         block_scale F8_E4M3 [rows, in/16] (one per 16-element block),
//         global_scale F32 scalar. `in` is a multiple of 16.
//
// Launch:
//   dim3 grid ( ceil(rows*in / DEQUANT_NVFP4_LOCAL), 1, 1 )
//   dim3 block( DEQUANT_NVFP4_LOCAL, 1, 1 )

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#ifndef DEQUANT_NVFP4_LOCAL
#define DEQUANT_NVFP4_LOCAL 256
#endif

// Decode a 4-bit E2M1 value (low nibble of `nib`). Magnitudes
// {0,.5,1,1.5,2,3,4,6}; bit 3 is the sign.
__device__ __forceinline__ float dq_e2m1(unsigned nib)
{
    const float mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float v = mag[nib & 0x7u];
    return (nib & 0x8u) ? -v : v;
}

// Decode an 8-bit E4M3 (FP8, bias 7) byte. Handles subnormals; S.1111.111
// is NaN (E4M3 has no infinities).
__device__ __forceinline__ float dq_e4m3(unsigned b)
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

extern "C" __global__ __launch_bounds__(DEQUANT_NVFP4_LOCAL)
void dequant_nvfp4(
    const unsigned char* __restrict__ packed,      // (rows, in/2)
    const unsigned char* __restrict__ block_scale, // (rows, in/16), E4M3
    const float                       global_scale,
    __nv_bfloat16*       __restrict__ out,         // (rows, in)
    const int                         rows,
    const int                         in)
{
    const long gid   = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long total = static_cast<long>(rows) * in;
    if (gid >= total) {
        return;
    }
    const int r = static_cast<int>(gid / in);
    const int j = static_cast<int>(gid % in);

    const unsigned char byte = packed[static_cast<long>(r) * (in / 2) + (j >> 1)];
    const unsigned      nib  = (j & 1) ? (byte >> 4) : (byte & 0x0Fu);
    const float         blk  = dq_e4m3(block_scale[static_cast<long>(r) * (in / 16) + (j >> 4)]);

    out[gid] = __float2bfloat16(global_scale * blk * dq_e2m1(nib));
}