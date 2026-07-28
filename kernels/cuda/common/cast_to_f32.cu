// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Widen a half-width float tensor (BF16 or F16) to F32. The NVFP4 W4A16
// checkpoint stores the unquantised tensors the runtime reads as F32 device
// pointers — norms (attn/q/k/ssm/ffn/post/output), the SSM scalars (ssm_a,
// ssm_dt), conv1d, biases and the (untied) router/embedding — in BF16 or
// F32. Quantised matmul weights stay BF16 and feed the BF16 matmul path;
// these passthrough tensors are cast to F32 here so the RMS-norm / SSM /
// bias kernels (which all `static_cast<const float*>` the weight) read a
// real F32 buffer. F32 sources need no cast and are copied byte-for-byte by
// the caller.
//
// Launch:
//   dim3 grid ( ceil(n / CAST_TO_F32_LOCAL), 1, 1 )
//   dim3 block( CAST_TO_F32_LOCAL, 1, 1 )

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#ifndef CAST_TO_F32_LOCAL
#define CAST_TO_F32_LOCAL 256
#endif

extern "C" __global__ __launch_bounds__(CAST_TO_F32_LOCAL)
void cast_bf16_to_f32(
    const __nv_bfloat16* __restrict__ src, // (n,) BF16
    float*               __restrict__ out, // (n,)
    const long                        n)
{
    const long gid = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= n) {
        return;
    }
    out[gid] = __bfloat162float(src[gid]);
}

extern "C" __global__ __launch_bounds__(CAST_TO_F32_LOCAL)
void cast_f16_to_f32(
    const __half* __restrict__ src, // (n,) F16
    float*        __restrict__ out, // (n,)
    const long                 n)
{
    const long gid = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= n) {
        return;
    }
    out[gid] = __half2float(src[gid]);
}