// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// 5.18.10.4 — BF16 SSM-state coherence de-risk. Round an F32 buffer to BF16
// precision IN PLACE (x = bf16(x) widened back to f32). Used to SIMULATE storing
// the recurrent GatedDeltaNet SSM/conv state in BF16 (halving the ~256 MiB/layer/
// step R+W, the only remaining gdn.recur decode lever) WITHOUT the invasive
// F32->BF16 storage change: the backend calls this on the state slice after each
// recurrence advance (gated MIMIRMIND_SSM_BF16_SIM), so the next step/chunk reads
// the bf16-rounded state — exactly the accumulator-rounding a real bf16 store
// would incur. Answers the coherence question (needle / long-context) before any
// storage rewrite. NOT a production path.
//
// Launch:
//   dim3 grid ( ceil(n / SSM_ROUND_BF16_LOCAL), 1, 1 )
//   dim3 block( SSM_ROUND_BF16_LOCAL, 1, 1 )

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#ifndef SSM_ROUND_BF16_LOCAL
#define SSM_ROUND_BF16_LOCAL 256
#endif

extern "C" __global__ __launch_bounds__(SSM_ROUND_BF16_LOCAL)
void ssm_round_bf16(
    float* __restrict__ buf,  // (n,) F32, rounded in place to bf16 precision
    const int           n)
{
    const int gid = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= n) {
        return;
    }
    buf[gid] = __bfloat162float(__float2bfloat16(buf[gid]));
}
