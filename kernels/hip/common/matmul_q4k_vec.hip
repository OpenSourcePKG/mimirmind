// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Matrix-vector multiply with Q4_K weights, on-the-fly dequant.
//
//   Y[n] = sum_{k=0..K-1} X[k] * dequant_q4_k(W, n, k)
//
//   X:  [K]     F32 dense vector (single token / M=1)
//   W:  [N, K]  Q4_K — each row is (K/256) super-blocks of 144 bytes.
//   Y:  [N]     F32 dense vector
//
// Sibling of matmul_q5_0_vec.hip / matmul_q6k_vec.hip: same warp
// layout (RDNA3 warpSize=32, one warp per output row, LOCAL=128 =
// 4 warps per WG, OUTPUTS_PER_GROUP=4). Q4_K is the asymmetric
// K-quant — every sub-block has its own scale AND min, so per
// element:  value = d * scale * nibble - dmin * min.
//
// X is read directly from global memory. An earlier version staged X
// into an LDS tile with per-tile __syncthreads barriers; on gfx1101
// that cost throughput (this decode matvec is ~75 % of decode GPU
// time, memory/latency-bound, and X is tiny + stays hot in L2 so the
// staging bought nothing while the barriers stalled the warps). A/B
// bench on gfx1101: dropping the LDS tiling was 1.08-1.20x faster
// bit-for-bit across FFN/attn shapes.
//
// Launch:
//   dim3 grid ( ceil(N / OUTPUTS_PER_GROUP), 1, 1 )
//   dim3 block( MATMUL_Q4K_LOCAL, 1, 1 )
//
// Q4_K super-block layout (bit-identical to llama.cpp
// dequantize_row_q4_K and src/compute/quant/Q4K.cpp:63-105):
//
//   fp16    d          (bytes 0..1)    — scale-of-scales
//   fp16    dmin       (bytes 2..3)    — scale-of-mins
//   uint8   scales[12] (bytes 4..15)   — 8 × (6-bit scale, 6-bit min),
//                                        packed via ggml's
//                                        get_scale_min_k4 format
//   uint8   qs[128]    (bytes 16..143) — 256 nibbles, low-nibble first
//
// The 256 elements split into eight 32-element sub-blocks; sub-blocks
// j and j+1 (j = 0, 2, 4, 6) share the same 32-byte qs chunk — low
// nibbles feed sub-block j (elements 0..31), high nibbles feed
// sub-block j+1 (elements 32..63). Each sub-block j has its own
// (scale_j, min_j) pair and dequant  value = (d*scale_j) * nibble - (dmin*min_j).
//
// Lane assignment: laneId l ∈ [0..32) handles the 8 elements at
// positions {l, 32+l, 64+l, ..., 224+l} — one element per sub-block,
// eight sub-blocks per super-block, exactly 32 lanes × 8 = 256 = full
// super-block.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#ifndef MATMUL_Q4K_LOCAL
#define MATMUL_Q4K_LOCAL 128
#endif

#define MATMUL_Q4K_WARPS             (MATMUL_Q4K_LOCAL / 32)
#define MATMUL_Q4K_OUTPUTS_PER_GROUP MATMUL_Q4K_WARPS

#define Q4K_BLOCK_ELEMENTS 256
#define Q4K_BLOCK_BYTES    144

namespace {

// Full-warp inclusive reduction via shuffle-down. warpSize is 32 on
// RDNA3; five rounds of halving sum the whole warp into lane 0.
__device__ __forceinline__ float warpReduceSum(float v) {
    v += __shfl_down(v, 16);
    v += __shfl_down(v,  8);
    v += __shfl_down(v,  4);
    v += __shfl_down(v,  2);
    v += __shfl_down(v,  1);
    return v;
}

// Unpack one of the 8 (scale, min) pairs from the 12-byte packed
// scales field. Matches ggml's get_scale_min_k4 and Q4K.cpp:48-61.
__device__ __forceinline__ void getScaleMinK4(
    int j, const unsigned char* q, unsigned int& scale, unsigned int& min) {
    if (j < 4) {
        scale = static_cast<unsigned int>(q[j]     & 0x3Fu);
        min   = static_cast<unsigned int>(q[j + 4] & 0x3Fu);
    } else {
        scale = static_cast<unsigned int>(
            (q[j + 4] & 0x0Fu) | ((q[j - 4] >> 6) << 4));
        min   = static_cast<unsigned int>(
            (q[j + 4] >> 4)    | ((q[j    ] >> 6) << 4));
    }
}

} // namespace

extern "C" __global__ __launch_bounds__(MATMUL_Q4K_LOCAL)
void matmul_q4k_vec(
    const float*         __restrict__ X,
    const unsigned char* __restrict__ W,
          float*         __restrict__ Y,
    const int                          K,
    const int                          N)
{
    const int wg     = blockIdx.x;
    const int tid    = threadIdx.x;
    const int warpId = tid / 32;
    const int laneId = tid % 32;
    const int n      = wg * MATMUL_Q4K_OUTPUTS_PER_GROUP + warpId;
    if (n >= N) return;

    const int nBlocks = K / Q4K_BLOCK_ELEMENTS;
    const int l       = laneId;

    const unsigned char* row =
        W + static_cast<size_t>(n) * static_cast<size_t>(nBlocks) * Q4K_BLOCK_BYTES;

    float sum = 0.0f;

    for (int b = 0; b < nBlocks; ++b) {
        const unsigned char* block = row + b * Q4K_BLOCK_BYTES;

        const __half* d_ptr    = reinterpret_cast<const __half*>(block);
        const __half* dmin_ptr = reinterpret_cast<const __half*>(block + 2);
        const float d    = __half2float(d_ptr[0]);
        const float dmin = __half2float(dmin_ptr[0]);

        const unsigned char* scales = block + 4;
        const unsigned char* qs     = block + 16;
        const int xBase = b * Q4K_BLOCK_ELEMENTS;   // global X offset of block

        #pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const int jLo = 2 * pair;
            const int jHi = jLo + 1;

            unsigned int sLo, mLo, sHi, mHi;
            getScaleMinK4(jLo, scales, sLo, mLo);
            getScaleMinK4(jHi, scales, sHi, mHi);

            const float dLo  = d    * static_cast<float>(sLo);
            const float mmLo = dmin * static_cast<float>(mLo);
            const float dHi  = d    * static_cast<float>(sHi);
            const float mmHi = dmin * static_cast<float>(mHi);

            const unsigned char qb = qs[pair * 32 + l];
            const float nibLo = static_cast<float>(qb & 0x0Fu);
            const float nibHi = static_cast<float>(qb >> 4);

            const float wLo = dLo * nibLo - mmLo;
            const float wHi = dHi * nibHi - mmHi;

            const float xLo = X[xBase + jLo * 32 + l];
            const float xHi = X[xBase + jHi * 32 + l];

            sum = __fmaf_rn(xLo, wLo, sum);
            sum = __fmaf_rn(xHi, wHi, sum);
        }
    }

    sum = warpReduceSum(sum);

    if (laneId == 0) {
        Y[n] = sum;
    }
}
