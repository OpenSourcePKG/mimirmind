// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Matrix-vector multiply with Q6_K weights, on-the-fly dequant.
//
//   Y[n] = sum_{k=0..K-1} X[k] * dequant_q6_k(W, n, k)
//
//   X:  [K]     F32 dense vector (single token / M=1)
//   W:  [N, K]  Q6_K — each row is (K/256) super-blocks of 210 bytes.
//   Y:  [N]     F32 dense vector
//
// Sibling of matmul_q5_0_vec.hip and matmul_q8_0_vec.hip: same warp
// layout (RDNA3 warpSize=32, one warp per output row, LOCAL=128 =
// 4 warps per WG, OUTPUTS_PER_GROUP=4). Only the per-lane dequant
// differs — Q6_K packs 256 6-bit signed quants per super-block with
// 16 per-sub-block int8 scales and a single fp16 super-scale.
//
// X is read directly from global memory (was: staged into an LDS tile
// with per-tile __syncthreads barriers). On gfx1101 the staging cost
// throughput on this decode matvec — X is tiny + L2-hot, the barriers
// only stalled the warps. Same change + measurement as matmul_q4k_vec.
//
// Launch:
//   dim3 grid ( ceil(N / OUTPUTS_PER_GROUP), 1, 1 )
//   dim3 block( MATMUL_Q6K_LOCAL, 1, 1 )
//
// Q6_K super-block layout (bit-identical to llama.cpp
// dequantize_row_q6_K and src/compute/quant/Q6K.cpp:59-98):
//
//   uint8   ql[128]   (bytes 0..127)   — 4-bit low-nibbles
//   uint8   qh[64]    (bytes 128..191) — 2-bit high-pairs (4 per byte)
//   int8    sc[16]    (bytes 192..207) — sub-block scales
//   fp16    d         (bytes 208..209) — super-block scale
//
// Element indexing (mirroring the CPU reference exactly):
//   The 256 elements split into two 128-element halves. Within each
//   half, for l in [0..32), four elements share qh[l]:
//     elem l+0   uses ql[l+0]  low nibble + qh[l] bits[0:1] + sc[is+0]
//     elem l+32  uses ql[l+32] low nibble + qh[l] bits[2:3] + sc[is+2]
//     elem l+64  uses ql[l+0]  high nibble + qh[l] bits[4:5] + sc[is+4]
//     elem l+96  uses ql[l+32] high nibble + qh[l] bits[6:7] + sc[is+6]
//   where is = l/16 (0 or 1 within the half). Second half advances
//   ql by 64, qh by 32, sc by 8.
//   Each 6-bit value: q = (nibble | (qh_pair << 4)) - 32  in [-32..31]
//   value = d * sc[sub_block] * q
//
// Lane assignment: each of the 32 lanes handles laneId == l for both
// halves, i.e. 8 elements per block per lane (4 in each half). 32
// lanes × 8 = 256 = full super-block, no cross-lane work per block.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#ifndef MATMUL_Q6K_LOCAL
#define MATMUL_Q6K_LOCAL 128
#endif

#define MATMUL_Q6K_WARPS             (MATMUL_Q6K_LOCAL / 32)
#define MATMUL_Q6K_OUTPUTS_PER_GROUP MATMUL_Q6K_WARPS

#define Q6K_BLOCK_ELEMENTS 256
#define Q6K_BLOCK_BYTES    210

namespace {

// Full-warp inclusive reduction via shuffle-down. warpSize is 32 on
// RDNA3, so five rounds of halving cover the whole warp. Returns the
// full sum in lane 0.
__device__ __forceinline__ float warpReduceSum(float v) {
    v += __shfl_down(v, 16);
    v += __shfl_down(v,  8);
    v += __shfl_down(v,  4);
    v += __shfl_down(v,  2);
    v += __shfl_down(v,  1);
    return v;
}

} // namespace

extern "C" __global__ __launch_bounds__(MATMUL_Q6K_LOCAL)
void matmul_q6k_vec(
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
    const int n      = wg * MATMUL_Q6K_OUTPUTS_PER_GROUP + warpId;
    if (n >= N) return;

    const int nBlocks = K / Q6K_BLOCK_ELEMENTS;

    // Precompute this lane's per-half scale-group index. l in [0..32),
    // is = l/16 ∈ {0,1} selects between the two sc[] runs within a half.
    const int l  = laneId;
    const int is = laneId >> 4;

    const unsigned char* row =
        W + static_cast<size_t>(n) * static_cast<size_t>(nBlocks) * Q6K_BLOCK_BYTES;

    float sum = 0.0f;

    for (int b = 0; b < nBlocks; ++b) {
        const unsigned char* block = row + b * Q6K_BLOCK_BYTES;

        const unsigned char* ql = block;
        const unsigned char* qh = block + 128;
        const signed char*   sc =
            reinterpret_cast<const signed char*>(block + 192);
        const __half*        d_ptr =
            reinterpret_cast<const __half*>(block + 208);
        const float d = __half2float(d_ptr[0]);

        const int xBase = b * Q6K_BLOCK_ELEMENTS;   // global X offset of block

        // ---- First 128-element half (elements 0..127) --------
        {
            const unsigned int ql0  = ql[l + 0];
            const unsigned int ql32 = ql[l + 32];
            const unsigned int qhv  = qh[l];

            const int q1 = static_cast<int>(
                (ql0  & 0x0Fu) | (((qhv >> 0) & 0x03u) << 4)) - 32;
            const int q2 = static_cast<int>(
                (ql32 & 0x0Fu) | (((qhv >> 2) & 0x03u) << 4)) - 32;
            const int q3 = static_cast<int>(
                (ql0  >> 4)    | (((qhv >> 4) & 0x03u) << 4)) - 32;
            const int q4 = static_cast<int>(
                (ql32 >> 4)    | (((qhv >> 6) & 0x03u) << 4)) - 32;

            const float s1 = static_cast<float>(sc[is + 0]);
            const float s2 = static_cast<float>(sc[is + 2]);
            const float s3 = static_cast<float>(sc[is + 4]);
            const float s4 = static_cast<float>(sc[is + 6]);

            sum = __fmaf_rn(X[xBase + l +  0], d * s1 * static_cast<float>(q1), sum);
            sum = __fmaf_rn(X[xBase + l + 32], d * s2 * static_cast<float>(q2), sum);
            sum = __fmaf_rn(X[xBase + l + 64], d * s3 * static_cast<float>(q3), sum);
            sum = __fmaf_rn(X[xBase + l + 96], d * s4 * static_cast<float>(q4), sum);
        }

        // ---- Second 128-element half (elements 128..255) -----
        {
            const unsigned int ql0  = ql[64 + l + 0];
            const unsigned int ql32 = ql[64 + l + 32];
            const unsigned int qhv  = qh[32 + l];

            const int q1 = static_cast<int>(
                (ql0  & 0x0Fu) | (((qhv >> 0) & 0x03u) << 4)) - 32;
            const int q2 = static_cast<int>(
                (ql32 & 0x0Fu) | (((qhv >> 2) & 0x03u) << 4)) - 32;
            const int q3 = static_cast<int>(
                (ql0  >> 4)    | (((qhv >> 4) & 0x03u) << 4)) - 32;
            const int q4 = static_cast<int>(
                (ql32 >> 4)    | (((qhv >> 6) & 0x03u) << 4)) - 32;

            const float s1 = static_cast<float>(sc[8 + is + 0]);
            const float s2 = static_cast<float>(sc[8 + is + 2]);
            const float s3 = static_cast<float>(sc[8 + is + 4]);
            const float s4 = static_cast<float>(sc[8 + is + 6]);

            sum = __fmaf_rn(X[xBase + 128 + l +  0], d * s1 * static_cast<float>(q1), sum);
            sum = __fmaf_rn(X[xBase + 128 + l + 32], d * s2 * static_cast<float>(q2), sum);
            sum = __fmaf_rn(X[xBase + 128 + l + 64], d * s3 * static_cast<float>(q3), sum);
            sum = __fmaf_rn(X[xBase + 128 + l + 96], d * s4 * static_cast<float>(q4), sum);
        }
    }

    sum = warpReduceSum(sum);

    if (laneId == 0) {
        Y[n] = sum;
    }
}
