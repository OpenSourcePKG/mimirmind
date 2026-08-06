// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Vectorised NVFP4 decode matvec on a DE-INTERLEAVED blocked layout.
//
// The existing blocked-NVFP4 weight is a 20-byte super (fp16 s0 | fp16 s1 |
// 16 E2M1 nibble-bytes = 32 elements). The 20-byte stride is not uint4-
// alignable, so the current kernels (matmul_nvfp4blk_vec, moe_grouped_gemm_
// nvfp4blk) read it byte/half-wise -> narrow memory transactions -> ~42% of
// peak DRAM bandwidth (latency-bound; the M4 occupancy fix is already applied).
//
// This PoC kernel consumes a DE-INTERLEAVED layout instead:
//   nib   [N, nSuper, 16]  bytes  — the 16 nibble-bytes of each super, 16-byte
//                                    aligned so a whole super loads as ONE uint4.
//   scale [N, nSuper, 2]   fp16   — (s0, s1) per super.
// Parallelisation: warp == output row n; lane l streams supers l, l+32, ...
// So 32 lanes load 32 CONSECUTIVE supers' nibbles as 32 consecutive uint4 =
// a single coalesced 512-byte transaction -> peak bandwidth. Each lane unpacks
// its super's 32 elements from registers (no smem, no per-element global read),
// applies the two fp16 scales, and FMAs against the shared activation tile.
//
//   y[n] = sum over all 32*nSuper elements  x[k] * (scale_block * e2m1(nib))
//
// Launch: grid.x = ceil(N / WARPS), block = LOCAL (128 = 4 warps).
//   dynamic smem: K floats (the activation vector, staged once).

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifndef NVBLK_DEINT_LOCAL
#define NVBLK_DEINT_LOCAL 128
#endif
#define NVBLK_DEINT_WARPS  (NVBLK_DEINT_LOCAL / 32)
#define NVBLK_SUPER_ELEMS  32

namespace {

__device__ __forceinline__ float dq_e2m1(unsigned nib) {
    const float mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float v = mag[nib & 0x7u];
    return (nib & 0x8u) ? -v : v;
}

__device__ __forceinline__ float warpReduceSum(float v) {
    v += __shfl_down_sync(0xffffffffu, v, 16);
    v += __shfl_down_sync(0xffffffffu, v,  8);
    v += __shfl_down_sync(0xffffffffu, v,  4);
    v += __shfl_down_sync(0xffffffffu, v,  2);
    v += __shfl_down_sync(0xffffffffu, v,  1);
    return v;
}

} // namespace

extern "C" __global__ __launch_bounds__(NVBLK_DEINT_LOCAL)
void matmul_nvfp4blk_deint_vec(
    const float*         __restrict__ X,        // [K] activation
    const unsigned char* __restrict__ nib,      // [N, nSuper, 16] bytes
    const __half*        __restrict__ scale,    // [N, nSuper, 2]
          float*         __restrict__ Y,        // [N]
    const int                          N,
    const int                          K)
{
    extern __shared__ float sx[];               // [K]
    const int tid    = threadIdx.x;
    const int lsize  = blockDim.x;
    const int warpId = tid / 32;
    const int laneId = tid % 32;
    const int nSuper = K / NVBLK_SUPER_ELEMS;

    for (int k = tid; k < K; k += lsize) {
        sx[k] = X[k];
    }
    __syncthreads();

    const int n = blockIdx.x * NVBLK_DEINT_WARPS + warpId;
    if (n >= N) {
        return;
    }

    // Row bases: nibbles 16-byte aligned per super, scales 2 fp16 per super.
    const uint4*  __restrict__ nibRow =
        reinterpret_cast<const uint4*>(nib + static_cast<size_t>(n) * nSuper * 16);
    const __half* __restrict__ scRow  =
        scale + static_cast<size_t>(n) * nSuper * 2;

    float acc = 0.0f;
    // Lane l streams supers l, l+32, ... ; the 32 lanes of a step load 32
    // consecutive uint4 = one coalesced 512-byte transaction.
    for (int sp = laneId; sp < nSuper; sp += 32) {
        const uint4 q = nibRow[sp];                 // 16 bytes = 32 nibbles
        const float s0 = __half2float(scRow[sp * 2 + 0]);
        const float s1 = __half2float(scRow[sp * 2 + 1]);
        const int   k0 = sp * NVBLK_SUPER_ELEMS;    // first element index
        const unsigned words[4] = {q.x, q.y, q.z, q.w};
        #pragma unroll
        for (int b = 0; b < 16; ++b) {              // 16 nibble-bytes
            const unsigned byte = (words[b >> 2] >> ((b & 3) * 8)) & 0xFFu;
            const int e = b * 2;                    // two elements per byte
            const float sc0 = (e     < 16) ? s0 : s1;
            const float sc1 = ((e+1) < 16) ? s0 : s1;
            acc = __fmaf_rn(sc0 * dq_e2m1(byte & 0x0F),  sx[k0 + e],     acc);
            acc = __fmaf_rn(sc1 * dq_e2m1(byte >> 4),    sx[k0 + e + 1], acc);
        }
    }
    const float s = warpReduceSum(acc);
    if (laneId == 0) {
        Y[n] = s;
    }
}
