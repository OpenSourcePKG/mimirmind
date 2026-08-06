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

// ---------------------------------------------------------------------------
// Grouped (device-driven, expert-indexed) variant for the MoE decode path.
// Same coalesced-uint4 lane==super load as above, but the tile schedule
// (moe_group_tiles) supplies (expert e, row0, rows) so one launch covers every
// active expert. Decode-shaped: rows is small (1 at nSeq=1); x rows are staged
// to smem (rows * K floats). De-interleaved expert bank:
//   nib   [nExperts, N, nSuper, 16]   scale [nExperts, N, nSuper, 2].
// ---------------------------------------------------------------------------
// Decode single-user path: nSeq==1 => at most one activation row per tile, so
// MAX_M=1 keeps the staged-activation smem at K floats (good occupancy). The
// GpuOps wrapper only dispatches this kernel when nSeq==1.
#ifndef NVBLK_DEINT_MAX_M
#define NVBLK_DEINT_MAX_M 1
#endif

// One thread per 20-byte super: split the interleaved blocked-NVFP4 bank into
// a contiguous 16-byte-aligned nibble bank + a 2-fp16 scale bank. Run once at
// first use; the result is cached by the caller.
//   src   [totalSupers, 20]   dstNib [totalSupers, 16]   dstSc [totalSupers, 4]
extern "C" __global__ void nvfp4blk_deinterleave(
    const unsigned char* __restrict__ src,
          unsigned char* __restrict__ dstNib,
          unsigned char* __restrict__ dstSc,
    const long                          totalSupers)
{
    const long sp = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (sp >= totalSupers) {
        return;
    }
    const unsigned char* s = src + sp * 20;
    unsigned char* dn = dstNib + sp * 16;
    unsigned char* ds = dstSc  + sp * 4;
    #pragma unroll
    for (int b = 0; b < 4;  ++b) ds[b] = s[b];
    #pragma unroll
    for (int b = 0; b < 16; ++b) dn[b] = s[4 + b];
}

extern "C" __global__ __launch_bounds__(NVBLK_DEINT_LOCAL)
void moe_grouped_gemm_nvfp4blk_deint(
    const float*         __restrict__ X,        // [R, K] compacted activations
    const unsigned char* __restrict__ nib,      // [nExperts, N, nSuper, 16]
    const __half*        __restrict__ scale,    // [nExperts, N, nSuper, 2]
          float*         __restrict__ Y,        // [R, N]
    const int*           __restrict__ tileExpert,
    const int*           __restrict__ tileRow0,
    const int*           __restrict__ tileRows,
    const int                          K,
    const int                          N)
{
    extern __shared__ float sxg[];              // [MAX_M * K]
    const int tt = blockIdx.y;
    const int e  = tileExpert[tt];
    if (e < 0) {
        return;
    }
    const int row0 = tileRow0[tt];
    int M = tileRows[tt];
    if (M > NVBLK_DEINT_MAX_M) {
        M = NVBLK_DEINT_MAX_M;
    }

    const int tid    = threadIdx.x;
    const int lsize  = blockDim.x;
    const int warpId = tid / 32;
    const int laneId = tid % 32;
    const int nSuper = K / NVBLK_SUPER_ELEMS;

    for (int i = tid; i < M * K; i += lsize) {
        sxg[i] = X[static_cast<size_t>(row0) * K + i];
    }
    __syncthreads();

    const int n = blockIdx.x * NVBLK_DEINT_WARPS + warpId;
    if (n >= N) {
        return;
    }

    const size_t rowSupers = static_cast<size_t>(nSuper);
    const uint4* __restrict__ nibRow = reinterpret_cast<const uint4*>(
        nib + ((static_cast<size_t>(e) * N + n) * rowSupers) * 16);
    const __half* __restrict__ scRow =
        scale + ((static_cast<size_t>(e) * N + n) * rowSupers) * 2;

    float acc[NVBLK_DEINT_MAX_M];
    #pragma unroll
    for (int m = 0; m < NVBLK_DEINT_MAX_M; ++m) acc[m] = 0.0f;

    for (int sp = laneId; sp < nSuper; sp += 32) {
        const uint4 q = nibRow[sp];
        const float s0 = __half2float(scRow[sp * 2 + 0]);
        const float s1 = __half2float(scRow[sp * 2 + 1]);
        const int   k0 = sp * NVBLK_SUPER_ELEMS;
        const unsigned words[4] = {q.x, q.y, q.z, q.w};
        #pragma unroll
        for (int b = 0; b < 16; ++b) {
            const unsigned byte = (words[b >> 2] >> ((b & 3) * 8)) & 0xFFu;
            const int   e0  = b * 2;
            const float w0  = ((e0     < 16) ? s0 : s1) * dq_e2m1(byte & 0x0F);
            const float w1  = ((e0 + 1 < 16) ? s0 : s1) * dq_e2m1(byte >> 4);
            for (int m = 0; m < M; ++m) {
                acc[m] = __fmaf_rn(w0, sxg[m * K + k0 + e0],     acc[m]);
                acc[m] = __fmaf_rn(w1, sxg[m * K + k0 + e0 + 1], acc[m]);
            }
        }
    }
    for (int m = 0; m < M; ++m) {
        const float s = warpReduceSum(acc[m]);
        if (laneId == 0) {
            Y[static_cast<size_t>(row0 + m) * N + n] = s;
        }
    }
}
