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
    // Branchless E2M1 (3-bit magnitude {0,.5,1,1.5,2,3,4,6}) -> fp32 by
    // constructing the IEEE-754 bit pattern in registers. Avoids the dynamic
    // index into a local `float mag[8]` array (which lands in local memory =
    // MIO-pipe traffic on the decode-GEMV's short-scoreboard-bound inner loop).
    // fp32 exp = 126 + (m>>1); mantissa bit = (m&1)<<22; zeroed when m==0.
    const unsigned m    = nib & 0x7u;
    const unsigned bits = ((126u + (m >> 1)) << 23) | ((m & 1u) << 22);
    const float    v    = __uint_as_float(bits) * static_cast<float>(m != 0u);
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
    // No smem staging: reading the activation from smem here means one smem
    // load per FMA, and with lane==super the access xTile[super*32+e] is
    // stride-32 (128 B) => 32-way bank conflict. ncu on the sibling m4 kernel
    // showed the grouped decode GEMM is MIO/shared-pipe bound (56-61% short-
    // scoreboard stall) at 24% DRAM / 103% occupancy -- NOT DRAM- or occupancy-
    // limited. So read x straight from global: it is tiny (one row, K floats),
    // L2-resident (re-read by every grid.x block of this tile), and lands on the
    // L1TEX pipe instead of the saturated MIO pipe.
    const int tt = blockIdx.y;
    const int e  = tileExpert[tt];
    if (e < 0) {
        return;
    }
    const int row0 = tileRow0[tt];

    const int tid    = threadIdx.x;
    const int warpId = tid / 32;
    const int laneId = tid % 32;
    const int nSuper = K / NVBLK_SUPER_ELEMS;

    const float* __restrict__ xRow = X + static_cast<size_t>(row0) * K;

    const int n = blockIdx.x * NVBLK_DEINT_WARPS + warpId;
    if (n >= N) {
        return;
    }

    const size_t rowSupers = static_cast<size_t>(nSuper);
    const uint4* __restrict__ nibRow = reinterpret_cast<const uint4*>(
        nib + ((static_cast<size_t>(e) * N + n) * rowSupers) * 16);
    const __half* __restrict__ scRow =
        scale + ((static_cast<size_t>(e) * N + n) * rowSupers) * 2;

    // MAX_M == 1 for the decode dispatch: a single activation row, so no
    // per-row loop and the x reads go straight to global (L2).
    float acc = 0.0f;
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
            acc = __fmaf_rn(w0, xRow[k0 + e0],     acc);
            acc = __fmaf_rn(w1, xRow[k0 + e0 + 1], acc);
        }
    }
    const float s = warpReduceSum(acc);
    if (laneId == 0) {
        Y[static_cast<size_t>(row0) * N + n] = s;
    }
}

// ---------------------------------------------------------------------------
// Increment 1 — "untried quadrant": de-interleaved uint4 weight read + the
// activation in REGISTERS (no shared memory at all).
//
// The committed grouped deint kernel above stages x to shared (sxg) and reads
// sxg[laneId*32 + e] in the inner loop — a stride-32 access => 32-way bank
// conflict, which is why deint regressed vs m1reg despite the wider weight load.
// The m1reg kernel (moe_grouped_gemm_nvfp4blk_m1reg) removed shared memory but
// reads weights from the INTERLEAVED 20-byte super, whose base (sp*20) is not
// 16-byte aligned, so a warp's nibble read can split across two sectors.
//
// This kernel takes both wins at once, decode single-user (nSeq==1, one row):
//   * lane == element position within a super (like m1reg) => the activation
//     read Xt[sp*32 + laneId] is coalesced and lives in a register, never smem;
//   * the 16 nibble-bytes of each super are 16-byte aligned in the de-inter-
//     leaved bank, so the whole super loads as ONE uint4 broadcast to the warp
//     (a single clean sector). Each lane extracts its own nibble from the uint4
//     in registers.
// It isolates the sector/alignment effect from the smem-conflict penalty that
// dominated the earlier deint measurement. One output column per warp.
extern "C" __global__ __launch_bounds__(NVBLK_DEINT_LOCAL)
void moe_grouped_gemm_nvfp4blk_deint_m1reg(
    const float*         __restrict__ X,        // [R, K] compacted activations
    const unsigned char* __restrict__ nib,      // [nExperts, N, nSuper, 16]
    const __half*        __restrict__ scale,    // [nExperts, N, nSuper, 2]
          float*         __restrict__ Y,        // [R, N]
    const int*           __restrict__ tileExpert,
    const int*           __restrict__ tileRow0,
    const int*           __restrict__ /*tileRows*/,
    const int                          K,
    const int                          N)
{
    const int tt = blockIdx.y;
    const int e  = tileExpert[tt];
    if (e < 0) {
        return;
    }
    const int laneId = threadIdx.x % 32;
    const int warpId = threadIdx.x / 32;
    const int n      = blockIdx.x * NVBLK_DEINT_WARPS + warpId;
    if (n >= N) {
        return;                                    // no smem/sync: safe early exit
    }
    const int row0   = tileRow0[tt];
    const int nSuper = K / NVBLK_SUPER_ELEMS;

    const float*  __restrict__ Xt = X + static_cast<size_t>(row0) * K;
    float*        __restrict__ Yt = Y + static_cast<size_t>(row0) * N;
    const uint4*  __restrict__ nibRow = reinterpret_cast<const uint4*>(
        nib + ((static_cast<size_t>(e) * N + n) * nSuper) * 16);
    const __half* __restrict__ scRow =
        scale + ((static_cast<size_t>(e) * N + n) * nSuper) * 2;

    float acc = 0.0f;
    for (int sp = 0; sp < nSuper; ++sp) {
        // Activation element `laneId` of super sp — coalesced across the warp,
        // straight into a register (no shared memory).
        const float xe = Xt[static_cast<size_t>(sp) * NVBLK_SUPER_ELEMS + laneId];
        // Whole super's 16 nibble-bytes as one aligned uint4 (same address for
        // all lanes => a single broadcast sector); extract this lane's nibble.
        const uint4 q = nibRow[sp];
        const unsigned word = (laneId < 16) ? ((laneId < 8) ? q.x : q.y)
                                            : ((laneId < 24) ? q.z : q.w);
        const unsigned byte = (word >> (((laneId >> 1) & 3) * 8)) & 0xFFu;
        const unsigned nibv = (laneId & 1) ? (byte >> 4) : (byte & 0x0Fu);
        const float s = (laneId < 16) ? __half2float(scRow[sp * 2 + 0])
                                      : __half2float(scRow[sp * 2 + 1]);
        acc = __fmaf_rn(s * dq_e2m1(nibv), xe, acc);
    }

    const float s = warpReduceSum(acc);
    if (laneId == 0) {
        Yt[n] = s;
    }
}

// ---------------------------------------------------------------------------
// Increment 5 — DENSE lmhead/dense NVFP4 matvec, register-x + uint4 deint.
//
// nvblkDeintVec (GpuMatmul) drives the lmhead GEMV (248 k vocab) when
// MIMIRMIND_NVFP4_DEINT=1. Its v1 kernel (matmul_nvfp4blk_deint_vec above)
// stages x to shared and reads xTile[super*32+e] per FMA (lane==super) — the
// same MIO/shared-pipe stall Inc 1 removed for the grouped MoE GEMM, which is
// why deint gave the lmhead nothing. This variant is the dense analogue of the
// grouped Inc-1 kernel: lane == element position (like matmul_nvfp4blk_vec), x
// coalesced into a register (no shared memory), the super's 16 nibble-bytes read
// as one aligned uint4 broadcast to the warp. Accumulation order (per-super,
// then warp-reduce over the 32 element-lanes) is IDENTICAL to matmul_nvfp4blk_vec
// -> bit-identical logits -> identical argmax/tokens.
extern "C" __global__ __launch_bounds__(NVBLK_DEINT_LOCAL)
void matmul_nvfp4blk_deint_reg_vec(
    const float*         __restrict__ X,        // [K] activation
    const unsigned char* __restrict__ nib,      // [N, nSuper, 16]
    const __half*        __restrict__ scale,    // [N, nSuper, 2]
          float*         __restrict__ Y,        // [N]
    const int                          N,
    const int                          K)
{
    const int laneId = threadIdx.x % 32;
    const int warpId = threadIdx.x / 32;
    const int n      = blockIdx.x * NVBLK_DEINT_WARPS + warpId;
    if (n >= N) {
        return;
    }
    const int nSuper = K / NVBLK_SUPER_ELEMS;
    const uint4*  __restrict__ nibRow =
        reinterpret_cast<const uint4*>(nib + static_cast<size_t>(n) * nSuper * 16);
    const __half* __restrict__ scRow = scale + static_cast<size_t>(n) * nSuper * 2;

    float acc = 0.0f;
    for (int sp = 0; sp < nSuper; ++sp) {
        const float xe = X[static_cast<size_t>(sp) * NVBLK_SUPER_ELEMS + laneId];
        const uint4 q = nibRow[sp];
        const unsigned word = (laneId < 16) ? ((laneId < 8) ? q.x : q.y)
                                            : ((laneId < 24) ? q.z : q.w);
        const unsigned byte = (word >> (((laneId >> 1) & 3) * 8)) & 0xFFu;
        const unsigned nibv = (laneId & 1) ? (byte >> 4) : (byte & 0x0Fu);
        const float s = (laneId < 16) ? __half2float(scRow[sp * 2 + 0])
                                      : __half2float(scRow[sp * 2 + 1]);
        acc = __fmaf_rn(s * dq_e2m1(nibv), xe, acc);
    }
    const float sred = warpReduceSum(acc);
    if (laneId == 0) {
        Y[n] = sred;
    }
}
