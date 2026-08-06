// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Device-driven grouped GEMM over blocked-NVFP4 expert weights
// (M-Cuda.MoeGroup, Sub-Step E-b) — the kernel that turns the host-driven
// Option 1 (per-expert host launch loop + one expOffset D2H per MoE layer)
// into a SINGLE device-driven launch, the only shape that can beat the
// fused-K batched path on GB10.
//
//   Y[row, n] = sum_k xCompact[row, k] * (scale_block * e2m1(W[e][n, k]))
//
// where the compacted rows are grouped by expert (moe_gather_rows), and a
// compact per-tile schedule (moe_group_tiles) assigns each grid.y block one
// tile = (expert e, absolute row range [row0, row0+rows), rows <= tileM). The
// block reads its expert + row range from the schedule ON THE DEVICE, so
// nothing crosses to the host: no expOffset D2H, no per-expert launch loop.
// The whole per-expert loop collapses into one launch per projection.
//
// This is the exact math of matmul_nvfp4blk_gemm (see it + matmul_nvfp4blk_vec
// for the 20-byte / 32-element super-block layout), generalised two ways:
//   * grid.y indexes the tile schedule instead of a fixed single problem, so
//     one launch covers every expert's rows;
//   * the weight base is offset by the tile's expert e into the contiguous
//     [nExperts][N][K] blocked bank, and X/Y are offset by the tile's row0.
// Each weight row is still read once and reused across the tile's <= tileM
// activation rows (tileM == GEMM_MAX_M == 16, the schedule guarantees it).
//
// Empty schedule slots carry tileExpert == -1 (the host over-provisions
// grid.y to the static upper bound maxTiles); those blocks early-exit. A tile
// always has rows >= 1 (moe_group_tiles never emits an empty tile), so no
// active block does zero work.
//
// Launch: grid.x = ceil(N / OUTPUTS_PER_GROUP), grid.y = maxTiles;
//         block  = MATMUL_NVBLK_GEMM_LOCAL (128).
//
// GD-b (M-Cuda.MoeGroup-Decode): the kernel body is templated on MAX_M and
// emitted twice via extern-C wrappers:
//   * moe_grouped_gemm_nvfp4blk     (MAX_M = 16) — prefill / large-M grouped.
//   * moe_grouped_gemm_nvfp4blk_m4  (MAX_M = 4)  — decode small-M.
// At decode the schedule (moe_group_tiles, tileM=4) emits <=4-row tiles, so the
// M4 variant needs only xTile[4*256]=4 KB shared + acc[4] regs instead of 16 KB
// + acc[16]. ncu on the M16 kernel at decode-M showed occupancy capped at
// ~41.7% by shared memory (Block Limit Shared Mem = 5), with ~72% of cycles
// stalled on scoreboard (L1TEX/shared) — i.e. latency-bound at low occupancy.
// The M4 variant lifts both the shared- and register-occupancy limits so the
// hardware has enough warps to hide that memory latency at small batch. This
// is the reference (DeepGEMM/SGLang) "adaptive occupancy at small batch" fix.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifndef MATMUL_NVBLK_GEMM_LOCAL
#define MATMUL_NVBLK_GEMM_LOCAL 128
#endif

#define MATMUL_NVBLK_GEMM_WARPS             (MATMUL_NVBLK_GEMM_LOCAL / 32)
#define MATMUL_NVBLK_GEMM_OUTPUTS_PER_GROUP MATMUL_NVBLK_GEMM_WARPS

#define NVBLK_SUPER_ELEMENTS 32
#define NVBLK_SUPER_BYTES    20
#define GEMM_X_TILE  256   // 8 supers

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

template <int MAX_M>
__device__ __forceinline__ void grouped_gemm_nvfp4blk_core(
    const float*         __restrict__ X,           // [R, K]  compacted activations
    const unsigned char* __restrict__ W,           // [nExperts, N, K] blocked NVFP4
          float*         __restrict__ Y,           // [R, N]  grouped output
    const int*           __restrict__ tileExpert,  // [maxTiles] expert id / -1
    const int*           __restrict__ tileRow0,    // [maxTiles] start row
    const int*           __restrict__ tileRows,    // [maxTiles] row count (1..MAX_M)
    const int                          K,
    const int                          N)
{
    __shared__ float xTile[MAX_M * GEMM_X_TILE];

    const int tt = blockIdx.y;                     // tile index into the schedule
    const int e  = tileExpert[tt];
    if (e < 0) {
        return;                                    // unused (over-provisioned) tile
    }
    const int row0 = tileRow0[tt];
    int M          = tileRows[tt];
    if (M > MAX_M) {
        M = MAX_M;                                 // defensive; schedule caps at tileM
    }

    const int tid    = threadIdx.x;
    const int lsize  = blockDim.x;
    const int warpId = tid / 32;
    const int laneId = tid % 32;
    const int n      = blockIdx.x * MATMUL_NVBLK_GEMM_OUTPUTS_PER_GROUP + warpId;
    const bool active = (n < N);
    const int nSuper = K / NVBLK_SUPER_ELEMENTS;

    // Expert weight base into the contiguous [nExperts][N][K] blocked bank,
    // and the tile's activation / output row bases.
    const unsigned char* __restrict__ Wexp =
        W + static_cast<size_t>(e) * static_cast<size_t>(N)
          * static_cast<size_t>(nSuper) * NVBLK_SUPER_BYTES;
    const float* __restrict__ Xt = X + static_cast<size_t>(row0) * K;
    float*       __restrict__ Yt = Y + static_cast<size_t>(row0) * N;

    float acc[MAX_M];
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;

    for (int tile = 0; tile < K; tile += GEMM_X_TILE) {
        const int tileK = min(GEMM_X_TILE, K - tile);
        for (int i = tid; i < M * tileK; i += lsize) {
            const int m  = i / tileK;
            const int kk = i % tileK;
            xTile[m * GEMM_X_TILE + kk] = Xt[static_cast<size_t>(m) * K + tile + kk];
        }
        __syncthreads();

        if (active) {
            const unsigned char* row =
                Wexp + static_cast<size_t>(n) * static_cast<size_t>(nSuper)
                     * NVBLK_SUPER_BYTES;
            const int superStart   = tile / NVBLK_SUPER_ELEMENTS;
            const int supersInTile = GEMM_X_TILE / NVBLK_SUPER_ELEMENTS;
            const int superEnd     = min(superStart + supersInTile, nSuper);
            for (int sp = superStart; sp < superEnd; ++sp) {
                const unsigned char* blk = row + sp * NVBLK_SUPER_BYTES;
                const float s0 = __half2float(*reinterpret_cast<const __half*>(blk));
                const float s1 = __half2float(*reinterpret_cast<const __half*>(blk + 2));
                const unsigned char* qs = blk + 4;
                const unsigned char byte = qs[laneId >> 1];
                const unsigned nib = (laneId & 1) ? (byte >> 4) : (byte & 0x0F);
                const float w = ((laneId < 16) ? s0 : s1) * dq_e2m1(nib);
                const int kk = (sp - superStart) * NVBLK_SUPER_ELEMENTS + laneId;
                for (int m = 0; m < M; ++m) {
                    acc[m] = __fmaf_rn(w, xTile[m * GEMM_X_TILE + kk], acc[m]);
                }
            }
        }
        __syncthreads();
    }

    for (int m = 0; m < M; ++m) {
        const float s = warpReduceSum(acc[m]);
        if (active && laneId == 0) {
            Yt[static_cast<size_t>(m) * N + n] = s;
        }
    }
}

// Decode single-row (M==1) core with the activation staged in REGISTERS, not
// shared memory. ncu on the m4 kernel showed decode stalls ~56-61% on "short
// scoreboard / waiting on data TO SHARED MEMORY" at full occupancy and only
// ~24% DRAM: the 1-smem-load-per-FMA activation read is the bottleneck, and it
// is a latency (not throughput) stall already at the occupancy ceiling — so
// N-blocking (fewer smem loads at the cost of more registers/less occupancy)
// regressed. This variant removes shared memory entirely: per K-tile each lane
// loads its GEMM_X_TILE/32 activation elements (one per super, at element
// position laneId) into registers with a coalesced global (L2-resident) load,
// then the inner loop reads x from registers. No smem, no __syncthreads, no MIO
// stall; only ~8 extra registers so occupancy is preserved. One output column
// per warp (like m4). Assumes one activation row per tile (nSeq==1 decode).
#define M1REG_SUPERS_PER_TILE (GEMM_X_TILE / NVBLK_SUPER_ELEMENTS)   // 8
__device__ __forceinline__ void grouped_gemm_nvfp4blk_core_m1reg(
    const float*         __restrict__ X,           // [R, K]  (row0 is the row)
    const unsigned char* __restrict__ W,           // [nExperts, N, K] blocked NVFP4
          float*         __restrict__ Y,           // [R, N]
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
    const int n      = blockIdx.x * MATMUL_NVBLK_GEMM_WARPS + warpId;
    if (n >= N) {
        return;                                    // no smem/sync: safe early exit
    }
    const int row0   = tileRow0[tt];
    const int nSuper = K / NVBLK_SUPER_ELEMENTS;

    const float*         __restrict__ Xt = X + static_cast<size_t>(row0) * K;
    float*               __restrict__ Yt = Y + static_cast<size_t>(row0) * N;
    const unsigned char* __restrict__ Wrow =
        W + ((static_cast<size_t>(e) * N + n) * nSuper) * NVBLK_SUPER_BYTES;

    float acc = 0.0f;
    for (int tile = 0; tile < K; tile += GEMM_X_TILE) {
        // Stage this lane's activation elements (one per super in the tile) into
        // registers; lanes 0..31 read 32 consecutive floats per super => coalesced.
        float xr[M1REG_SUPERS_PER_TILE];
#pragma unroll
        for (int j = 0; j < M1REG_SUPERS_PER_TILE; ++j) {
            const int k = tile + j * NVBLK_SUPER_ELEMENTS + laneId;
            xr[j] = (k < K) ? Xt[k] : 0.0f;
        }
        const int superStart = tile / NVBLK_SUPER_ELEMENTS;
#pragma unroll
        for (int j = 0; j < M1REG_SUPERS_PER_TILE; ++j) {
            const int sp = superStart + j;
            if (sp >= nSuper) {
                break;
            }
            const unsigned char* blk =
                Wrow + static_cast<size_t>(sp) * NVBLK_SUPER_BYTES;
            const float s0 = __half2float(*reinterpret_cast<const __half*>(blk));
            const float s1 = __half2float(*reinterpret_cast<const __half*>(blk + 2));
            const unsigned char byte = blk[4 + (laneId >> 1)];
            const unsigned nib = (laneId & 1) ? (byte >> 4) : (byte & 0x0F);
            const float w = ((laneId < 16) ? s0 : s1) * dq_e2m1(nib);
            acc = __fmaf_rn(w, xr[j], acc);
        }
    }

    const float s = warpReduceSum(acc);
    if (laneId == 0) {
        Yt[n] = s;
    }
}

} // namespace

// M16 — prefill / large-M grouped GEMM (16 KB shared, acc[16]).
extern "C" __global__ __launch_bounds__(MATMUL_NVBLK_GEMM_LOCAL)
void moe_grouped_gemm_nvfp4blk(
    const float* __restrict__ X, const unsigned char* __restrict__ W,
    float* __restrict__ Y, const int* __restrict__ tileExpert,
    const int* __restrict__ tileRow0, const int* __restrict__ tileRows,
    const int K, const int N)
{
    grouped_gemm_nvfp4blk_core<16>(X, W, Y, tileExpert, tileRow0, tileRows, K, N);
}

// M4 — decode small-M grouped GEMM (4 KB shared, acc[4]); lifts the shared- and
// register-occupancy limits ncu flagged on the M16 kernel at decode-M.
extern "C" __global__ __launch_bounds__(MATMUL_NVBLK_GEMM_LOCAL)
void moe_grouped_gemm_nvfp4blk_m4(
    const float* __restrict__ X, const unsigned char* __restrict__ W,
    float* __restrict__ Y, const int* __restrict__ tileExpert,
    const int* __restrict__ tileRow0, const int* __restrict__ tileRows,
    const int K, const int N)
{
    grouped_gemm_nvfp4blk_core<4>(X, W, Y, tileExpert, tileRow0, tileRows, K, N);
}

// M1-REG — decode single-user (M==1) grouped GEMM with the activation staged in
// registers instead of shared memory, removing the ncu-measured MIO/short-
// scoreboard stall without spending occupancy. Dispatched only at nSeq==1.
extern "C" __global__ __launch_bounds__(MATMUL_NVBLK_GEMM_LOCAL)
void moe_grouped_gemm_nvfp4blk_m1reg(
    const float* __restrict__ X, const unsigned char* __restrict__ W,
    float* __restrict__ Y, const int* __restrict__ tileExpert,
    const int* __restrict__ tileRow0, const int* __restrict__ tileRows,
    const int K, const int N)
{
    grouped_gemm_nvfp4blk_core_m1reg(X, W, Y, tileExpert, tileRow0, tileRows, K, N);
}
