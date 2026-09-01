// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Wide-M FP16 tensor-core grouped GEMM over blocked-NVFP4 expert weights
// (roadmap 5.21.6 Increment I/I.b) — the prefill-throughput replacement for the
// CUDA-core moe_grouped_gemm_nvfp4blk (MAX_M=16). Same schedule-driven contract
// and math:
//
//   Y[row, n] = sum_k X[row, k] * (scale_block * e2m1(W[e][n, k]))
//
// W4A16 (Marlin-style): the 4-bit weight is read from DRAM at 4-bit (the
// bandwidth wall stays on the 4-bit read), dequantised to fp16 in shared memory,
// X is cast fp32->fp16 in shared memory, and 16x16x16 wmma mmas accumulate in
// fp32. The grouped schedule emits tileM<=16 -> the wmma m=16 tile is fully
// filled at prefill (vs 1/16 at decode = why decode TC is NO-GO, prefill GO).
//
// I.b load pipeline (the ~7x over the naive nibble-scatter version):
//   * arithmetic dq_e2m1 (NO mag[8] table -> no local-memory spill per decode);
//   * process a full 32-element super PER round (2 wmma K-sub-steps), so each
//     20-byte super is read EXACTLY ONCE per weight row;
//   * a 16-wide K-half lands in one super-half -> ONE uniform fp16 scale + 8
//     contiguous nibble bytes read as two aligned uint32 (no 1-byte loads, no
//     redundant header re-reads). 32 lanes = 16 rows x 2 halves per round.
//
// Numerically NOT bit-identical to the fp32 CUDA-core kernel (fp16 rounding of X
// and of scale*e2m1); validated to L2-relative tolerance in moe_gemm_tc_bench.
//
// One CTA = TC_WARPS warps sharing the tile's A (X); each warp owns a 16-wide
// N-block. Launch: grid = dim3(ceil(N/(16*TC_WARPS)), maxTiles, 1),
//                  block = 32 * TC_WARPS. Requires K % 32 == 0 (real MoE dims).

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>

using namespace nvcuda;

#define NVBLK_SUPER_ELEMENTS 32
#define NVBLK_SUPER_BYTES    20
#define TC_TILE              16   // wmma m=n=k=16
#define TC_KBLK              32   // one super processed per round (2 K-sub-steps)

#ifndef TC_WARPS
#define TC_WARPS 4                // n-blocks handled per CTA
#endif

namespace {

// Bit-exact e2m1 magnitude decode WITHOUT a runtime-indexed local table
// (a float mag[8] indexed by a divergent value spills to local memory). Table:
// {0,0.5,1,1.5,2,3,4,6}; normal path (mm>1) = 2^((mm>>1)-1)*(1+0.5*(mm&1)).
__device__ __forceinline__ float dq_e2m1(unsigned nib) {
    const unsigned mm   = nib & 0x7u;
    const unsigned bits = (((mm >> 1) + 126u) << 23) | ((mm & 1u) << 22);
    float v = __uint_as_float(bits);
    v = (mm < 2u) ? (0.5f * static_cast<float>(mm)) : v;
    return (nib & 0x8u) ? -v : v;
}

} // namespace

extern "C" __global__ __launch_bounds__(TC_WARPS * 32)
void moe_grouped_gemm_nvfp4blk_tc(
    const float*         __restrict__ X,           // [R, K] compacted activations
    const unsigned char* __restrict__ W,           // [nExperts, N, K] blocked NVFP4
          float*         __restrict__ Y,           // [R, N] grouped output
    const int*           __restrict__ tileExpert,  // [maxTiles] expert id / -1
    const int*           __restrict__ tileRow0,    // [maxTiles] start row
    const int*           __restrict__ tileRows,    // [maxTiles] row count (1..16)
    const int                          K,
    const int                          N)
{
    const int tt = blockIdx.y;
    const int e  = tileExpert[tt];
    if (e < 0) {
        return;                                    // over-provisioned tile
    }
    const int row0 = tileRow0[tt];
    int       M    = tileRows[tt];
    if (M > TC_TILE) M = TC_TILE;
    const int nSuper   = K / NVBLK_SUPER_ELEMENTS;
    const int rowBytes = nSuper * NVBLK_SUPER_BYTES;
    const unsigned char* __restrict__ Wexp =
        W + static_cast<size_t>(e) * N * static_cast<size_t>(rowBytes);

    const int warp = threadIdx.x >> 5;             // 0..TC_WARPS-1
    const int lane = threadIdx.x & 31;
    const int n0   = (blockIdx.x * TC_WARPS + warp) * TC_TILE;   // this warp's N-block

    // A shared by the whole CTA: xs[m][k] for a 32-wide K block, ldm=TC_KBLK.
    __shared__ __half xs[TC_TILE * TC_KBLK];
    // B per warp: ws[k][n] for a 32-wide K block, ldm=TC_TILE.
    __shared__ __half ws[TC_WARPS][TC_KBLK * TC_TILE];
    __shared__ float  cs[TC_WARPS][TC_TILE * TC_TILE];

    // Weight-staging role: 32 lanes = 16 rows x 2 super-halves.
    const int wRow  = lane & 15;                   // 0..15  (n column within block)
    const int wHalf = lane >> 4;                   // 0/1    (super half)
    const bool wRowValid = (n0 + wRow) < N;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cF;
    wmma::fill_fragment(cF, 0.0f);

    for (int sp = 0; sp < nSuper; ++sp) {
        const int k0 = sp * TC_KBLK;
        // Stage A once for the CTA: xs[m][kk] = fp16(X[row0+m][k0+kk]).
        for (int i = threadIdx.x; i < TC_TILE * TC_KBLK; i += blockDim.x) {
            const int m  = i / TC_KBLK;
            const int kk = i % TC_KBLK;
            xs[i] = __float2half((m < M)
                ? X[static_cast<size_t>(row0 + m) * K + (k0 + kk)]
                : 0.0f);
        }
        // Stage B per warp: each lane dequantises its (row, half) = 16 elements
        // via aligned loads (one fp16 scale + two uint32 of nibble bytes).
        {
            float scale = 0.0f; uint32_t u0 = 0, u1 = 0;
            if (wRowValid) {
                const unsigned char* blk =
                    Wexp + static_cast<size_t>(n0 + wRow) * rowBytes + sp * NVBLK_SUPER_BYTES;
                scale = __half2float(*reinterpret_cast<const __half*>(blk + wHalf * 2));
                const uint32_t* q = reinterpret_cast<const uint32_t*>(blk + 4 + wHalf * 8);
                u0 = q[0]; u1 = q[1];
            }
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                const uint32_t word = (j < 8) ? u0 : u1;
                const unsigned byte  = (word >> (8 * ((j & 7) >> 1))) & 0xFFu;
                const unsigned nib   = (j & 1) ? (byte >> 4) : (byte & 0x0F);
                const int      kk    = wHalf * 16 + j;   // k within the 32-block
                ws[warp][kk * TC_TILE + wRow] =
                    __float2half(wRowValid ? scale * dq_e2m1(nib) : 0.0f);
            }
        }
        __syncthreads();

        // Two wmma K-sub-steps over the 32-wide block.
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> aF;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> bF;
        #pragma unroll
        for (int s = 0; s < 2; ++s) {
            wmma::load_matrix_sync(aF, xs + s * TC_TILE, TC_KBLK);
            wmma::load_matrix_sync(bF, ws[warp] + s * TC_TILE * TC_TILE, TC_TILE);
            wmma::mma_sync(cF, aF, bF, cF);
        }
        __syncthreads();
    }

    wmma::store_matrix_sync(cs[warp], cF, TC_TILE, wmma::mem_row_major);
    __syncwarp();
    for (int i = lane; i < TC_TILE * TC_TILE; i += 32) {
        const int m = i / TC_TILE;
        const int n = i % TC_TILE;
        if (m < M && (n0 + n) < N) {
            Y[static_cast<size_t>(row0 + m) * N + (n0 + n)] = cs[warp][i];
        }
    }
}
