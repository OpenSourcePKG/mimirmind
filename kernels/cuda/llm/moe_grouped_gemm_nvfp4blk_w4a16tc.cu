// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// W4A16 Tensor-Core grouped GEMM over blocked-NVFP4 expert weights
// (M-Cuda.MoeGroup — Bragi 5.7, Marlin-style TC decode-MoE-GEMM, Increment I0).
//
//   Y[row, n] = sum_k xCompact[row, k] * (scale_block * e2m1(W[e][n, k]))
//
// Same math + same grouped-tile schedule as moe_grouped_gemm_nvfp4blk (the
// CUDA-Core golden), but the K-contraction runs on BF16 tensor cores instead of
// scalar FMA. This is the "W4A16" path: the 4-bit NVFP4 WEIGHT is dequantised to
// BF16 in shared memory, the 16-bit ACTIVATION stays BF16 (NO activation quant —
// exactly what lets Marlin avoid the per-step act-quant + swizzled-SFB-scatter
// overhead that made the W4A4 FP4-TC decode (GD-c) lose in serving).
//
// Per (tile, N-slab of 16 columns) one WARP does one 16x16x16 wmma output tile:
//   * dequant W[e][n0:n0+16, k0:k0+16] -> shared BF16 wB[16][16] (matrix_b, k-major)
//   * load    X[row0:row0+M, k0:k0+16] -> shared BF16 xB[16][16] (matrix_a, m-major)
//   * wmma::mma_sync accumulate in F32; store the 16x16 result to Y.
// TC_K=16 == exactly one group-16 NVFP4 scale half-block, so each K-tile applies
// a single scale per (n) — clean, no cross-block scale mixing.
//
// Weight bank layout (see matmul_nvfp4blk_vec / moe_grouped_gemm_nvfp4blk): the
// contiguous [nExperts][N][K] blocked bank has, per (e,n) row, nSuper=K/32
// super-blocks of NVBLK_SUPER_BYTES=20 bytes each: [s0:fp16][s1:fp16][16 bytes =
// 32 E2M1 nibbles]. s0 scales the first 16 K-elements of the super, s1 the next
// 16. Nibble for element `elem` (0..31): byte = qs[elem>>1], nib = (elem&1)?hi:lo.
//
// Numerics: BF16 tensor cores + F32 accum -> ~bf16 relL2 vs the F32 golden (same
// regime as matmul_bf16_gemm_tc, ~0.2%). NOT bit-identical by design.
//
// Launch: grid.x = ceil(N / TC_N), grid.y = maxTiles; block = 32 (one warp).
// Over-provisioned schedule slots carry tileExpert == -1 and early-exit.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>

using namespace nvcuda;

#ifndef MOE_W4A16TC_LOCAL
#define MOE_W4A16TC_LOCAL 32          // one warp per (tile, N-slab)
#endif

#define NVBLK_SUPER_ELEMENTS 32
#define NVBLK_SUPER_BYTES    20
#define TC_M 16
#define TC_N 16
#define TC_K 16

namespace {

__device__ __forceinline__ float dq_e2m1(unsigned nib) {
    const float mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float v = mag[nib & 0x7u];
    return (nib & 0x8u) ? -v : v;
}

} // namespace

extern "C" __global__ __launch_bounds__(MOE_W4A16TC_LOCAL)
void moe_grouped_gemm_nvfp4blk_w4a16tc(
    const float*         __restrict__ X,           // [R, K]  compacted activations (f32)
    const unsigned char* __restrict__ W,           // [nExperts, N, K] blocked NVFP4
          float*         __restrict__ Y,           // [R, N]  grouped output (f32)
    const int*           __restrict__ tileExpert,  // [maxTiles] expert id / -1
    const int*           __restrict__ tileRow0,    // [maxTiles] start row
    const int*           __restrict__ tileRows,    // [maxTiles] row count (1..TC_M)
    const int                          K,
    const int                          N)
{
    const int tt = blockIdx.y;
    const int e  = tileExpert[tt];
    if (e < 0) {
        return;                                    // unused (over-provisioned) tile
    }
    const int row0 = tileRow0[tt];
    int M          = tileRows[tt];
    if (M > TC_M) {
        M = TC_M;
    }

    const int n0     = blockIdx.x * TC_N;          // this warp's N-slab start column
    const int lane   = threadIdx.x;                // 0..31 (single warp)
    const int nSuper = K / NVBLK_SUPER_ELEMENTS;

    __shared__ __nv_bfloat16 xB[TC_M][TC_K];       // activation tile (matrix_a)
    __shared__ __nv_bfloat16 wB[TC_K][TC_N];       // dequant weight tile (matrix_b)
    __shared__ float         cTile[TC_M][TC_N];

    wmma::fragment<wmma::accumulator, TC_M, TC_N, TC_K, float> cFrag;
    wmma::fill_fragment(cFrag, 0.0f);

    const float*         __restrict__ Xt   = X + static_cast<size_t>(row0) * K;
    const unsigned char* __restrict__ Wexp =
        W + static_cast<size_t>(e) * static_cast<size_t>(N)
          * static_cast<size_t>(nSuper) * NVBLK_SUPER_BYTES;

    for (int k0 = 0; k0 < K; k0 += TC_K) {
        // ---- activation tile: xB[m][kk] = bf16(X[row0+m][k0+kk]), pad rows >= M ----
        for (int idx = lane; idx < TC_M * TC_K; idx += 32) {
            const int m  = idx / TC_K;
            const int kk = idx % TC_K;
            const float v = (m < M) ? Xt[static_cast<size_t>(m) * K + k0 + kk] : 0.0f;
            xB[m][kk] = __float2bfloat16(v);
        }

        // ---- weight tile: wB[kk][nn] = bf16(scale * e2m1(W[e][n0+nn][k0+kk])) ----
        // super index of this K-tile and which fp16 scale half it maps to.
        const int sp     = k0 / NVBLK_SUPER_ELEMENTS;
        const int base16 = k0 % NVBLK_SUPER_ELEMENTS;   // 0 -> s0 half, 16 -> s1 half
        for (int idx = lane; idx < TC_K * TC_N; idx += 32) {
            const int kk = idx / TC_N;
            const int nn = idx % TC_N;
            const int n  = n0 + nn;
            float wv = 0.0f;
            if (n < N && sp < nSuper) {
                const unsigned char* blk =
                    Wexp + (static_cast<size_t>(n) * nSuper + sp) * NVBLK_SUPER_BYTES;
                const __half* sh = reinterpret_cast<const __half*>(blk);
                const float scale = __half2float(base16 ? sh[1] : sh[0]);
                const int elem = base16 + kk;           // 0..31 within the super-block
                const unsigned char byte = blk[4 + (elem >> 1)];
                const unsigned nib = (elem & 1) ? (byte >> 4) : (byte & 0x0F);
                wv = scale * dq_e2m1(nib);
            }
            wB[kk][nn] = __float2bfloat16(wv);
        }
        __syncwarp();

        wmma::fragment<wmma::matrix_a, TC_M, TC_N, TC_K, __nv_bfloat16, wmma::row_major> aF;
        wmma::fragment<wmma::matrix_b, TC_M, TC_N, TC_K, __nv_bfloat16, wmma::row_major> bF;
        wmma::load_matrix_sync(aF, &xB[0][0], TC_K);
        wmma::load_matrix_sync(bF, &wB[0][0], TC_N);
        wmma::mma_sync(cFrag, aF, bF, cFrag);
        __syncwarp();
    }

    wmma::store_matrix_sync(&cTile[0][0], cFrag, TC_N, wmma::mem_row_major);
    __syncwarp();

    float* __restrict__ Yt = Y + static_cast<size_t>(row0) * N;
    for (int idx = lane; idx < TC_M * TC_N; idx += 32) {
        const int m  = idx / TC_N;
        const int nn = idx % TC_N;
        const int n  = n0 + nn;
        if (m < M && n < N) {
            Yt[static_cast<size_t>(m) * N + n] = cTile[m][nn];
        }
    }
}
