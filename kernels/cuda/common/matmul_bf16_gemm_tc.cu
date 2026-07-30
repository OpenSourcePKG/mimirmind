// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Tensor-core variant of matmul_bf16_gemm (serving decode dense GEMM).
//
//   Y[m, n] = sum_{k} X[m, k] * bf16_to_f32(W[n, k])
//
//   X:  [M, K]  F32 dense activations (M = batch = nSeq)
//   W:  [N, K]  BF16 dense weights, row-major (2 bytes / element)
//   Y:  [M, N]  F32 dense output
//
// Same math/shape as matmul_bf16_gemm (the scalar warp-per-column kernel),
// but the K-contraction runs on BF16 tensor cores (wmma 16x16x16, F32 accum).
// The activations are rounded F32 -> BF16 when staged into shared memory (the
// standard quality-safe BF16-TC path; not bit-exact vs the scalar F32 kernel,
// ~0.2% relL2). BF16 wmma is a compute_80 feature — no sm_120a/121a flag.
//
// One warp computes one 16x16 output tile. W[N,K] row-major is transposed into
// a [K,N] shared tile so the wmma matrix_b (row_major) sees B[k,n] = W[n,k].
// M/N/K need not be multiples of 16 — out-of-range rows/cols are zero-padded on
// stage and dropped on store.
//
// Launch:
//   dim3 grid ( ceil(N / 16), ceil(M / 16), 1 )
//   dim3 block( 32, 1, 1 )

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>

using namespace nvcuda;

#define TC_M 16
#define TC_N 16
#define TC_K 16

extern "C" __global__ __launch_bounds__(32)
void matmul_bf16_gemm_tc(
    const float*          __restrict__ X,   // [M, K]
    const __nv_bfloat16*  __restrict__ W,   // [N, K]
          float*          __restrict__ Y,   // [M, N]
    const int                          K,
    const int                          N,
    const int                          M)
{
    const int lane  = threadIdx.x;
    const int mBase = blockIdx.y * TC_M;
    const int nBase = blockIdx.x * TC_N;

    __shared__ __nv_bfloat16 xB[TC_M][TC_K];   // [M, K] activation tile
    __shared__ __nv_bfloat16 wB[TC_K][TC_N];   // [K, N] transposed weight tile
    __shared__ float         cTile[TC_M][TC_N];

    wmma::fragment<wmma::accumulator, TC_M, TC_N, TC_K, float> cFrag;
    wmma::fill_fragment(cFrag, 0.0f);

    for (int kBase = 0; kBase < K; kBase += TC_K) {
        // Stage X[mBase.., kBase..] -> xB (F32 -> BF16, zero-pad OOB).
        for (int i = lane; i < TC_M * TC_K; i += 32) {
            const int m  = i / TC_K;
            const int kk = i % TC_K;
            const int mA = mBase + m;
            const int kA = kBase + kk;
            xB[m][kk] = __float2bfloat16(
                (mA < M && kA < K) ? X[static_cast<size_t>(mA) * K + kA] : 0.0f);
        }
        // Stage W[nBase.., kBase..] transposed -> wB[k][n] (W[N,K] -> B[K,N]).
        for (int i = lane; i < TC_K * TC_N; i += 32) {
            const int kk = i / TC_N;
            const int nn = i % TC_N;
            const int nA = nBase + nn;
            const int kA = kBase + kk;
            wB[kk][nn] = (nA < N && kA < K)
                             ? W[static_cast<size_t>(nA) * K + kA]
                             : __float2bfloat16(0.0f);
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, TC_M, TC_N, TC_K, __nv_bfloat16, wmma::row_major> aF;
        wmma::fragment<wmma::matrix_b, TC_M, TC_N, TC_K, __nv_bfloat16, wmma::row_major> bF;
        wmma::load_matrix_sync(aF, &xB[0][0], TC_K);
        wmma::load_matrix_sync(bF, &wB[0][0], TC_N);
        wmma::mma_sync(cFrag, aF, bF, cFrag);
        __syncthreads();
    }

    wmma::store_matrix_sync(&cTile[0][0], cFrag, TC_N, wmma::mem_row_major);
    __syncthreads();

    for (int i = lane; i < TC_M * TC_N; i += 32) {
        const int m  = i / TC_N;
        const int n  = i % TC_N;
        const int mA = mBase + m;
        const int nA = nBase + n;
        if (mA < M && nA < N) {
            Y[static_cast<size_t>(mA) * N + nA] = cTile[m][n];
        }
    }
}
