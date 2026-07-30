// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// TF32 tensor-core variant of matmul_bf16_gemm_tc (serving decode dense GEMM).
//
//   Y[m, n] = sum_{k} X[m, k] * bf16_to_f32(W[n, k])
//
//   X:  [M, K]  F32 dense activations (M = batch = nSeq)
//   W:  [N, K]  BF16 dense weights, row-major (2 bytes / element)
//   Y:  [M, N]  F32 dense output
//
// Same math/shape as matmul_bf16_gemm_tc, but the K-contraction runs on TF32
// tensor cores instead of BF16. The point is FIDELITY, not speed: the F32
// activations are rounded to TF32 (10-bit mantissa) on stage instead of BF16
// (7-bit), so the matmul tracks the scalar F32 reference more closely. (The
// weights are already BF16 in memory — 7-bit — so TF32 cannot add precision
// there; only the activation rounding improves. That is the noise source the
// E-FP4.3 near-tie flip was traced to.)
//
// TF32 wmma is m16n16k8 (half the K of BF16's k16). To keep the K-loop's
// staging/sync overhead the same as the BF16 kernel, each iteration stages a
// 16-wide K tile and issues TWO k8 mma_sync calls over it.
//
// TF32 wmma is a compute_80 feature — no sm_120a/121a flag.
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
#define TC_K 8         // TF32 mma contraction (m16n16k8)
#define TC_KT 16       // K tile staged per loop iteration = 2 * TC_K

extern "C" __global__ __launch_bounds__(32)
void matmul_bf16_gemm_tf32_tc(
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

    __shared__ float xB[TC_M][TC_KT];   // [M, K] activation tile (F32)
    __shared__ float wB[TC_KT][TC_N];   // [K, N] transposed weight tile (F32)
    __shared__ float cTile[TC_M][TC_N];

    wmma::fragment<wmma::accumulator, TC_M, TC_N, TC_K, float> cFrag;
    wmma::fill_fragment(cFrag, 0.0f);

    for (int kBase = 0; kBase < K; kBase += TC_KT) {
        // Stage X[mBase.., kBase..] -> xB (F32, zero-pad OOB).
        for (int i = lane; i < TC_M * TC_KT; i += 32) {
            const int m  = i / TC_KT;
            const int kk = i % TC_KT;
            const int mA = mBase + m;
            const int kA = kBase + kk;
            xB[m][kk] = (mA < M && kA < K) ? X[static_cast<size_t>(mA) * K + kA]
                                           : 0.0f;
        }
        // Stage W[nBase.., kBase..] transposed -> wB[k][n] (BF16 -> F32).
        for (int i = lane; i < TC_KT * TC_N; i += 32) {
            const int kk = i / TC_N;
            const int nn = i % TC_N;
            const int nA = nBase + nn;
            const int kA = kBase + kk;
            wB[kk][nn] = (nA < N && kA < K)
                             ? __bfloat162float(W[static_cast<size_t>(nA) * K + kA])
                             : 0.0f;
        }
        __syncthreads();

        // Two k8 TF32 mma over the 16-wide K tile.
        wmma::fragment<wmma::matrix_a, TC_M, TC_N, TC_K, wmma::precision::tf32, wmma::row_major> a0, a1;
        wmma::fragment<wmma::matrix_b, TC_M, TC_N, TC_K, wmma::precision::tf32, wmma::row_major> b0, b1;
        wmma::load_matrix_sync(a0, &xB[0][0],     TC_KT);
        wmma::load_matrix_sync(a1, &xB[0][TC_K],  TC_KT);
        wmma::load_matrix_sync(b0, &wB[0][0],     TC_N);
        wmma::load_matrix_sync(b1, &wB[TC_K][0],  TC_N);
#pragma unroll
        for (int t = 0; t < a0.num_elements; ++t) {
            a0.x[t] = wmma::__float_to_tf32(a0.x[t]);
            a1.x[t] = wmma::__float_to_tf32(a1.x[t]);
        }
#pragma unroll
        for (int t = 0; t < b0.num_elements; ++t) {
            b0.x[t] = wmma::__float_to_tf32(b0.x[t]);
            b1.x[t] = wmma::__float_to_tf32(b1.x[t]);
        }
        wmma::mma_sync(cFrag, a0, b0, cFrag);
        wmma::mma_sync(cFrag, a1, b1, cFrag);
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
