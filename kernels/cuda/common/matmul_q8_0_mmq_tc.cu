// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// M-Cuda.MMQ B1b — Q8_0 int8 TENSOR-CORE quantized-matmul GEMM (prefill, M>1).
//
// The B1 dp4a kernel (matmul_q8_0_mmq) gave ~17% prefill; this one runs the
// int8 dot on the Blackwell int8 tensor cores via nvcuda::wmma (16x16x16 s8 ->
// s32), for a higher compute multiplier. wmma is used instead of raw mma.sync
// PTX so the per-thread fragment layout is handled by the API (the main
// correctness hazard of hand-written TC kernels).
//
// One warp computes one 16(M) x 16(N) output tile. K is walked in 32-element
// Q8_0 blocks (= two wmma k=16 steps). A raw int32 accumulator is only ever
// summed WITHIN a block (the Q8_0 scale is per (n, block) and the activation
// scale per (m, block)); at the block boundary the 16x16 int32 partial is
// scaled by the rank-1 (d_n * s_m) outer product and added to an fp32
// accumulator. So no int32 is accumulated across blocks with differing scales.
//
// Scope: prefill / M>1 ONLY (M=1 decode stays on the GEMV path).

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define Q8_0_BLOCK_ELEMENTS 32
#define Q8_0_BLOCK_BYTES    34
#define TC_M 16
#define TC_N 16
#define TC_K 16
#define TC_WARP 32

extern "C" __global__ __launch_bounds__(TC_WARP)
void matmul_q8_0_mmq_tc(
    const float*         __restrict__ X,   // [M, K] fp32
    const unsigned char* __restrict__ W,   // [N, K/32] Q8_0 blocks
          float*         __restrict__ Y,   // [M, N] fp32
    const int                         K,
    const int                         N,
    const int                         M)
{
    // One warp per (16M x 16N) tile.
    const int lane   = threadIdx.x;            // 0..31
    const int mBase  = blockIdx.y * TC_M;
    const int nBase  = blockIdx.x * TC_N;
    const int nBlocks = K / Q8_0_BLOCK_ELEMENTS;

    __shared__ signed char xI8[TC_M][Q8_0_BLOCK_ELEMENTS];   // activations, one block
    __shared__ signed char wI8[Q8_0_BLOCK_ELEMENTS][TC_N];   // weights, one block
    __shared__ int         cI32[TC_M][TC_N];                 // per-block int32 partial
    __shared__ float       acc[TC_M][TC_N];                  // fp32 accumulator
    __shared__ float       sM[TC_M];                         // act scale per m (block)
    __shared__ float       dN[TC_N];                         // weight scale per n (block)

    for (int idx = lane; idx < TC_M * TC_N; idx += TC_WARP) {
        acc[idx / TC_N][idx % TC_N] = 0.0f;
    }
    __syncwarp();

    for (int b = 0; b < nBlocks; ++b) {
        const int kBase = b * Q8_0_BLOCK_ELEMENTS;

        // --- quantize activations for this block: one m-row per lane<16 ------
        if (lane < TC_M) {
            const int mAct = mBase + lane;
            float amax = 0.0f;
            if (mAct < M) {
                #pragma unroll
                for (int e = 0; e < Q8_0_BLOCK_ELEMENTS; ++e) {
                    const float a = fabsf(X[static_cast<size_t>(mAct) * K + kBase + e]);
                    amax = a > amax ? a : amax;
                }
            }
            const float scale = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
            const float inv   = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
            sM[lane] = scale;
            #pragma unroll
            for (int e = 0; e < Q8_0_BLOCK_ELEMENTS; ++e) {
                const float x = (mAct < M)
                    ? X[static_cast<size_t>(mAct) * K + kBase + e] : 0.0f;
                xI8[lane][e] = static_cast<signed char>(rintf(x * inv));
            }
        }

        // --- stage weights for this block: qs int8 -> wI8[k][n], d -> dN[n] --
        for (int idx = lane; idx < TC_N * Q8_0_BLOCK_ELEMENTS; idx += TC_WARP) {
            const int nn = idx / Q8_0_BLOCK_ELEMENTS;
            const int kk = idx % Q8_0_BLOCK_ELEMENTS;
            const int nAct = nBase + nn;
            signed char q = 0;
            if (nAct < N) {
                const unsigned char* block =
                    W + (static_cast<size_t>(nAct) * nBlocks + b) * Q8_0_BLOCK_BYTES;
                q = static_cast<signed char>(block[2 + kk]);
                if (kk == 0) {
                    dN[nn] = __half2float(*reinterpret_cast<const __half*>(block));
                }
            } else if (kk == 0) {
                dN[nn] = 0.0f;
            }
            wI8[kk][nn] = q;
        }
        __syncthreads();

        // --- two wmma k-steps (16+16 = 32) into a per-block int32 partial ----
        wmma::fragment<wmma::accumulator, TC_M, TC_N, TC_K, int> cFrag;
        wmma::fill_fragment(cFrag, 0);
        #pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            wmma::fragment<wmma::matrix_a, TC_M, TC_N, TC_K,
                           signed char, wmma::row_major> aFrag;
            wmma::fragment<wmma::matrix_b, TC_M, TC_N, TC_K,
                           signed char, wmma::row_major> bFrag;
            wmma::load_matrix_sync(aFrag, &xI8[0][ks * TC_K], Q8_0_BLOCK_ELEMENTS);
            wmma::load_matrix_sync(bFrag, &wI8[ks * TC_K][0], TC_N);
            wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
        }
        wmma::store_matrix_sync(&cI32[0][0], cFrag, TC_N, wmma::mem_row_major);
        __syncthreads();

        // --- scale the 16x16 int32 partial by (d_n * s_m), add to fp32 acc ---
        for (int idx = lane; idx < TC_M * TC_N; idx += TC_WARP) {
            const int m = idx / TC_N;
            const int n = idx % TC_N;
            acc[m][n] += static_cast<float>(cI32[m][n]) * dN[n] * sM[m];
        }
        __syncthreads();
    }

    // --- write out ------------------------------------------------------------
    for (int idx = lane; idx < TC_M * TC_N; idx += TC_WARP) {
        const int m = idx / TC_N;
        const int n = idx % TC_N;
        const int mAct = mBase + m;
        const int nAct = nBase + n;
        if (mAct < M && nAct < N) {
            Y[static_cast<size_t>(mAct) * N + nAct] = acc[m][n];
        }
    }
}