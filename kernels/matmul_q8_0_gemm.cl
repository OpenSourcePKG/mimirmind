// Batched matrix-matrix multiply with Q8_0 weights, on-the-fly dequant.
// M-way generalisation of matmul_q8_0_vec: processes M_TILE token rows
// per workgroup so the (already cheap) Q8_0 dequant plus the per-block
// scale multiply amortise across M_TILE mads.
//
//   Y[m, n] = sum_{k=0..K-1} X[m, k] * dequant_q8_0(W, n, k)
//
//   X:  [M, K]         F32 row-major
//   W:  [N, K/32]      Q8_0 blocks (34 B each: fp16 d + 32×int8)
//   Y:  [M, N]         F32 row-major
//
// Launch geometry:
//   local_size_x          = MATMUL_Q8_0_LOCAL   (64)
//   sub_group_size        = MATMUL_Q8_0_SG      (16)
//   outputs per workgroup = MATMUL_Q8_0_LOCAL / MATMUL_Q8_0_SG  (= 4)
//   global_size_x         = ceil(N / 4) * 64
//   global_size_y         = ceil(M / MATMUL_Q8_0_GEMM_M_TILE)
//
// See ADR "GEMM statt matvec für Prefill" (M5i) for the design context.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

#ifndef MATMUL_Q8_0_LOCAL
#define MATMUL_Q8_0_LOCAL 64
#endif

#ifndef MATMUL_Q8_0_SG
#define MATMUL_Q8_0_SG 16
#endif

#ifndef MATMUL_Q8_0_GEMM_M_TILE
#define MATMUL_Q8_0_GEMM_M_TILE 8
#endif

#define MATMUL_Q8_0_OUTPUTS_PER_GROUP (MATMUL_Q8_0_LOCAL / MATMUL_Q8_0_SG)

#define Q8_0_BLOCK_ELEMENTS 32
#define Q8_0_BLOCK_BYTES    34

// 1024 elements = 32 blocks per K-tile.
// SLM: X_TILE_ELEMENTS * MATMUL_Q8_0_GEMM_M_TILE * 4 B
//    = 1024 * 4 * 4 = 16 KiB per workgroup.
//
// Layout is xTile[k_slot][m_slot] — see matmul_q6k_gemm.cl for the
// SLM bank-conflict rationale.
#define X_TILE_ELEMENTS 1024

__attribute__((reqd_work_group_size(MATMUL_Q8_0_LOCAL, 1, 1)))
__attribute__((intel_reqd_sub_group_size(MATMUL_Q8_0_SG)))
__kernel void matmul_q8_0_gemm(
    __global const float* X,
    __global const uchar* W,
    __global       float* Y,
    const int             K,
    const int             N,
    const int             M)
{
    __local float xTile[X_TILE_ELEMENTS][MATMUL_Q8_0_GEMM_M_TILE];

    const int  wgN     = (int)get_group_id(0);
    const int  wgM     = (int)get_group_id(1);
    const int  sgInWg  = (int)get_sub_group_id();           // 0..3
    const int  sgLocal = (int)get_sub_group_local_id();     // 0..15
    const int  tid     = (int)get_local_id(0);
    const int  lsize   = (int)get_local_size(0);
    const int  n       = wgN * MATMUL_Q8_0_OUTPUTS_PER_GROUP + sgInWg;
    const int  mBase   = wgM * MATMUL_Q8_0_GEMM_M_TILE;
    const bool nActive = (n < N);
    const int  nBlocks = K / Q8_0_BLOCK_ELEMENTS;

    float sum[MATMUL_Q8_0_GEMM_M_TILE];

    #pragma unroll
    for (int mm = 0; mm < MATMUL_Q8_0_GEMM_M_TILE; ++mm) {
        sum[mm] = 0.0f;
    }

    for (int tile = 0; tile < K; tile += X_TILE_ELEMENTS) {
        const int tileK = min(X_TILE_ELEMENTS, K - tile);

        const int loadTotal = MATMUL_Q8_0_GEMM_M_TILE * X_TILE_ELEMENTS;
        for (int idx = tid; idx < loadTotal; idx += lsize) {
            const int  mSlot = idx / X_TILE_ELEMENTS;
            const int  iSlot = idx - mSlot * X_TILE_ELEMENTS;
            const int  mAct  = mBase + mSlot;
            const bool valid = (mAct < M) && (iSlot < tileK);
            xTile[iSlot][mSlot] =
                valid ? X[(size_t)mAct * (size_t)K + tile + iSlot] : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (nActive) {
            __global const uchar* row =
                W + (size_t)n * (size_t)nBlocks * Q8_0_BLOCK_BYTES;

            const int blockStart   = tile / Q8_0_BLOCK_ELEMENTS;
            const int blocksInTile = X_TILE_ELEMENTS / Q8_0_BLOCK_ELEMENTS;
            const int blockEnd     = min(blockStart + blocksInTile, nBlocks);

            for (int b = blockStart; b < blockEnd; ++b) {
                __global const uchar* block = row + b * Q8_0_BLOCK_BYTES;
                const float d =
                    vload_half(0, (__global const half*)(block));
                __global const char* qs =
                    (__global const char*)(block + 2);

                const int xLocalBase = (b - blockStart) * Q8_0_BLOCK_ELEMENTS;

                // 16 subgroup lanes cover the 32-element block. Each
                // lane dequantises 2 quants across the strided sweep
                // and MACs into every M_TILE accumulator.
                for (int l = sgLocal; l < Q8_0_BLOCK_ELEMENTS;
                     l += MATMUL_Q8_0_SG) {
                    const float w = d * (float)qs[l];

                    #pragma unroll
                    for (int mm = 0; mm < MATMUL_Q8_0_GEMM_M_TILE; ++mm) {
                        sum[mm] = mad(xTile[xLocalBase + l][mm], w, sum[mm]);
                    }
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    #pragma unroll
    for (int mm = 0; mm < MATMUL_Q8_0_GEMM_M_TILE; ++mm) {
        float s = sub_group_reduce_add(sum[mm]);
        if (nActive && sgLocal == 0) {
            const int mAct = mBase + mm;
            if (mAct < M) {
                Y[(size_t)mAct * (size_t)N + n] = s;
            }
        }
    }
}