// Matrix-vector multiply with Q8_0 weights, on-the-fly dequant.
//
//   Y[n] = sum_{k=0..K-1} X[k] * dequant_q8_0(W, n, k)
//
//   X:  [K]     F32 dense vector (single token / M=1)
//   W:  [N, K]  Q8_0 — each row is (K/32) blocks of 34 bytes.
//   Y:  [N]     F32 dense vector
//
// Launch geometry (mirrors matmul_q6k_vec):
//   local_size_x          = MATMUL_Q8_0_LOCAL   (64)
//   sub_group_size        = MATMUL_Q8_0_SG      (16) via intel_reqd_sub_group_size
//   outputs per workgroup = MATMUL_Q8_0_LOCAL / MATMUL_Q8_0_SG  (= 4)
//   global_size_x         = ceil(N / 4) * 64
//
// Q8_0 block layout (matches ggml block_q8_0):
//   fp16  d        — block scale (2 B)
//   int8  qs[32]   — 32 signed 8-bit quants (32 B)
// value[i] = d * qs[i]
//
// Q8_0 has no sub-scales and no bit packing — the dequant is a single
// multiply per element, much simpler than Q4_K / Q6_K. Per-thread
// accumulation in plain FP32 is sufficient: each sub-group lane sums
// ~K/16 terms (~176 for K=2816) which stays well inside FP32's
// representable range with mantissa noise dominated by the input X.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

#ifndef MATMUL_Q8_0_LOCAL
#define MATMUL_Q8_0_LOCAL 64
#endif

#ifndef MATMUL_Q8_0_SG
#define MATMUL_Q8_0_SG 16
#endif

#define MATMUL_Q8_0_OUTPUTS_PER_GROUP (MATMUL_Q8_0_LOCAL / MATMUL_Q8_0_SG)

#define Q8_0_BLOCK_ELEMENTS 32
#define Q8_0_BLOCK_BYTES    34

// 1024 elements = 32 blocks = 4 KiB SLM per workgroup.
#define X_TILE_ELEMENTS 1024

__attribute__((reqd_work_group_size(MATMUL_Q8_0_LOCAL, 1, 1)))
__attribute__((intel_reqd_sub_group_size(MATMUL_Q8_0_SG)))
__kernel void matmul_q8_0_vec(
    __global const float* X,
    __global const uchar* W,
    __global       float* Y,
    const int             K,
    const int             N)
{
    __local float xTile[X_TILE_ELEMENTS];

    const int  wg      = (int)get_group_id(0);
    const int  sgInWg  = (int)get_sub_group_id();           // 0..3
    const int  sgLocal = (int)get_sub_group_local_id();     // 0..15
    const int  tid     = (int)get_local_id(0);
    const int  lsize   = (int)get_local_size(0);
    const int  n       = wg * MATMUL_Q8_0_OUTPUTS_PER_GROUP + sgInWg;
    const bool active  = (n < N);
    const int  nBlocks = K / Q8_0_BLOCK_ELEMENTS;

    float sum = 0.0f;

    for (int tile = 0; tile < K; tile += X_TILE_ELEMENTS) {
        const int tileK = min(X_TILE_ELEMENTS, K - tile);
        for (int i = tid; i < tileK; i += lsize) {
            xTile[i] = X[tile + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (active) {
            __global const uchar* row =
                W + (size_t)n * (size_t)nBlocks * Q8_0_BLOCK_BYTES;

            const int blockStart    = tile / Q8_0_BLOCK_ELEMENTS;
            const int blocksInTile  = X_TILE_ELEMENTS / Q8_0_BLOCK_ELEMENTS;
            const int blockEnd      = min(blockStart + blocksInTile, nBlocks);

            for (int b = blockStart; b < blockEnd; ++b) {
                __global const uchar* block = row + b * Q8_0_BLOCK_BYTES;
                const float d =
                    vload_half(0, (__global const half*)(block));
                __global const char* qs =
                    (__global const char*)(block + 2);

                const int xLocalBase = (b - blockStart) * Q8_0_BLOCK_ELEMENTS;

                // 16 lanes split the 32-element block: each lane handles
                // 2 quants. Strided so adjacent lanes touch adjacent
                // elements (better X-tile cache behaviour).
                for (int l = sgLocal; l < Q8_0_BLOCK_ELEMENTS; l += MATMUL_Q8_0_SG) {
                    sum = mad(xTile[xLocalBase + l],
                              d * (float)qs[l],
                              sum);
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    sum = sub_group_reduce_add(sum);

    if (active && sgLocal == 0) {
        Y[n] = sum;
    }
}