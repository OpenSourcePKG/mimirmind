// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

// Matrix-vector multiply with Q8_0 weights in REORDERED row layout.
//
// Structurally identical to matmul_q8_0_vec.cl — same launch geometry,
// same math, same fp32 accumulator — but the row-local layout is
// scales-then-quants (see src/compute/quant/Q8_0.hpp for the contract)
// instead of interleaved 34-byte blocks. The native kernel pays a
// scattered-load tax because the 34-byte block stride breaks Xe subgroup
// coalescing (Scale @ offset 0, quants @ offset 2, next block's scale
// @ offset 34 — the 2-byte gap ejects every subgroup vector load).
//
// Reorder splits each row into two contiguous regions:
//   [ nBlocks * fp16 scales (2 B each, 2*nBlocks bytes total) ]
//   [ nBlocks * 32 int8 quants (32 B each, 32*nBlocks bytes total) ]
// Total row size is unchanged (nBlocks * 34 B). Scales are naturally
// 2-aligned; quants start at 2*nBlocks which is 32-aligned for all K
// used by our production models (E4B: 256, 2048, 5120, 8192; 26B-A4B:
// 512, 2816, 5760, ... — all multiples of 64 so nBlocks is even).
//
// Blueprint: llama.cpp PR #21527 in the SYCL backend hit ~3.1x on
// Qwen3.5-27B (Xe2/Battlemage) via this transformation, lifting Q8_0
// mmvq bandwidth utilisation from ~21 % to ~66 %. See M8.K.Q8_0-Reorder
// in the perf-regression ledger for the expected impact on Xe-LPG.
//
//   Y[n] = sum_{k=0..K-1} X[k] * dequant_q8_0(W_reordered, n, k)
//
//   X:  [K]     F32 dense vector (single token / M=1)
//   W:  [N, K]  Q8_0 reordered — each row is 2*nBlocks scale bytes
//               followed by 32*nBlocks quant bytes (nBlocks = K/32).
//   Y:  [N]     F32 dense vector
//
// Launch geometry (identical to matmul_q8_0_vec):
//   local_size_x          = MATMUL_Q8_0_LOCAL    (64)
//   sub_group_size        = MATMUL_Q8_0_SG       (16) via intel_reqd_sub_group_size
//   outputs per workgroup = MATMUL_Q8_0_LOCAL / MATMUL_Q8_0_SG  (= 4)
//   global_size_x         = ceil(N / 4) * 64

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

// 1024 elements = 32 blocks = 4 KiB SLM per workgroup — matches the
// native vec kernel's tile size so bench comparisons only differ in the
// load pattern, not in the tiling strategy.
#define X_TILE_ELEMENTS 1024

__attribute__((reqd_work_group_size(MATMUL_Q8_0_LOCAL, 1, 1)))
__attribute__((intel_reqd_sub_group_size(MATMUL_Q8_0_SG)))
__kernel void matmul_q8_0_vec_reorder(
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
            // Row base: total row size is unchanged (nBlocks * 34 B) so
            // the outer stride computation stays identical to the native
            // kernel. Only the WITHIN-row layout is different.
            __global const uchar* row =
                W + (size_t)n * (size_t)nBlocks * Q8_0_BLOCK_BYTES;

            // Reorder layout: scales region first (2*nBlocks bytes),
            // then quants region (32*nBlocks bytes). Casting to `half*`
            // and `char*` lets us index by block number directly instead
            // of chasing 34-byte block strides.
            __global const half* scales =
                (__global const half*)row;
            __global const char* quants =
                (__global const char*)(row + 2 * nBlocks);

            const int blockStart    = tile / Q8_0_BLOCK_ELEMENTS;
            const int blocksInTile  = X_TILE_ELEMENTS / Q8_0_BLOCK_ELEMENTS;
            const int blockEnd      = min(blockStart + blocksInTile, nBlocks);

            for (int b = blockStart; b < blockEnd; ++b) {
                // Clean fp16 load at 2-byte stride — every scale sits
                // exactly one half apart, so the pointer arithmetic is
                // trivially coalescable across the subgroup.
                const float d = vload_half(b, scales);
                // Quants row for this block is a 32-byte contiguous run
                // at a 32-byte-aligned offset (2*nBlocks is a multiple
                // of 32 for all production K). 16 lanes × 2 quants each
                // now maps to a single wide subgroup byte-load pattern
                // instead of the strided 34-byte block accesses.
                __global const char* qs =
                    quants + b * Q8_0_BLOCK_ELEMENTS;

                const int xLocalBase = (b - blockStart) * Q8_0_BLOCK_ELEMENTS;

                // 16 lanes split the 32-element block: each lane handles
                // 2 quants. Strided so adjacent lanes touch adjacent
                // elements (better X-tile cache behaviour) — same lane
                // partition as the native kernel so both kernels produce
                // bit-identical fp32 output for the same input.
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
