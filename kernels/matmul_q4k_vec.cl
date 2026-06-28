// Matrix-vector multiply with Q4_K weights, on-the-fly dequant.
//
//   Y[n] = sum_{k=0..K-1} X[k] * dequant_q4k(W, n, k)
//
//   X:  [K]     F32 dense vector (single token / M=1)
//   W:  [N, K]  Q4_K — each row is (K/256) super-blocks of 144 bytes
//   Y:  [N]     F32 dense vector
//
// Launch: global = N work-items in groups of MATMUL_Q4K_LOCAL.
//
// Optimisation vs the first naive version: X is cooperatively cached
// into __local memory in tiles of X_TILE_ELEMENTS (= 4 super-blocks).
// Without this each of the 64 threads in a workgroup re-reads the full
// X vector independently — for d_model=3584 that's ~900 KB of redundant
// global reads per workgroup; tiling drops it to ~14 KB.
//
// Reference: ggml-quants.c dequantize_row_q4_K. Sub-block iteration
// matches compute/Dequant.cpp dequantQ4K.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifndef MATMUL_Q4K_LOCAL
#define MATMUL_Q4K_LOCAL 64
#endif

#define Q4K_BLOCK_ELEMENTS 256
#define Q4K_BLOCK_BYTES    144

// Must be a multiple of Q4K_BLOCK_ELEMENTS (256). 1024 = 4 super-blocks
// = 4 KiB SLM per workgroup, well within Intel iGPU SLM (>=64 KiB).
#define X_TILE_ELEMENTS 1024

inline uchar2 q4k_scale_min(int j, __global const uchar* sc) {
    uchar s, m;
    if (j < 4) {
        s = sc[j]     & 0x3F;
        m = sc[j + 4] & 0x3F;
    } else {
        s = (sc[j + 4] & 0x0F) | ((sc[j - 4] >> 6) << 4);
        m = (sc[j + 4] >> 4)   | ((sc[j]     >> 6) << 4);
    }
    return (uchar2)(s, m);
}

__attribute__((reqd_work_group_size(MATMUL_Q4K_LOCAL, 1, 1)))
__kernel void matmul_q4k_vec(
    __global const float* X,
    __global const uchar* W,
    __global       float* Y,
    const int             K,
    const int             N)
{
    __local float xTile[X_TILE_ELEMENTS];

    const int  n      = (int)get_global_id(0);
    const int  tid    = (int)get_local_id(0);
    const int  lsize  = (int)get_local_size(0);
    const bool active = (n < N);
    const int  nSuper = K / Q4K_BLOCK_ELEMENTS;

    float sum = 0.0f;

    // Iterate K in X_TILE_ELEMENTS-sized tiles. Each tile is loaded
    // cooperatively by the whole workgroup, then each (active) thread
    // walks its row of W over the super-blocks that map into this tile.
    for (int tile = 0; tile < K; tile += X_TILE_ELEMENTS) {

        // All threads participate in the cooperative load (barriers
        // require uniform control flow).
        const int tileK = min(X_TILE_ELEMENTS, K - tile);
        for (int i = tid; i < tileK; i += lsize) {
            xTile[i] = X[tile + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (active) {
            __global const uchar* row =
                W + (size_t)n * (size_t)nSuper * Q4K_BLOCK_BYTES;

            const int sbStart    = tile / Q4K_BLOCK_ELEMENTS;
            const int sbInTile   = X_TILE_ELEMENTS / Q4K_BLOCK_ELEMENTS;
            const int sbEnd      = min(sbStart + sbInTile, nSuper);

            for (int sb = sbStart; sb < sbEnd; ++sb) {
                __global const uchar* block = row + sb * Q4K_BLOCK_BYTES;

                const float d    = vload_half(0, (__global const half*)block);
                const float dmin = vload_half(1, (__global const half*)block);

                __global const uchar* scales = block + 4;    // 12 bytes
                __global const uchar* qs     = block + 16;   // 128 bytes

                const int xLocalBase = (sb - sbStart) * Q4K_BLOCK_ELEMENTS;

                for (int j = 0; j < 8; j += 2) {
                    const uchar2 sm0 = q4k_scale_min(j,     scales);
                    const uchar2 sm1 = q4k_scale_min(j + 1, scales);
                    const float  d1  = d    * (float)sm0.x;
                    const float  m1  = dmin * (float)sm0.y;
                    const float  d2  = d    * (float)sm1.x;
                    const float  m2  = dmin * (float)sm1.y;

                    const int qsOffset = (j / 2) * 32;
                    const int xLoBase  = xLocalBase + j * 32;
                    const int xHiBase  = xLocalBase + (j + 1) * 32;

                    for (int l = 0; l < 32; ++l) {
                        const uchar q   = qs[qsOffset + l];
                        const float qLo = (float)(q & 0x0F);
                        const float qHi = (float)(q >> 4);
                        sum = mad(xTile[xLoBase + l], d1 * qLo - m1, sum);
                        sum = mad(xTile[xHiBase + l], d2 * qHi - m2, sum);
                    }
                }
            }
        }

        // All threads wait before the next tile is overwritten.
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (active) {
        Y[n] = sum;
    }
}