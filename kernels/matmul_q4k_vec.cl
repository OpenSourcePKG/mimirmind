// Matrix-vector multiply with Q4_K weights, on-the-fly dequant.
//
//   Y[n] = sum_{k=0..K-1} X[k] * dequant_q4k(W, n, k)
//
//   X:  [K]     F32 dense vector (single token / M=1)
//   W:  [N, K]  Q4_K — each row is (K/256) super-blocks of 144 bytes
//                in the layout matching ggml's block_q4_K.
//   Y:  [N]     F32 dense vector
//
// Launch: global = N work-items, local = MATMUL_Q4K_LOCAL.
//   One work-item produces one Y[n]. Each work-item walks its row of
//   W super-block by super-block, decodes scale + mins, and dots
//   against the matching slice of X. No intra-workgroup cooperation
//   yet — pure data-parallel per output. X is re-read by every thread;
//   first optimisation pass should be a __local cache of an X tile.
//
// Reference: ggml-quants.c dequantize_row_q4_K. Sub-block iteration
// matches our compute/Dequant.cpp dequantQ4K (paired sub-blocks
// j, j+1 share one 32-byte qs chunk: lower nibble -> sub-block j,
// upper nibble -> sub-block j+1).

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifndef MATMUL_Q4K_LOCAL
#define MATMUL_Q4K_LOCAL 64
#endif

#define Q4K_BLOCK_ELEMENTS 256
#define Q4K_BLOCK_BYTES    144

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
    const int n = (int)get_global_id(0);
    if (n >= N) {
        return;
    }

    const int nSuper = K / Q4K_BLOCK_ELEMENTS;

    __global const uchar* row =
        W + (size_t)n * (size_t)nSuper * Q4K_BLOCK_BYTES;

    float sum = 0.0f;
    for (int sb = 0; sb < nSuper; ++sb) {
        __global const uchar* block = row + sb * Q4K_BLOCK_BYTES;

        const float d    = vload_half(0, (__global const half*)block);
        const float dmin = vload_half(1, (__global const half*)block);

        __global const uchar* scales = block + 4;    // 12 bytes
        __global const uchar* qs     = block + 16;   // 128 bytes (256 nibbles)

        const int sbBase = sb * Q4K_BLOCK_ELEMENTS;

        // Eight sub-blocks of 32 elements, processed two at a time
        // (paired: lower nibble -> sub-block j, upper nibble -> j+1).
        for (int j = 0; j < 8; j += 2) {
            const uchar2 sm0 = q4k_scale_min(j,     scales);
            const uchar2 sm1 = q4k_scale_min(j + 1, scales);
            const float  d1  = d    * (float)sm0.x;
            const float  m1  = dmin * (float)sm0.y;
            const float  d2  = d    * (float)sm1.x;
            const float  m2  = dmin * (float)sm1.y;

            const int qsOffset = (j / 2) * 32;
            const int xLoBase  = sbBase + j * 32;
            const int xHiBase  = sbBase + (j + 1) * 32;

            for (int l = 0; l < 32; ++l) {
                const uchar q   = qs[qsOffset + l];
                const float qLo = (float)(q & 0x0F);
                const float qHi = (float)(q >> 4);
                sum = mad(X[xLoBase + l], d1 * qLo - m1, sum);
                sum = mad(X[xHiBase + l], d2 * qHi - m2, sum);
            }
        }
    }

    Y[n] = sum;
}