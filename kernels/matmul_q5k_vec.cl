// Matrix-vector multiply with Q5_K weights, on-the-fly dequant.
//
//   Y[n] = sum_{k=0..K-1} X[k] * dequant_q5k(W, n, k)
//
//   X:  [K]     F32 dense vector (single token / M=1)
//   W:  [N, K]  Q5_K — each row is (K/256) super-blocks of 176 bytes
//   Y:  [N]     F32 dense vector
//
// Launch geometry (mirrors matmul_q4k_vec):
//   local_size_x          = MATMUL_Q5K_LOCAL   (64)
//   sub_group_size        = MATMUL_Q5K_SG      (16) via intel_reqd_sub_group_size
//   outputs per workgroup = MATMUL_Q5K_LOCAL / MATMUL_Q5K_SG  (= 4)
//   global_size_x         = ceil(N / 4) * 64
//
// Q5_K super-block layout (176 B):
//   fp16  d          [0..1]
//   fp16  dmin       [2..3]
//   uchar scales[12] [4..15]     8× 6-bit scales + 8× 6-bit mins
//   uchar qh[32]     [16..47]    high bit of each 5-bit quant, one bit
//                                per quant, with a per-sub-super-block
//                                2-bit-per-iteration mask shift
//   uchar qs[128]    [48..175]   lower 4 bits of the 256 quants (256
//                                nibbles, lo-nibble first per byte)
//
// Per element: q = (qs_nibble) | (qh_bit << 4)   in [0..31]
//              value = d * scale * q - dmin * min
//
// Reference: ggml-quants.c dequantize_row_q5_K. Sub-block iteration
// matches compute/quant/Q5K.cpp dequant.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

#ifndef MATMUL_Q5K_LOCAL
#define MATMUL_Q5K_LOCAL 64
#endif

#ifndef MATMUL_Q5K_SG
#define MATMUL_Q5K_SG 16
#endif

#define MATMUL_Q5K_OUTPUTS_PER_GROUP (MATMUL_Q5K_LOCAL / MATMUL_Q5K_SG)

#define Q5K_BLOCK_ELEMENTS 256
#define Q5K_BLOCK_BYTES    176

// Must be a multiple of Q5K_BLOCK_ELEMENTS (256). 1024 = 4 super-blocks
// = 4 KiB SLM per workgroup, well within Intel iGPU SLM (>=64 KiB).
#define X_TILE_ELEMENTS 1024

inline uchar2 q5k_scale_min(int j, __global const uchar* sc) {
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

__attribute__((reqd_work_group_size(MATMUL_Q5K_LOCAL, 1, 1)))
__attribute__((intel_reqd_sub_group_size(MATMUL_Q5K_SG)))
__kernel void matmul_q5k_vec(
    __global const float* X,
    __global const uchar* W,
    __global       float* Y,
    const int             K,
    const int             N)
{
    __local float xTile[X_TILE_ELEMENTS];

    const int  wg       = (int)get_group_id(0);
    const int  sgInWg   = (int)get_sub_group_id();          // 0..3
    const int  sgLocal  = (int)get_sub_group_local_id();    // 0..15
    const int  tid      = (int)get_local_id(0);
    const int  lsize    = (int)get_local_size(0);
    const int  n        = wg * MATMUL_Q5K_OUTPUTS_PER_GROUP + sgInWg;
    const bool active   = (n < N);
    const int  nSuper   = K / Q5K_BLOCK_ELEMENTS;

    float sum = 0.0f;

    for (int tile = 0; tile < K; tile += X_TILE_ELEMENTS) {

        // Cooperative X-tile load — uniform control flow.
        const int tileK = min(X_TILE_ELEMENTS, K - tile);
        for (int i = tid; i < tileK; i += lsize) {
            xTile[i] = X[tile + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (active) {
            __global const uchar* row =
                W + (size_t)n * (size_t)nSuper * Q5K_BLOCK_BYTES;

            const int sbStart    = tile / Q5K_BLOCK_ELEMENTS;
            const int sbInTile   = X_TILE_ELEMENTS / Q5K_BLOCK_ELEMENTS;
            const int sbEnd      = min(sbStart + sbInTile, nSuper);

            for (int sb = sbStart; sb < sbEnd; ++sb) {
                __global const uchar* block = row + sb * Q5K_BLOCK_BYTES;

                const float d    = vload_half(0, (__global const half*)block);
                const float dmin = vload_half(1, (__global const half*)block);

                __global const uchar* scales = block + 4;    // 12 bytes
                __global const uchar* qh     = block + 16;   // 32 bytes
                __global const uchar* qs     = block + 48;   // 128 bytes

                const int xLocalBase = (sb - sbStart) * Q5K_BLOCK_ELEMENTS;

                // qh masks shift by 2 bits per (j, j+1) pair — starts
                // at u1=0x01/u2=0x02 and covers all 256 quants via
                // four iterations of the outer loop below.
                uchar u1 = 0x01;
                uchar u2 = 0x02;

                for (int j = 0; j < 8; j += 2) {
                    const uchar2 sm0 = q5k_scale_min(j,     scales);
                    const uchar2 sm1 = q5k_scale_min(j + 1, scales);
                    const float  d1  = d    * (float)sm0.x;
                    const float  m1  = dmin * (float)sm0.y;
                    const float  d2  = d    * (float)sm1.x;
                    const float  m2  = dmin * (float)sm1.y;

                    const int qsOffset = (j / 2) * 32;
                    const int xLoBase  = xLocalBase + j * 32;
                    const int xHiBase  = xLocalBase + (j + 1) * 32;

                    // 16 threads share the 32-element half — each does 2.
                    for (int l = sgLocal; l < 32; l += MATMUL_Q5K_SG) {
                        const uchar q    = qs[qsOffset + l];
                        const uchar qhL  = qh[l];
                        const float qLo  = (float)((q & 0x0F) + ((qhL & u1) ? 16 : 0));
                        const float qHi  = (float)((q >> 4)   + ((qhL & u2) ? 16 : 0));
                        sum = mad(xTile[xLoBase + l], d1 * qLo - m1, sum);
                        sum = mad(xTile[xHiBase + l], d2 * qHi - m2, sum);
                    }

                    u1 <<= 2;
                    u2 <<= 2;
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Collapse the 16 partial sums in this subgroup into one. Every
    // member must call this even when the subgroup is inactive.
    sum = sub_group_reduce_add(sum);

    if (active && sgLocal == 0) {
        Y[n] = sum;
    }
}