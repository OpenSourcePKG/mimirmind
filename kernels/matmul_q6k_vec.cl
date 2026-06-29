// Matrix-vector multiply with Q6_K weights, on-the-fly dequant.
//
//   Y[n] = sum_{k=0..K-1} X[k] * dequant_q6k(W, n, k)
//
//   X:  [K]     F32 dense vector (single token / M=1)
//   W:  [N, K]  Q6_K — each row is (K/256) super-blocks of 210 bytes.
//   Y:  [N]     F32 dense vector
//
// Launch geometry (M5h subgroup-reduce):
//   local_size_x          = MATMUL_Q6K_LOCAL   (64)
//   sub_group_size        = MATMUL_Q6K_SG      (16) via intel_reqd_sub_group_size
//   outputs per workgroup = MATMUL_Q6K_LOCAL / MATMUL_Q6K_SG  (= 4)
//   global_size_x         = ceil(N / 4) * 64
//
// Q6_K super-block layout (matches ggml block_q6_K):
//   uint8  ql[128]    : lower 4 bits of 256 6-bit quants
//   uint8  qh[64]     : upper 2 bits of 256 6-bit quants
//   int8   scales[16] : 16 signed scales, one per 16-element sub-block
//   fp16   d          : super-block scale
//
// Per element: q in [-32, 31], value = d * scales[is] * q

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

#ifndef MATMUL_Q6K_LOCAL
#define MATMUL_Q6K_LOCAL 64
#endif

#ifndef MATMUL_Q6K_SG
#define MATMUL_Q6K_SG 16
#endif

#define MATMUL_Q6K_OUTPUTS_PER_GROUP (MATMUL_Q6K_LOCAL / MATMUL_Q6K_SG)

#define Q6K_BLOCK_ELEMENTS 256
#define Q6K_BLOCK_BYTES    210

// 1024 elements = 4 super-blocks = 4 KiB SLM per workgroup.
#define X_TILE_ELEMENTS 1024

__attribute__((reqd_work_group_size(MATMUL_Q6K_LOCAL, 1, 1)))
__attribute__((intel_reqd_sub_group_size(MATMUL_Q6K_SG)))
__kernel void matmul_q6k_vec(
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
    const int  n       = wg * MATMUL_Q6K_OUTPUTS_PER_GROUP + sgInWg;
    const bool active  = (n < N);
    const int  nSuper  = K / Q6K_BLOCK_ELEMENTS;

    // FP64 per-thread accumulator (probe): eliminates FP32 reorder noise
    // across the ~K/16 terms each lane sums. Sub-group reduce stays FP32.
    double sum = 0.0;

    for (int tile = 0; tile < K; tile += X_TILE_ELEMENTS) {
        const int tileK = min(X_TILE_ELEMENTS, K - tile);
        for (int i = tid; i < tileK; i += lsize) {
            xTile[i] = X[tile + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (active) {
            __global const uchar* row =
                W + (size_t)n * (size_t)nSuper * Q6K_BLOCK_BYTES;

            const int sbStart  = tile / Q6K_BLOCK_ELEMENTS;
            const int sbInTile = X_TILE_ELEMENTS / Q6K_BLOCK_ELEMENTS;
            const int sbEnd    = min(sbStart + sbInTile, nSuper);

            for (int sb = sbStart; sb < sbEnd; ++sb) {
                __global const uchar* block = row + sb * Q6K_BLOCK_BYTES;

                __global const uchar* ql = block;              // 128 bytes
                __global const uchar* qh = block + 128;        // 64 bytes
                __global const char*  sc =
                    (__global const char*)(block + 192);       // 16 signed
                const float d =
                    vload_half(0, (__global const half*)(block + 208));

                const int xLocalBase = (sb - sbStart) * Q6K_BLOCK_ELEMENTS;

                // Two 128-element halves per super-block.
                for (int hIdx = 0; hIdx < 2; ++hIdx) {
                    const int xHalfBase = xLocalBase + hIdx * 128;
                    __global const uchar* qlp = ql + hIdx * 64;
                    __global const uchar* qhp = qh + hIdx * 32;
                    __global const char*  scp = sc + hIdx * 8;

                    // 16 threads share the 32-element inner span (4 mads each).
                    for (int l = sgLocal; l < 32; l += MATMUL_Q6K_SG) {
                        const int is = l / 16;

                        const char q1 = (char)((qlp[l +  0] & 0x0F) |
                                               (((qhp[l] >> 0) & 0x03) << 4)) - 32;
                        const char q2 = (char)((qlp[l + 32] & 0x0F) |
                                               (((qhp[l] >> 2) & 0x03) << 4)) - 32;
                        const char q3 = (char)((qlp[l +  0] >> 4) |
                                               (((qhp[l] >> 4) & 0x03) << 4)) - 32;
                        const char q4 = (char)((qlp[l + 32] >> 4) |
                                               (((qhp[l] >> 6) & 0x03) << 4)) - 32;

                        const double dd = (double)d;
                        const double s0 = dd * (double)scp[is + 0];
                        const double s2 = dd * (double)scp[is + 2];
                        const double s4 = dd * (double)scp[is + 4];
                        const double s6 = dd * (double)scp[is + 6];

                        sum = fma((double)xTile[xHalfBase + l +  0], s0 * (double)q1, sum);
                        sum = fma((double)xTile[xHalfBase + l + 32], s2 * (double)q2, sum);
                        sum = fma((double)xTile[xHalfBase + l + 64], s4 * (double)q3, sum);
                        sum = fma((double)xTile[xHalfBase + l + 96], s6 * (double)q4, sum);
                    }
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float fsum = (float)sum;
    fsum = sub_group_reduce_add(fsum);

    if (active && sgLocal == 0) {
        Y[n] = fsum;
    }
}