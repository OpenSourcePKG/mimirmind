// Matrix-vector multiply with Q6_K weights, on-the-fly dequant.
//
//   Y[n] = sum_{k=0..K-1} X[k] * dequant_q6k(W, n, k)
//
//   X:  [K]     F32 dense vector (single token / M=1)
//   W:  [N, K]  Q6_K — each row is (K/256) super-blocks of 210 bytes.
//   Y:  [N]     F32 dense vector
//
// Q6_K super-block layout (matches ggml block_q6_K):
//   uint8  ql[128]    : lower 4 bits of 256 6-bit quants
//   uint8  qh[64]     : upper 2 bits of 256 6-bit quants
//   int8   scales[16] : 16 signed scales, one per 16-element sub-block
//   fp16   d          : super-block scale
//
// Per element: q in [-32, 31] from packed nibbles,
//   value = d * scales[is] * q
//
// Launch: global = N work-items grouped into ceil(N / MATMUL_Q6K_LOCAL)
//   workgroups. One work-item produces one Y[n]. Same data-parallel
//   pattern as matmul_q4k_vec — no shared X cache yet.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifndef MATMUL_Q6K_LOCAL
#define MATMUL_Q6K_LOCAL 64
#endif

#define Q6K_BLOCK_ELEMENTS 256
#define Q6K_BLOCK_BYTES    210

__attribute__((reqd_work_group_size(MATMUL_Q6K_LOCAL, 1, 1)))
__kernel void matmul_q6k_vec(
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

    const int nSuper = K / Q6K_BLOCK_ELEMENTS;
    __global const uchar* row =
        W + (size_t)n * (size_t)nSuper * Q6K_BLOCK_BYTES;

    float sum = 0.0f;
    for (int sb = 0; sb < nSuper; ++sb) {
        __global const uchar* block = row + sb * Q6K_BLOCK_BYTES;

        __global const uchar* ql = block;              // 128 bytes
        __global const uchar* qh = block + 128;        // 64 bytes
        __global const char*  sc = (__global const char*)(block + 192);  // 16 signed bytes
        const float d = vload_half(0, (__global const half*)(block + 208));

        const int sbBase = sb * Q6K_BLOCK_ELEMENTS;

        // Two 128-element halves per super-block. ('half' is a reserved
        // type name in OpenCL — use hIdx as the loop variable.)
        for (int hIdx = 0; hIdx < 2; ++hIdx) {
            const int hBase = sbBase + hIdx * 128;
            __global const uchar* qlp = ql + hIdx * 64;
            __global const uchar* qhp = qh + hIdx * 32;
            __global const char*  scp = sc + hIdx * 8;

            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;

                const char q1 = (char)((qlp[l +  0] & 0x0F) |
                                       (((qhp[l] >> 0) & 0x03) << 4)) - 32;
                const char q2 = (char)((qlp[l + 32] & 0x0F) |
                                       (((qhp[l] >> 2) & 0x03) << 4)) - 32;
                const char q3 = (char)((qlp[l +  0] >> 4) |
                                       (((qhp[l] >> 4) & 0x03) << 4)) - 32;
                const char q4 = (char)((qlp[l + 32] >> 4) |
                                       (((qhp[l] >> 6) & 0x03) << 4)) - 32;

                const float s0 = d * (float)scp[is + 0];
                const float s2 = d * (float)scp[is + 2];
                const float s4 = d * (float)scp[is + 4];
                const float s6 = d * (float)scp[is + 6];

                sum = mad(X[hBase + l +  0], s0 * (float)q1, sum);
                sum = mad(X[hBase + l + 32], s2 * (float)q2, sum);
                sum = mad(X[hBase + l + 64], s4 * (float)q3, sum);
                sum = mad(X[hBase + l + 96], s6 * (float)q4, sum);
            }
        }
    }

    Y[n] = sum;
}