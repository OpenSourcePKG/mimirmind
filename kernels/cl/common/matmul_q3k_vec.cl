// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// SLM-free Q3_K matvec with on-the-fly dequant. Sibling of
// matmul_q6k_vec.cl: same WG geometry (LOCAL=64, SG=8, 8 outputs/WG, one
// sub-group per output row), plain-fma split accumulators, no local X-tile
// (for a matvec X is shared by every workgroup and stays cache-hot).
//
//   Y[n] = sum_{k=0..K-1} X[k] * dequant_q3_k(W, n, k)
//
//   X: [K]    F32 dense vector (single token / M=1)
//   W: [N, K] Q3_K — each row is (K/256) super-blocks of 110 bytes
//   Y: [N]    F32 dense vector
//
// Q3_K super-block layout (bit-identical to llama.cpp dequantize_row_q3_K
// and src/compute/quant/Q3K.cpp):
//   uint8 hmask[32]  (bytes 0..31)   high 1 bit of 256 3-bit quants
//   uint8 qs[64]     (bytes 32..95)  low 2 bits of 256 3-bit quants
//   uint8 scales[12] (bytes 96..107) 16 x 6-bit unsigned scales (packed)
//   fp16  d          (bytes 108..109) super-block scale
//
// Element reconstruction (mirrors the CPU reference exactly). The 256
// elements are two 128-element halves; each half has 4 sub-blocks of 32.
// For sub-block sb = 4*half + j (j in 0..3) and element e in 0..31:
//   shift = 2*j ; mBit = sb
//   low2  = (qs[half*32 + e] >> shift) & 0x3
//   hi1   = (hmask[e]        >> mBit ) & 0x1
//   qv    = low2 - (hi1 ? 0 : 4)                     in [-4, 3]
//   is    = 2*sb + (e < 16 ? 0 : 1)                  two scales per sub-block
//   value = d * (scales[is] - 32) * qv
//   k     = half*128 + j*32 + e                      (dequant output order)
//
// The host launches ceil(N/kOutputsPerGroup)=ceil(N/4) workgroups (the
// shared vec-launch geometry); with 8 outputs/WG here that is a 2x
// workgroup over-launch whose extra groups all take the `n >= N` early
// return — free on Xe-LPG (empty-workgroup execution costs nothing). No
// host change needed; results are byte-correct (parity-gated at startup).

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

#ifndef MATMUL_Q3K_LOCAL
#define MATMUL_Q3K_LOCAL 64
#endif

#ifndef MATMUL_Q3K_SG
#define MATMUL_Q3K_SG 8
#endif

#define MATMUL_Q3K_OUTPUTS_PER_GROUP (MATMUL_Q3K_LOCAL / MATMUL_Q3K_SG)

#define Q3K_BLOCK_ELEMENTS 256
#define Q3K_BLOCK_BYTES    110

// Unpack the 12-byte packed scales into 16 6-bit unsigned scales in
// [0..63]. Same kmask1/kmask2 trick as llama.cpp dequantize_row_q3_K and
// src/compute/quant/Q3K.cpp::unpackScales — bit-identical. Runs once per
// super-block per lane (redundant across the sub-group; ~50 cheap ALU ops
// beat a barrier on Xe-LPG).
static inline void unpackScalesQ3K(__global const uchar* p, uchar out[16]) {
    const uint kmask1 = 0x03030303u;
    const uint kmask2 = 0x0f0f0f0fu;

    const uint aux0 = (uint)p[0] | ((uint)p[1] << 8) |
                      ((uint)p[2] << 16) | ((uint)p[3] << 24);
    const uint aux1 = (uint)p[4] | ((uint)p[5] << 8) |
                      ((uint)p[6] << 16) | ((uint)p[7] << 24);
    const uint tmp  = (uint)p[8] | ((uint)p[9] << 8) |
                      ((uint)p[10] << 16) | ((uint)p[11] << 24);

    const uint a0 = ( aux0        & kmask2) | (((tmp >> 0) & kmask1) << 4);
    const uint a1 = ( aux1        & kmask2) | (((tmp >> 2) & kmask1) << 4);
    const uint a2 = ((aux0 >> 4)  & kmask2) | (((tmp >> 4) & kmask1) << 4);
    const uint a3 = ((aux1 >> 4)  & kmask2) | (((tmp >> 6) & kmask1) << 4);

    out[ 0] = (uchar)( a0        & 0xFFu);
    out[ 1] = (uchar)((a0 >>  8) & 0xFFu);
    out[ 2] = (uchar)((a0 >> 16) & 0xFFu);
    out[ 3] = (uchar)((a0 >> 24) & 0xFFu);
    out[ 4] = (uchar)( a1        & 0xFFu);
    out[ 5] = (uchar)((a1 >>  8) & 0xFFu);
    out[ 6] = (uchar)((a1 >> 16) & 0xFFu);
    out[ 7] = (uchar)((a1 >> 24) & 0xFFu);
    out[ 8] = (uchar)( a2        & 0xFFu);
    out[ 9] = (uchar)((a2 >>  8) & 0xFFu);
    out[10] = (uchar)((a2 >> 16) & 0xFFu);
    out[11] = (uchar)((a2 >> 24) & 0xFFu);
    out[12] = (uchar)( a3        & 0xFFu);
    out[13] = (uchar)((a3 >>  8) & 0xFFu);
    out[14] = (uchar)((a3 >> 16) & 0xFFu);
    out[15] = (uchar)((a3 >> 24) & 0xFFu);
}

__attribute__((reqd_work_group_size(MATMUL_Q3K_LOCAL, 1, 1)))
__attribute__((intel_reqd_sub_group_size(MATMUL_Q3K_SG)))
__kernel void matmul_q3k_vec(
    __global const float* X,
    __global const uchar* W,
    __global       float* Y,
    const int             K,
    const int             N)
{
    const int wg      = (int)get_group_id(0);
    const int sgInWg  = (int)get_sub_group_id();
    const int sgLocal = (int)get_sub_group_local_id();
    const int n       = wg * MATMUL_Q3K_OUTPUTS_PER_GROUP + sgInWg;
    if (n >= N) return;

    const int nSuper = K / Q3K_BLOCK_ELEMENTS;
    __global const uchar* row =
        W + (size_t)n * (size_t)nSuper * Q3K_BLOCK_BYTES;

    float acc0 = 0.0f, acc1 = 0.0f;

    for (int sb = 0; sb < nSuper; ++sb) {
        __global const uchar* block = row + sb * Q3K_BLOCK_BYTES;

        __global const uchar* hmask = block;          // 32 bytes
        __global const uchar* qs    = block + 32;     // 64 bytes
        const float d = vload_half(0, (__global const half*)(block + 108));

        uchar scales[16];
        unpackScalesQ3K(block + 96, scales);

        const int xBlockBase = sb * Q3K_BLOCK_ELEMENTS;

        // Two 128-element halves per super-block.
        for (int hIdx = 0; hIdx < 2; ++hIdx) {
            __global const uchar* qsh = qs + hIdx * 32;
            const int xHalfBase = xBlockBase + hIdx * 128;

            // Four 32-element sub-blocks per half.
            for (int j = 0; j < 4; ++j) {
                const int sbi   = 4 * hIdx + j;   // 0..7
                const int shift = 2 * j;
                const int mBit  = sbi;

                // SG lanes stride the 32-element inner span (4 mads each).
                for (int e = sgLocal; e < 32; e += MATMUL_Q3K_SG) {
                    const uint low2 = ((uint)qsh[e]   >> shift) & 0x3u;
                    const uint hi1  = ((uint)hmask[e] >> mBit)  & 0x1u;
                    const int  qv   = (int)low2 - (hi1 ? 0 : 4);

                    const int   is = 2 * sbi + (e < 16 ? 0 : 1);
                    const float w  = d * (float)((int)scales[is] - 32) *
                                         (float)qv;

                    // Split accumulator by sub-block parity to break the
                    // dependency chain (the Q6_K-vec lesson: geometry over
                    // load-width for register-heavy K-quants).
                    if (j & 1) {
                        acc1 = mad(X[xHalfBase + j * 32 + e], w, acc1);
                    } else {
                        acc0 = mad(X[xHalfBase + j * 32 + e], w, acc0);
                    }
                }
            }
        }
    }

    const float sum = sub_group_reduce_add(acc0 + acc1);

    if (sgLocal == 0) {
        Y[n] = sum;
    }
}
