// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// DP4A matvec for Q6_K weights with pre-quantised per-32-block int8
// activation (x_quant_q8_1_blocks) — ported from llama.cpp's
// ggml-cuda vec_dot_q6_K_q8_1_impl_mmvq (ggml/src/ggml-cuda/vecdotq.cuh),
// restructured as a standalone one-warp-per-output-row kernel instead of
// their generic multi-quant-type templated mul_mat_vec_q driver.
//
// Why not mimirmind's own existing Q6_K dequant layout (as used in
// moe_down_fused_k_q6k.cu / the L0 gate-up kernel): that layout groups,
// per lane, 4 elements at offsets {0,32,64,96} — which sit in 4 DIFFERENT
// 16-element scale sub-blocks. dp4a packs 4 int8 values into one SIMD dot
// with a single implicit scale; four differently-scaled elements can't
// share one dp4a call. llama.cpp's own grouping instead packs, per lane,
// 4 CONSECUTIVE elements that share one scale byte, which is what makes
// dp4a applicable at all. Ported mechanically to preserve that property
// exactly rather than re-deriving new bit tricks against mimirmind's
// existing (differently-grouped, dp4a-incompatible) lane layout.
//
// Block layout (ggml block_q6_K, unchanged from the existing fp32 kernel):
//   ql[128]      quants, lower 4 bits
//   qh[64]       quants, upper 2 bits
//   scales[16]   int8, one per 16-element sub-block
//   d            ggml_half, super-block scale
// QK_K=256, QR6_K=2, QI6_K=QK_K/(4*QR6_K)=32 — exactly one warp (32 lanes)
// per Q6_K super-block; each lane owns a unique `iqs` in [0,32).
//
// Per lane (iqs = lane), per super-block (kbx):
//   bq8_offset    = 4*(iqs/16) + (iqs%16)/8
//   scale_offset  = 8*(iqs/16) + (iqs%16)/4
//   vh_shift      = 2*((iqs%16)/8)
//   vl            = 4 bytes of ql at byte offset 4*iqs           (get_int_b2)
//   vh            = (4 bytes of qh at byte offset
//                     4*(8*(iqs/16) + iqs%8)) >> vh_shift        (get_int_b2)
//   for i in {0,1} (QR6_K):
//     vil = (vl >> 4*i) & 0x0F0F0F0F
//     vih = ((vh >> 4*i) << 4) & 0x30303030
//     vi  = __vsub4(vil | vih, 0x20202020)      // 4 packed int8, - 32 each
//     u_i = 4 bytes of the (bq8_offset + 2*i)-th 32-wide q8_1 activation
//           block at int-index (iqs % 8)                        (get_int_b4)
//     sumf += d8[bq8_offset + 2*i] * ( dp4a(vi, u_i, 0) * scales[scale_offset + 4*i] )
//   row_partial += d * sumf   (d = this super-block's fp16 scale)
//
// Each Q6_K super-block's 256 elements are quantised into 8 consecutive
// 32-wide q8_1 blocks by x_quant_q8_1_blocks (block kbx owns blocks
// [kbx*8, kbx*8+8)); bq8_offset/i above pick 2 of those 8 per lane, and
// across all 32 lanes every one of the 8 blocks is read by exactly 4
// lanes (matches upstream exactly).
//
// Final reduction: plain warp32 __shfl_xor_sync tree over the per-lane
// `row_partial` accumulated across all super-blocks — every lane holds
// disjoint (iqs-selected) contributions to the SAME row, summing all 32
// gives the exact row total.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define MATMUL_Q6K_DP4A_LOCAL 128
#define MATMUL_Q6K_DP4A_WARP  32
#define MATMUL_Q6K_DP4A_OUTPUTS_PER_GROUP \
    (MATMUL_Q6K_DP4A_LOCAL / MATMUL_Q6K_DP4A_WARP)

#define Q6K_SUPERBLOCK_ELEMENTS 256
#define Q6K_SUPERBLOCK_BYTES    210
#define Q6K_QL_BYTES            128
#define Q6K_QH_BYTES            64
#define Q6K_QI8_BLOCKS_PER_SUPER 8   // 256 / 32

static __device__ __forceinline__ float warp32_reduce_sum(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 16, 32);
    v += __shfl_xor_sync(0xffffffffu, v, 8, 32);
    v += __shfl_xor_sync(0xffffffffu, v, 4, 32);
    v += __shfl_xor_sync(0xffffffffu, v, 2, 32);
    v += __shfl_xor_sync(0xffffffffu, v, 1, 32);
    return v;
}

// get_int_b2 equivalent: read 4 bytes starting at byte offset `byteOff`
// from a buffer that is only guaranteed 2-byte aligned (ql/qh are plain
// uint8_t arrays inside a 210 B packed struct with no 4-byte guarantee).
static __device__ __forceinline__ int get4_unaligned(const unsigned char* p, int byteOff) {
    const unsigned char* q = p + byteOff;
    return  static_cast<int>(q[0])
         | (static_cast<int>(q[1]) << 8)
         | (static_cast<int>(q[2]) << 16)
         | (static_cast<int>(q[3]) << 24);
}

extern "C" __global__ __launch_bounds__(MATMUL_Q6K_DP4A_LOCAL)
void matmul_q6k_vec_dp4a(
    const signed char*   __restrict__ Xq,       // [K]        int8 activation
    const float*         __restrict__ XqScale,  // [K/32]     per-block scale
    const unsigned char* __restrict__ W,        // [N, K]     Q6_K rows
          float*         __restrict__ Y,        // [N]
    const int                         K,
    const int                         N)
{
    const int  wg       = blockIdx.x;
    const int  tid      = threadIdx.x;
    const int  warpInWg = tid / MATMUL_Q6K_DP4A_WARP;
    const int  iqs      = tid % MATMUL_Q6K_DP4A_WARP;   // this lane's llama.cpp `iqs`
    const int  n        = wg * MATMUL_Q6K_DP4A_OUTPUTS_PER_GROUP + warpInWg;
    const bool active   = (n < N);

    const int nSuper = K / Q6K_SUPERBLOCK_ELEMENTS;

    // Precompute this lane's fixed (iqs-derived) offsets — identical for
    // every super-block, only the base pointers move.
    const int bq8_offset   = 4 * (iqs / 16) + (iqs % 16) / 8;
    const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
    const int vh_shift      = 2 * ((iqs % 16) / 8);
    const int vl_byteOff    = 4 * iqs;
    const int vh_byteOff    = 4 * (8 * (iqs / 16) + (iqs % 8));
    const int u_intIdx      = iqs % 8;

    float rowPartial = 0.0f;

    if (active) {
        const unsigned char* __restrict__ rowBase =
            W + static_cast<size_t>(n) * static_cast<size_t>(nSuper)
              * static_cast<size_t>(Q6K_SUPERBLOCK_BYTES);

        for (int kbx = 0; kbx < nSuper; ++kbx) {
            const unsigned char* __restrict__ sb = rowBase + kbx * Q6K_SUPERBLOCK_BYTES;
            const unsigned char* __restrict__ ql = sb;
            const unsigned char* __restrict__ qh = sb + Q6K_QL_BYTES;
            const signed char*   __restrict__ scales =
                reinterpret_cast<const signed char*>(sb + Q6K_QL_BYTES + Q6K_QH_BYTES);
            const float d = __half2float(
                *reinterpret_cast<const __half*>(sb + Q6K_QL_BYTES + Q6K_QH_BYTES + 16));

            const int vl     = get4_unaligned(ql, vl_byteOff);
            const int vhFull = get4_unaligned(qh, vh_byteOff);
            const int vh     = vhFull >> vh_shift;

            const signed char* xq8base = Xq      + kbx * Q6K_SUPERBLOCK_ELEMENTS;
            const float*       xs8base = XqScale + kbx * Q6K_QI8_BLOCKS_PER_SUPER;

            float sumf = 0.0f;

            #pragma unroll
            for (int i = 0; i < 2; ++i) {   // QR6_K = 2
                const int vil = (vl >> (4 * i)) & 0x0F0F0F0F;
                const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
                const int vi  = __vsub4(vil | vih, 0x20202020);

                const int   qBlk = bq8_offset + 2 * i;
                const signed char* uPtr =
                    xq8base + qBlk * 32 + u_intIdx * 4;
                const int u = get4_unaligned(
                    reinterpret_cast<const unsigned char*>(uPtr), 0);
                const float d8 = xs8base[qBlk];
                const signed char sc = scales[scale_offset + 4 * i];

                sumf += d8 * (static_cast<float>(__dp4a(vi, u, 0)) * static_cast<float>(sc));
            }

            rowPartial += d * sumf;
        }
    }

    rowPartial = warp32_reduce_sum(rowPartial);

    if (active && iqs == 0) {
        Y[n] = rowPartial;
    }
}
