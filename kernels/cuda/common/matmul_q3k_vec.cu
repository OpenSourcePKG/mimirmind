// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
// Ported from kernels_hip/matmul_q3k_vec.hip — Track 4 mechanical port, no functional change intended.

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// Matrix-vector multiply with Q3_K weights, on-the-fly dequant.
//
//   Y[n] = sum_{k=0..K-1} X[k] * dequant_q3_k(W, n, k)
//
//   X:  [K]     F32 dense vector (single token / M=1)
//   W:  [N, K]  Q3_K — each row is (K/256) super-blocks of 110 bytes.
//   Y:  [N]     F32 dense vector
//
// Sibling of matmul_q6k_vec.hip / matmul_q5_0_vec.hip / matmul_q8_0_vec.hip:
// RDNA3 warpSize=32, one warp per output row, LOCAL=128 (4 warps per WG),
// OUTPUTS_PER_GROUP=4. Only the per-lane dequant differs — Q3_K packs
// 256 signed 3-bit quants per super-block, 16 6-bit unsigned scales
// packed into 12 bytes, and a single fp16 super-scale.
//
// Launch:
//   dim3 grid ( ceil(N / OUTPUTS_PER_GROUP), 1, 1 )
//   dim3 block( MATMUL_Q3K_LOCAL, 1, 1 )
//
// Q3_K super-block layout (bit-identical to llama.cpp
// dequantize_row_q3_K and src/compute/quant/Q3K.cpp):
//
//   uint8   hmask[32]  (bytes 0..31)   — high 1 bit of 256 3-bit quants
//                                        (packed 8 quants / byte)
//   uint8   qs[64]     (bytes 32..95)  — low 2 bits of 256 3-bit quants
//                                        (packed 4 quants / byte)
//   uint8   scales[12] (bytes 96..107) — 16 x 6-bit unsigned scales packed
//                                        via kmask1=0x03030303 / kmask2=0x0f0f0f0f
//   fp16    d          (bytes 108..109) — super-block scale
//
// Element indexing (mirroring the CPU reference exactly):
//   256 elements split into two 128-element halves. Each half has 4
//   sub-blocks of 32 elements. Within a sub-block sb ∈ [0..7]:
//     half   = sb / 4         (0 or 1)
//     sub_j  = sb % 4         (which sub-block within the half)
//     shift  = 2 * sub_j      (bit position into qs byte, 0/2/4/6)
//     mBit   = sb             (bit position into hmask byte, 0..7)
//   For lane l ∈ [0..31] (one element per sub-block per lane):
//     qs_idx = half * 32 + l  (qs is 64 B; halves occupy [0..31] and [32..63])
//     hm_idx = l              (hmask is 32 B; same range for both halves,
//                              only the bit position mBit differs)
//     is     = 2 * sb + (l >= 16 ? 1 : 0)   (2 scales per sub-block:
//                                            first for l ∈ [0..15],
//                                            second for l ∈ [16..31])
//     low_2  = (qs[qs_idx]  >> shift) & 0x3
//     hi_1   = (hmask[hm_idx] >> mBit)  & 0x1
//     qv     = low_2 - (hi_1 ? 0 : 4)              ∈ [-4, 3]
//     value  = d * (scales[is] - 32) * qv
//
// Lane assignment: each of the 32 lanes handles laneId == l for all 8
// sub-blocks (2 halves × 4). 32 lanes × 8 = 256 = full super-block.
// No cross-lane work per block.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifndef MATMUL_Q3K_LOCAL
#define MATMUL_Q3K_LOCAL 128
#endif

#define MATMUL_Q3K_WARPS             (MATMUL_Q3K_LOCAL / 32)
#define MATMUL_Q3K_OUTPUTS_PER_GROUP MATMUL_Q3K_WARPS

#define Q3K_BLOCK_ELEMENTS 256
#define Q3K_BLOCK_BYTES    110
#define X_TILE_ELEMENTS    1024

namespace {

// Full-warp inclusive reduction via shuffle-down. warpSize is 32 on
// RDNA3, so five rounds of halving cover the whole warp. Returns the
// full sum in lane 0.
__device__ __forceinline__ float warpReduceSum(float v) {
    v += __shfl_down_sync(0xffffffffu, v, 16);
    v += __shfl_down_sync(0xffffffffu, v,  8);
    v += __shfl_down_sync(0xffffffffu, v,  4);
    v += __shfl_down_sync(0xffffffffu, v,  2);
    v += __shfl_down_sync(0xffffffffu, v,  1);
    return v;
}

// Unpack the 12-byte packed scales into 16 6-bit unsigned scales in
// [0..63]. Same kmask1/kmask2 trick as llama.cpp's dequantize_row_q3_K
// (see src/compute/quant/Q3K.cpp:unpackScales for the CPU-side
// reference — bit-identical).
__device__ __forceinline__ void unpackScalesQ3K(
    const unsigned char* __restrict__ packed,
    unsigned char (&out)[16])
{
    constexpr unsigned int kmask1 = 0x03030303u;
    constexpr unsigned int kmask2 = 0x0f0f0f0fu;

    // Byte-level assembly of the three uint32_t words. The device-side
    // load is guaranteed 1-byte aligned; go through unions rather than
    // reinterpret_cast<uint32_t*> because callers may pass unaligned
    // block pointers.
    unsigned int aux0 = static_cast<unsigned int>(packed[0])       |
                       (static_cast<unsigned int>(packed[1]) <<  8) |
                       (static_cast<unsigned int>(packed[2]) << 16) |
                       (static_cast<unsigned int>(packed[3]) << 24);
    unsigned int aux1 = static_cast<unsigned int>(packed[4])       |
                       (static_cast<unsigned int>(packed[5]) <<  8) |
                       (static_cast<unsigned int>(packed[6]) << 16) |
                       (static_cast<unsigned int>(packed[7]) << 24);
    const unsigned int tmp = static_cast<unsigned int>(packed[ 8])       |
                            (static_cast<unsigned int>(packed[ 9]) <<  8) |
                            (static_cast<unsigned int>(packed[10]) << 16) |
                            (static_cast<unsigned int>(packed[11]) << 24);

    const unsigned int a2 = ((aux0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    const unsigned int a3 = ((aux1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    const unsigned int a0 = ( aux0       & kmask2) | (((tmp >> 0) & kmask1) << 4);
    const unsigned int a1 = ( aux1       & kmask2) | (((tmp >> 2) & kmask1) << 4);

    out[ 0] = static_cast<unsigned char>((a0 >>  0) & 0xFFu);
    out[ 1] = static_cast<unsigned char>((a0 >>  8) & 0xFFu);
    out[ 2] = static_cast<unsigned char>((a0 >> 16) & 0xFFu);
    out[ 3] = static_cast<unsigned char>((a0 >> 24) & 0xFFu);
    out[ 4] = static_cast<unsigned char>((a1 >>  0) & 0xFFu);
    out[ 5] = static_cast<unsigned char>((a1 >>  8) & 0xFFu);
    out[ 6] = static_cast<unsigned char>((a1 >> 16) & 0xFFu);
    out[ 7] = static_cast<unsigned char>((a1 >> 24) & 0xFFu);
    out[ 8] = static_cast<unsigned char>((a2 >>  0) & 0xFFu);
    out[ 9] = static_cast<unsigned char>((a2 >>  8) & 0xFFu);
    out[10] = static_cast<unsigned char>((a2 >> 16) & 0xFFu);
    out[11] = static_cast<unsigned char>((a2 >> 24) & 0xFFu);
    out[12] = static_cast<unsigned char>((a3 >>  0) & 0xFFu);
    out[13] = static_cast<unsigned char>((a3 >>  8) & 0xFFu);
    out[14] = static_cast<unsigned char>((a3 >> 16) & 0xFFu);
    out[15] = static_cast<unsigned char>((a3 >> 24) & 0xFFu);
}

} // namespace

extern "C" __global__ __launch_bounds__(MATMUL_Q3K_LOCAL)
void matmul_q3k_vec(
    const float*         __restrict__ X,
    const unsigned char* __restrict__ W,
          float*         __restrict__ Y,
    const int                          K,
    const int                          N)
{
    __shared__ float xTile[X_TILE_ELEMENTS];

    const int wg      = blockIdx.x;
    const int tid     = threadIdx.x;
    const int lsize   = blockDim.x;
    const int warpId  = tid / 32;
    const int laneId  = tid % 32;
    const int n       = wg * MATMUL_Q3K_OUTPUTS_PER_GROUP + warpId;
    const bool active = (n < N);
    const int nBlocks = K / Q3K_BLOCK_ELEMENTS;

    // Each lane consumes two of the 16 scales per sub-block: one for
    // its l ∈ [0..15] slot and one for its l ∈ [16..31] slot. Precompute
    // which of the two so the per-sub-block work is just an index add.
    const int lane_hi = (laneId >= 16) ? 1 : 0;

    float sum = 0.0f;

    for (int tile = 0; tile < K; tile += X_TILE_ELEMENTS) {
        const int tileK = min(X_TILE_ELEMENTS, K - tile);

        for (int i = tid; i < tileK; i += lsize) {
            xTile[i] = X[tile + i];
        }
        __syncthreads();

        if (active) {
            const unsigned char* row =
                W + static_cast<size_t>(n)
                  * static_cast<size_t>(nBlocks)
                  * Q3K_BLOCK_BYTES;

            const int blockStart   = tile / Q3K_BLOCK_ELEMENTS;
            const int blocksInTile = X_TILE_ELEMENTS / Q3K_BLOCK_ELEMENTS;
            const int blockEnd     = min(blockStart + blocksInTile, nBlocks);

            for (int b = blockStart; b < blockEnd; ++b) {
                const unsigned char* block = row + b * Q3K_BLOCK_BYTES;

                const unsigned char* hmask = block;
                const unsigned char* qs    = block + 32;
                const __half*        d_ptr =
                    reinterpret_cast<const __half*>(block + 108);
                const float d = __half2float(d_ptr[0]);

                // Scales unpack — 12 → 16 6-bit unsigned. Runs once per
                // block per lane (redundant across the warp; ~50 cheap
                // ALU ops beats a shared-memory sync).
                unsigned char scales[16];
                unpackScalesQ3K(block + 96, scales);

                // Cache the two qs bytes for this lane's l-slot (one per
                // half) and the shared hmask byte. All 8 sub-blocks reuse
                // these three values with different shift / mBit.
                const unsigned int qs_h0  = qs[laneId];       // half 0 qs[l]
                const unsigned int qs_h1  = qs[32 + laneId];  // half 1 qs[l]
                const unsigned int hm_byte = hmask[laneId];

                const int xLocalBase = (b - blockStart) * Q3K_BLOCK_ELEMENTS;

                // Unrolled 2-halves × 4-sub-blocks loop. Each iteration
                // contributes one FMA (one element per lane per sub-block).
                #pragma unroll
                for (int half = 0; half < 2; ++half) {
                    const unsigned int qs_byte = half ? qs_h1 : qs_h0;
                    const int          half_off = half * 128;

                    #pragma unroll
                    for (int sub_j = 0; sub_j < 4; ++sub_j) {
                        const int sb    = 4 * half + sub_j;
                        const int shift = 2 * sub_j;
                        const int mBit  = sb;

                        const unsigned int low2 = (qs_byte >> shift) & 0x3u;
                        const unsigned int hi1  = (hm_byte >> mBit)  & 0x1u;
                        const int qv =
                            static_cast<int>(low2) - (hi1 ? 0 : 4);

                        const int is = 2 * sb + lane_hi;
                        const float sc_val =
                            static_cast<float>(scales[is]) - 32.0f;
                        const float wv = d * sc_val * static_cast<float>(qv);

                        const int elem_off =
                            half_off + sub_j * 32 + laneId;
                        sum = __fmaf_rn(xTile[xLocalBase + elem_off],
                                        wv, sum);
                    }
                }
            }
        }

        __syncthreads();
    }

    sum = warpReduceSum(sum);

    if (active && laneId == 0) {
        Y[n] = sum;
    }
}
