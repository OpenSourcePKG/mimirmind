// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// NVFP4 -> blocked-NVFP4 repackage (load-time, one-shot, LOSSLESS).
//
// Takes the checkpoint's native NVFP4 weight (packed E2M1 nibbles + per-16
// E4M3 block scales + one F32 global) and repackages it into a single-pointer
// blocked format the matmul_nvfp4blk kernels consume — WITHOUT re-quantising.
// The e2m1 nibbles are copied verbatim; the per-16 scale global*e4m3(block) is
// folded to one fp16 per block. So this keeps the exact NVFP4 values (same as
// the BF16 the loader would otherwise materialise), just 4-bit in memory.
//
//   packed:     [rows, in/2]   U8, 2 E2M1 nibbles/byte (elem 2j low, 2j+1 high)
//   blockScale: [rows, in/16]  U8 E4M3, one per 16-element NVFP4 block
//   global:     F32 scalar
//   dst:        [rows, in/32]  super-blocks of 20 bytes:
//                 fp16 s0 | fp16 s1 | 16 packed bytes (2 NVFP4 blocks = 32 elems)
//               value[e] = s_(e/16 within super) * e2m1(nibble_e)
//               s_k = global * e4m3(blockScale)
//
// Launch: grid( rows, in/32, 1 ), block( 16, 1, 1 )   (16 = packed bytes/super)

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define NVBLK_SUPER_ELEMENTS 32
#define NVBLK_SUPER_BYTES    20   // 2*fp16 + 16 packed
#define NVBLK_PACKED_BYTES   16

namespace {

// E4M3 decode matching dequant_nvfp4.cu (bias 7; S.1111.111 = NaN).
__device__ __forceinline__ float rp_e4m3(unsigned b) {
    const unsigned s = (b >> 7) & 0x1u;
    const unsigned e = (b >> 3) & 0xFu;
    const unsigned m = b & 0x7u;
    const float sign = s ? -1.0f : 1.0f;
    if (e == 0u) return sign * ldexpf(static_cast<float>(m) / 8.0f, -6);
    if (e == 0xFu && m == 0x7u) return __int_as_float(0x7fffffff);
    return sign * ldexpf(1.0f + static_cast<float>(m) / 8.0f,
                         static_cast<int>(e) - 7);
}

} // namespace

extern "C" __global__ __launch_bounds__(16)
void repackage_nvfp4_to_blk(
    const unsigned char* __restrict__ packed,      // [rows, in/2]
    const unsigned char* __restrict__ blockScale,  // [rows, in/16]
    const float                        global,
          unsigned char* __restrict__ dst,         // [rows, in/32 * 20]
    const int                          in)         // K, multiple of 32
{
    const int row   = blockIdx.x;
    const int super = blockIdx.y;    // 32-element super-block index
    const int lid   = threadIdx.x;   // 0..15 (one packed byte each)

    const int    nSuper   = in / NVBLK_SUPER_ELEMENTS;
    const size_t dstBase  = (static_cast<size_t>(row) * nSuper + super)
                          * NVBLK_SUPER_BYTES;

    // Two E4M3 block scales -> fp16(global * e4m3) headers (threads 0/1).
    if (lid < 2) {
        const int bIdx = super * 2 + lid;   // NVFP4 block index within the row
        const unsigned char bs =
            blockScale[static_cast<size_t>(row) * (in / 16) + bIdx];
        const float s = global * rp_e4m3(bs);
        *reinterpret_cast<__half*>(dst + dstBase + lid * 2) = __float2half(s);
    }

    // Copy the 16 packed bytes (32 E2M1 nibbles) verbatim.
    const size_t srcByte = static_cast<size_t>(row) * (in / 2)
                         + static_cast<size_t>(super) * NVBLK_PACKED_BYTES + lid;
    dst[dstBase + 4 + lid] = packed[srcByte];
}
