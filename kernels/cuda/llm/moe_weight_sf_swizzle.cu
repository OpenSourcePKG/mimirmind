// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling
//
// M-Cuda.MoeGroup Sub-Step E-d.2b — device swizzle of an NVFP4 weight
// block-scale (SF) tensor into the CUTLASS 128x4 layout the block-scaled
// grouped GEMM consumes (SFB). One expert's `[rows, ksf]` F8_E4M3 scales
// (row-major, ksf = in_features / 16) -> its swizzled slot in the SFB bank.
//
// The offset formula is byte-identical to the host
// mimirmind::core::modelopt::swizzledScaleOffset (BlockScaleSwizzle.hpp) and to
// the act-quant kernel's aq_sf_offset — all three verified bit-exact against
// cute's tile_atom_to_shape_SFA/SFB on GB10. Doing it on the device avoids a
// D2H/H2D round-trip at load time.
//
//   src:  [rows, ksf]  U8 F8_E4M3, row-major
//   dst:  swizzled slot; the caller MUST pre-zero it (padding rows/cols to
//         round_up(rows,128) / round_up(ksf,4) are not visited here).
//
// Launch: grid( rows, ceil(ksf / SF_SWIZZLE_LOCAL), 1 ), block( SF_SWIZZLE_LOCAL ).

#include <cuda_runtime.h>

#ifndef SF_SWIZZLE_LOCAL
#define SF_SWIZZLE_LOCAL 128
#endif

extern "C" __global__ __launch_bounds__(SF_SWIZZLE_LOCAL)
void moe_weight_sf_swizzle(
    const unsigned char* __restrict__ src,   // [rows, ksf]
          unsigned char* __restrict__ dst,   // swizzled (pre-zeroed)
    const int rows,
    const int ksf)
{
    const int r = blockIdx.x;
    const int s = blockIdx.y * blockDim.x + threadIdx.x;
    if (r >= rows || s >= ksf) {
        return;
    }
    const int numKTiles = (ksf + 3) / 4;

    // == swizzledScaleOffset(r, s, *, numKTiles): atom (128x4), K-tile fast.
    const int  a_m = r & 127;            // r % 128
    const long b_m = r >> 7;             // r / 128
    const int  a_s = s & 3;              // s % 4
    const long b_s = s >> 2;             // s / 4
    const long atomOff =
        static_cast<long>((a_m & 31)) * 16 + static_cast<long>(a_m >> 5) * 4 + a_s;
    const long off = atomOff + 512L * (b_s + static_cast<long>(numKTiles) * b_m);

    dst[off] = src[static_cast<long>(r) * ksf + s];
}
