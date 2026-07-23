// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstddef>
#include <cstdint>

namespace mimirmind::core::modelopt {

/**
 * On-load swizzle of an NVFP4 block-scale (SF) tensor into the layout the
 * Blackwell sm_120 block-scaled GEMM consumes.
 *
 * ModelOpt stores the per-16-element block scales row-major as
 * `[rows, ksf]` F8_E4M3 (1 byte per scale, `ksf = in_features / 16`). The
 * CUTLASS sm_120 collective (`Sm1xxBlockScaledConfig`, the 79a/79b NVFP4
 * path) expects the scales in a swizzled "128x4" layout instead. Feeding
 * row-major scales to that kernel produces silently wrong results, so the
 * transform is a load-time step, not a perf-later item.
 *
 * The exact offset was derived from CUTLASS's K-major SF atom
 *   `((32,4),(16,4)) : ((16,4),(0,1))`, Blk_MN=128, Blk_SF=4
 * and **verified bit-exact against `cute` `tile_atom_to_shape_SFA`** on GB10
 * (sm_120a, CUDA 13) across shapes up to 248320x2048 — 31.8M coordinates,
 * zero mismatches. See the Synaipse dev note for the verification harness.
 *
 * Storage offset of scale-block `s` (0..ksf-1) of row `m` (0..rows-1), with
 * `mTiles = ceil(rows/128)`, `ksfTiles = ceil(ksf/4)`:
 *   a_m = m % 128; b_m = m / 128
 *   a_s = s %  4;  b_s = s /  4
 *   atomOff = (a_m % 32)*16 + (a_m / 32)*4 + a_s
 *   offset  = atomOff + 512 * (b_s + ksfTiles * b_m)
 * The rows dimension is padded to a multiple of 128 and ksf to a multiple of
 * 4; padding scales are zero.
 */

/// Number of 128x4 tiles along the padded rows dimension.
[[nodiscard]] constexpr std::uint64_t sfRowTiles(std::uint64_t rows) noexcept {
    return (rows + 127) / 128;
}

/// Number of tiles along the padded scale-block (ksf) dimension.
[[nodiscard]] constexpr std::uint64_t sfKTiles(std::uint64_t ksf) noexcept {
    return (ksf + 3) / 4;
}

/// Byte size of the swizzled SF tensor (F8_E4M3, 1 byte/scale): one full
/// 128x4 atom (512 scales) per tile.
[[nodiscard]] constexpr std::size_t swizzledBlockScaleBytes(std::uint64_t rows,
                                                            std::uint64_t ksf) noexcept {
    return static_cast<std::size_t>(512) * sfRowTiles(rows) * sfKTiles(ksf);
}

/// Storage offset (in scales == bytes) of scale-block `s` of row `m`.
[[nodiscard]] constexpr std::size_t
swizzledScaleOffset(std::uint64_t m, std::uint64_t s,
                    std::uint64_t mTiles, std::uint64_t ksfTiles) noexcept {
    (void)mTiles; // rows-tile stride is ksfTiles * 512 (K-tile is the fast tile)
    const std::uint64_t a_m = m % 128, b_m = m / 128;
    const std::uint64_t a_s = s % 4,   b_s = s / 4;
    const std::uint64_t atomOff = (a_m % 32) * 16 + (a_m / 32) * 4 + a_s;
    return static_cast<std::size_t>(atomOff + 512 * (b_s + ksfTiles * b_m));
}

/**
 * Swizzle a row-major `[rows, ksf]` F8_E4M3 block-scale tensor (`src`,
 * `rows*ksf` bytes) into the CUTLASS SF layout at `dst`
 * (`swizzledBlockScaleBytes(rows, ksf)` bytes). `dst` is fully written:
 * every mapped scale is copied and all padding bytes are zeroed. `src` and
 * `dst` must not alias.
 */
void swizzleBlockScale(const std::uint8_t* src, std::uint64_t rows,
                       std::uint64_t ksf, std::uint8_t* dst) noexcept;

} // namespace mimirmind::core::modelopt