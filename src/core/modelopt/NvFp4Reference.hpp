// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstddef>
#include <cstdint>

namespace mimirmind::core::modelopt {

/**
 * CPU reference decode for the NVFP4 (W4A16) weight format. This is the
 * oracle every Phase-C GPU kernel is checked against — a slow, obviously
 * correct dequantiser, not a fast path.
 *
 * NVFP4 reconstruction per element:
 *   value = global_scale * e4m3(block_scale) * e2m1(nibble)
 * where the 16 elements of a block share one F8_E4M3 `block_scale`, and one
 * F32 `global_scale` applies to the whole tensor.
 *
 * The scalar decoders `e2m1ToFloat` / `e4m3ToFloat` are verified bit-exact
 * against CUTLASS's `float_e2m1_t` / `float_e4m3_t` on GB10; see the Synaipse
 * dev note.
 */

/// Decode a 4-bit E2M1 value (in the low nibble of `nib`) to float. The
/// representable magnitudes are {0, .5, 1, 1.5, 2, 3, 4, 6}; bit 3 is the
/// sign. No inf/NaN/subnormal-below-0.5.
[[nodiscard]] float e2m1ToFloat(std::uint8_t nib) noexcept;

/// Decode an 8-bit E4M3 (FP8, bias 7) byte to float. Handles subnormals;
/// 0x7F/0xFF are NaN (E4M3 has no infinities). Range ~[-448, 448].
[[nodiscard]] float e4m3ToFloat(std::uint8_t byte) noexcept;

/**
 * Dequantise a `[rows, in]` NVFP4 weight to row-major f32.
 *   packed      : U8      [rows, in/2]   two E2M1 nibbles per byte, element
 *                                        2j in the low nibble, 2j+1 in high
 *   blockScale  : F8_E4M3 [rows, in/16]  one scale per 16-element block
 *   globalScale : F32 scalar
 *   out         : f32     [rows, in]
 * `in` must be a multiple of 16.
 */
void dequantNvfp4(const std::uint8_t* packed,
                  const std::uint8_t* blockScale,
                  float               globalScale,
                  std::uint64_t       rows,
                  std::uint64_t       in,
                  float*              out) noexcept;

/**
 * Dequantise an FP8 (E4M3) weight to f32. This is the attention-projection
 * format on the W4A16 checkpoint: an unpacked F8_E4M3 weight plus one F32
 * per-tensor `weight_scale` (the `input_scale` is for activation quant and
 * is unused on the weight-only path).
 *   weight : F8_E4M3 [n]   (row-major, one byte per element)
 *   out    : f32      [n]
 *   out[i] = weightScale * e4m3(weight[i])
 */
void dequantFp8(const std::uint8_t* weight,
                float               weightScale,
                std::uint64_t       n,
                float*              out) noexcept;

} // namespace mimirmind::core::modelopt