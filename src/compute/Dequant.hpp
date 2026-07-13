#pragma once

#include "core/gguf/GgufTypes.hpp"

#include <cstddef>
#include <cstdint>

namespace mimirmind::compute {

/**
 * Convert `nelements` consecutive elements of `src` (in GGUF layout for
 * `type`) into F32 at `dst`. The caller is responsible for both pointers
 * being valid for the read/write extent.
 *
 * For block-quantised types (Q4_K, Q6_K, ...) `nelements` must be a
 * multiple of the type's block size and `src` must point at the start of
 * a block. Throws std::runtime_error on unsupported type or misaligned
 * `nelements`.
 *
 * Supported now: F32 (passthrough), F16, BF16, Q4_K. Other types will be
 * added when a forward-pass step needs them.
 */
void dequantToF32(core::gguf::GgmlType type,
                  const void*     src,
                  std::size_t     nelements,
                  float*          dst);

/// Single half → float (IEEE-754 binary16 → binary32). Used by the
/// quantised dequant paths.
[[nodiscard]] float halfToFloat(std::uint16_t h) noexcept;

/// Single bfloat16 → float. The MSB-16 of a binary32, shifted into place.
[[nodiscard]] float bf16ToFloat(std::uint16_t b) noexcept;

} // namespace mimirmind::compute