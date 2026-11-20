// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/gguf/GgufTypes.hpp"

#include <cstddef>
#include <cstdint>

namespace mimirmind::hugin {

/**
 * Host F32 decode of the weights as the engine holds them, including the
 * load-time formats the GGUF registry does not know (roadmap 7.4.4.2: the
 * Kolibri-1 train block reads its frozen weights back from the engine):
 *
 *  - FP8_E4M3  34 B per 32 elements: fp16 scale + 32 E4M3 bytes
 *  - NVFP4_BLK 20 B per 32 elements: fp16 scale of elements 0..15, fp16 scale
 *              of 16..31, then 16 bytes of E2M1 nibbles (element 2j low)
 *  - NVFP4_TC  one expert of a routed bank: E2M1 nibbles [rows, in/2], the
 *              expert's CUTLASS-swizzled UE4M3 block scales (one per 16
 *              elements, BlockScaleSwizzle layout) and its F32 global
 *
 * Every other type goes through compute::dequantToF32. Hugin-only on
 * purpose: registering these in compute::QuantTypeRegistry would change the
 * engine's matmul dispatch.
 */
class HostDequant {
public:
    /// `n` elements of `type` at `src` -> F32 `dst` (not for NVFP4_TC).
    static void toF32(core::gguf::GgmlType type, const void* src, std::size_t n, float* dst);

    /// One NVFP4_TC expert [rows, in] -> F32 `dst` [rows, in].
    static void nvfp4TcToF32(const std::uint8_t* nibbles, const std::uint8_t* swizzledScales,
                             float global, std::size_t rows, std::size_t in, float* dst);
};

} // namespace mimirmind::hugin
