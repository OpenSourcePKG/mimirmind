// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "compute/quant/Q3K.hpp"

#include "compute/Dequant.hpp"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace mimirmind::compute::quant {

namespace {

// Scales unpack: the 12-byte `scales[]` field encodes 16 6-bit
// unsigned scales via the same trick llama.cpp uses in
// `dequantize_row_q3_K`. After the unpack `aux[]` holds 16 bytes each
// in [0..63]; the caller subtracts 32 per lane to get signed [-32..31].
//
// Kept verbatim to preserve bit-parity against the reference; see
// `ggml-quants.c` `dequantize_row_q3_K`.
inline void unpackScales(const std::uint8_t (&packed)[12],
                         std::uint8_t (&outScales)[16]) noexcept {
    constexpr std::uint32_t kmask1 = 0x03030303U;
    constexpr std::uint32_t kmask2 = 0x0f0f0f0fU;

    std::uint32_t aux[4]{};
    std::memcpy(aux, packed, 12);
    const std::uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0]         & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1]         & kmask2) | (((tmp >> 2) & kmask1) << 4);
    std::memcpy(outScales, aux, 16);
}

} // namespace

const Q3K& Q3K::instance() noexcept {
    static const Q3K inst;
    return inst;
}

core::gguf::GgmlType Q3K::ggmlType() const noexcept {
    return core::gguf::GgmlType::Q3_K;
}

std::string_view Q3K::name() const noexcept {
    return "Q3_K";
}

std::size_t Q3K::blockElements() const noexcept {
    return kBlockElements;
}

std::size_t Q3K::blockBytes() const noexcept {
    return kBlockBytes;
}

std::string_view Q3K::gpuMatmulModule() const noexcept {
    // The abstract module name is only consumed by the L0 loader
    // (src/compute/l0/GpuMatmul.cpp:139), which iterates all quant
    // types at ctx-init and eager-loads their SPV. There is no
    // Q3_K L0 kernel today — return the empty sentinel so L0 skips
    // us and falls back to the CPU matmul path. The HIP native
    // kernel (kernels_hip/matmul_q3k_vec.hip) is loaded by name
    // directly in HipGpuMatmul::Impl's constructor, so this empty
    // string does not disable it there.
    return "";
}

std::string_view Q3K::gpuMatmulGemmModule() const noexcept {
    // No GEMM variant today; empty sentinel keeps GpuMatmul dispatch
    // on the vec path regardless of M.
    return "";
}

std::size_t Q3K::gpuMatmulGemmMTile() const noexcept {
    return kGemmMTile;
}

void Q3K::dequantToF32(const void* src,
                       std::size_t nelements,
                       float*      dst) const {
    if (nelements % kBlockElements != 0) {
        throw std::runtime_error(
            "dequant Q3_K: nelements=" + std::to_string(nelements) +
            " is not a multiple of " + std::to_string(kBlockElements));
    }
    const std::size_t nblocks = nelements / kBlockElements;
    const auto* base = static_cast<const std::uint8_t*>(src);

    for (std::size_t b = 0; b < nblocks; ++b) {
        const auto* block = base + b * kBlockBytes;

        const std::uint8_t* hmask = block;           // 32 bytes
        const std::uint8_t* qs    = block + 32;      // 64 bytes

        std::uint8_t packedScales[12];
        std::memcpy(packedScales, block + 96, 12);
        std::uint8_t scales[16];
        unpackScales(packedScales, scales);

        std::uint16_t dHalf;
        std::memcpy(&dHalf, block + 108, sizeof(std::uint16_t));
        const float d = halfToFloat(dHalf);

        // Reference-style loop (mirrors `dequantize_row_q3_K` verbatim
        // so a future refactor lands next to a bit-parity test rather
        // than a hand-tuned reordering that drifts silently). Two
        // 128-element halves; each half has 4 sub-blocks of 32
        // elements, and each sub-block consumes two consecutive scales.
        int is    = 0;
        int mBit  = 0;   // bit index into hmask[l] for the current sub-block
        const std::uint8_t* q = qs;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                const float dl0 = d * (static_cast<int>(scales[is++]) - 32);
                for (int l = 0; l < 16; ++l) {
                    const int lowBits = (q[l] >> shift) & 0x3;
                    const int hiBit   = (hmask[l] >> mBit) & 0x1;
                    const int qv      = lowBits - (hiBit ? 0 : 4);
                    *dst++ = dl0 * static_cast<float>(qv);
                }
                const float dl1 = d * (static_cast<int>(scales[is++]) - 32);
                for (int l = 0; l < 16; ++l) {
                    const int lowBits = (q[l + 16] >> shift) & 0x3;
                    const int hiBit   = (hmask[l + 16] >> mBit) & 0x1;
                    const int qv      = lowBits - (hiBit ? 0 : 4);
                    *dst++ = dl1 * static_cast<float>(qv);
                }
                shift += 2;
                ++mBit;
            }
            q += 32;
        }
    }
}

} // namespace mimirmind::compute::quant
