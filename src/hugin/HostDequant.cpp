// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "hugin/HostDequant.hpp"

#include "compute/Dequant.hpp"
#include "core/modelopt/BlockScaleSwizzle.hpp"
#include "core/modelopt/NvFp4Reference.hpp"

#include <cstring>
#include <stdexcept>
#include <string>

namespace mimirmind::hugin {

namespace {

float fp16At(const std::uint8_t* p) {
    std::uint16_t h = 0;
    std::memcpy(&h, p, sizeof(h));
    return compute::halfToFloat(h);
}

void requireBlocks(std::size_t n, const char* type) {
    if (n % 32 != 0) {
        throw std::runtime_error(std::string("HostDequant: ") + type + " needs a multiple of 32 elements");
    }
}

} // namespace

void HostDequant::toF32(core::gguf::GgmlType type, const void* src, std::size_t n, float* dst) {
    const auto* b = static_cast<const std::uint8_t*>(src);
    switch (type) {
        case core::gguf::GgmlType::FP8_E4M3: {
            requireBlocks(n, "FP8_E4M3");
            for (std::size_t blk = 0; blk < n / 32; ++blk) {
                const std::uint8_t* p = b + blk * 34;
                const float s = fp16At(p);
                for (std::size_t i = 0; i < 32; ++i) dst[blk * 32 + i] = s * core::modelopt::e4m3ToFloat(p[2 + i]);
            }
            return;
        }
        case core::gguf::GgmlType::NVFP4_BLK: {
            requireBlocks(n, "NVFP4_BLK");
            for (std::size_t blk = 0; blk < n / 32; ++blk) {
                const std::uint8_t* p = b + blk * 20;
                const float s0 = fp16At(p), s1 = fp16At(p + 2);
                for (std::size_t j = 0; j < 16; ++j) {
                    const float s = j < 8 ? s0 : s1;
                    dst[blk * 32 + 2 * j]     = s * core::modelopt::e2m1ToFloat(p[4 + j] & 0x0FU);
                    dst[blk * 32 + 2 * j + 1] = s * core::modelopt::e2m1ToFloat(static_cast<std::uint8_t>(p[4 + j] >> 4));
                }
            }
            return;
        }
        case core::gguf::GgmlType::NVFP4_TC:
            throw std::runtime_error("HostDequant: NVFP4_TC needs its side banks (nvfp4TcToF32)");
        default:
            compute::dequantToF32(type, src, n, dst);
            return;
    }
}

void HostDequant::nvfp4TcToF32(const std::uint8_t* nibbles, const std::uint8_t* swizzledScales, float global,
                               std::size_t rows, std::size_t in, float* dst) {
    if (in % 16 != 0) throw std::runtime_error("HostDequant: NVFP4_TC needs in % 16 == 0");
    const std::uint64_t ksf = in / 16;
    const std::uint64_t mTiles = core::modelopt::sfRowTiles(rows), kTiles = core::modelopt::sfKTiles(ksf);
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t s = 0; s < ksf; ++s) {
            const float scale =
                global * core::modelopt::e4m3ToFloat(swizzledScales[core::modelopt::swizzledScaleOffset(r, s, mTiles, kTiles)]);
            const std::uint8_t* p = nibbles + r * (in / 2) + s * 8;
            float* o = dst + r * in + s * 16;
            for (std::size_t j = 0; j < 8; ++j) {
                o[2 * j]     = scale * core::modelopt::e2m1ToFloat(p[j] & 0x0FU);
                o[2 * j + 1] = scale * core::modelopt::e2m1ToFloat(static_cast<std::uint8_t>(p[j] >> 4));
            }
        }
    }
}

} // namespace mimirmind::hugin
