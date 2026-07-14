// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "compute/quant/Q8_0.hpp"

#include "compute/Dequant.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace mimirmind::compute::quant {

namespace {

/// IEEE-754 binary32 → binary16 (round-to-nearest-even). Scalar
/// implementation is fine — only invoked at load time, once per Q8_0
/// block during requantization.
std::uint16_t floatToHalfBits(float f) noexcept {
    std::uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    const std::uint32_t sign = (bits >> 16) & 0x8000U;
    std::int32_t        exp  = static_cast<std::int32_t>((bits >> 23) & 0xFFU);
    const std::uint32_t mant = bits & 0x7FFFFFU;

    if (exp == 0xFF) {
        return static_cast<std::uint16_t>(
            sign | 0x7C00U | (mant != 0 ? 0x0200U : 0U));
    }
    exp -= 127 - 15;
    if (exp <= 0) {
        if (exp < -10) return static_cast<std::uint16_t>(sign);
        const std::uint32_t m = (mant | 0x800000U) >> (14 - exp);
        return static_cast<std::uint16_t>(sign | m);
    }
    if (exp >= 0x1F) {
        return static_cast<std::uint16_t>(sign | 0x7C00U);
    }
    return static_cast<std::uint16_t>(sign
        | (static_cast<std::uint32_t>(exp) << 10)
        | (mant >> 13));
}

} // namespace

const Q8_0& Q8_0::instance() noexcept {
    static const Q8_0 inst;
    return inst;
}

core::gguf::GgmlType Q8_0::ggmlType() const noexcept {
    return core::gguf::GgmlType::Q8_0;
}

std::string_view Q8_0::name() const noexcept {
    return "Q8_0";
}

std::size_t Q8_0::blockElements() const noexcept {
    return kBlockElements;
}

std::size_t Q8_0::blockBytes() const noexcept {
    return kBlockBytes;
}

std::string_view Q8_0::gpuMatmulModule() const noexcept {
    return "matmul_q8_0_vec";
}

std::string_view Q8_0::gpuMatmulGemmModule() const noexcept {
    return "matmul_q8_0_gemm";
}

std::size_t Q8_0::gpuMatmulGemmMTile() const noexcept {
    return kGemmMTile;
}

void Q8_0::quantizeRow(const float* src, std::size_t K, void* dst) noexcept {
    // K must be a multiple of 32 — caller guarantees this (typical rows
    // are 2560/2048/512 all divisible by 32). If K is not aligned the
    // trailing elements would be silently ignored; instead just clamp
    // to a whole-block count here.
    const std::size_t nblocks = K / kBlockElements;
    auto* dstBytes = static_cast<std::uint8_t*>(dst);

    for (std::size_t b = 0; b < nblocks; ++b) {
        const float* srcBlock = src + b * kBlockElements;
        std::uint8_t* dstBlock = dstBytes + b * kBlockBytes;

        float absMax = 0.0F;
        for (std::size_t i = 0; i < kBlockElements; ++i) {
            absMax = std::max(absMax, std::fabs(srcBlock[i]));
        }
        const float scale = (absMax > 0.0F) ? (absMax / 127.0F) : 0.0F;
        const std::uint16_t dHalf = floatToHalfBits(scale);
        std::memcpy(dstBlock, &dHalf, 2);

        const float invScale = (scale > 0.0F) ? (1.0F / scale) : 0.0F;
        auto* qs = reinterpret_cast<std::int8_t*>(dstBlock + 2);
        for (std::size_t i = 0; i < kBlockElements; ++i) {
            const int q  = static_cast<int>(std::lround(srcBlock[i] * invScale));
            const int qc = std::clamp(q, -127, 127);
            qs[i] = static_cast<std::int8_t>(qc);
        }
    }
}

void Q8_0::reorderRow(const void* nativeRow,
                      std::size_t K,
                      void*       reorderedRow) noexcept {
    const std::size_t nblocks = K / kBlockElements;
    const auto* src = static_cast<const std::uint8_t*>(nativeRow);
    auto*       dst = static_cast<std::uint8_t*>(reorderedRow);

    std::uint8_t* dstScales = dst;                    // 2 * nblocks bytes
    std::uint8_t* dstQuants = dst + 2 * nblocks;      // 32 * nblocks bytes

    for (std::size_t b = 0; b < nblocks; ++b) {
        const std::uint8_t* srcBlock = src + b * kBlockBytes;
        std::memcpy(dstScales + b * 2,      srcBlock,       2);
        std::memcpy(dstQuants + b * 32,     srcBlock + 2,   32);
    }
}

void Q8_0::unreorderRow(const void* reorderedRow,
                        std::size_t K,
                        void*       nativeRow) noexcept {
    const std::size_t nblocks = K / kBlockElements;
    const auto* src = static_cast<const std::uint8_t*>(reorderedRow);
    auto*       dst = static_cast<std::uint8_t*>(nativeRow);

    const std::uint8_t* srcScales = src;
    const std::uint8_t* srcQuants = src + 2 * nblocks;

    for (std::size_t b = 0; b < nblocks; ++b) {
        std::uint8_t* dstBlock = dst + b * kBlockBytes;
        std::memcpy(dstBlock,     srcScales + b * 2,  2);
        std::memcpy(dstBlock + 2, srcQuants + b * 32, 32);
    }
}

void Q8_0::reorderMatrixInPlace(void*       base,
                                std::size_t N,
                                std::size_t K,
                                void*       rowScratch) noexcept {
    const std::size_t rowB = rowBytes(K);
    auto* baseBytes = static_cast<std::uint8_t*>(base);
    for (std::size_t n = 0; n < N; ++n) {
        std::uint8_t* row = baseBytes + n * rowB;
        std::memcpy(rowScratch, row, rowB);
        reorderRow(rowScratch, K, row);
    }
}

void Q8_0::unreorderMatrixInPlace(void*       base,
                                  std::size_t N,
                                  std::size_t K,
                                  void*       rowScratch) noexcept {
    const std::size_t rowB = rowBytes(K);
    auto* baseBytes = static_cast<std::uint8_t*>(base);
    for (std::size_t n = 0; n < N; ++n) {
        std::uint8_t* row = baseBytes + n * rowB;
        std::memcpy(rowScratch, row, rowB);
        unreorderRow(rowScratch, K, row);
    }
}

void Q8_0::dequantRowFromReorderedToF32(const void* reorderedRow,
                                        std::size_t K,
                                        float*      dst) noexcept {
    const std::size_t nblocks = K / kBlockElements;
    const auto* src = static_cast<const std::uint8_t*>(reorderedRow);

    const std::uint8_t* scales = src;
    const auto*         quants = reinterpret_cast<const std::int8_t*>(
        src + 2 * nblocks);

    for (std::size_t b = 0; b < nblocks; ++b) {
        std::uint16_t dHalf;
        std::memcpy(&dHalf, scales + b * 2, sizeof(std::uint16_t));
        const float d = halfToFloat(dHalf);

        const std::int8_t* qs = quants + b * kBlockElements;
        for (std::size_t l = 0; l < kBlockElements; ++l) {
            *dst++ = d * static_cast<float>(qs[l]);
        }
    }
}

void Q8_0::dequantToF32(const void* src,
                        std::size_t nelements,
                        float*      dst) const {
    if (nelements % kBlockElements != 0) {
        throw std::runtime_error(
            "dequant Q8_0: nelements=" + std::to_string(nelements) +
            " is not a multiple of " + std::to_string(kBlockElements));
    }
    const std::size_t nblocks = nelements / kBlockElements;
    const auto* base = static_cast<const std::uint8_t*>(src);

    for (std::size_t b = 0; b < nblocks; ++b) {
        const auto* block = base + b * kBlockBytes;

        std::uint16_t dHalf;
        std::memcpy(&dHalf, block, sizeof(std::uint16_t));
        const float d = halfToFloat(dHalf);

        const auto* qs = reinterpret_cast<const std::int8_t*>(block + 2);
        for (std::size_t l = 0; l < kBlockElements; ++l) {
            *dst++ = d * static_cast<float>(qs[l]);
        }
    }
}

} // namespace mimirmind::compute::quant