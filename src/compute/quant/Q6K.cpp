// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "compute/quant/Q6K.hpp"

#include "compute/Dequant.hpp"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace mimirmind::compute::quant {

const Q6K& Q6K::instance() noexcept {
    static const Q6K inst;
    return inst;
}

core::gguf::GgmlType Q6K::ggmlType() const noexcept {
    return core::gguf::GgmlType::Q6_K;
}

std::string_view Q6K::name() const noexcept {
    return "Q6_K";
}

std::size_t Q6K::blockElements() const noexcept {
    return kBlockElements;
}

std::size_t Q6K::blockBytes() const noexcept {
    return kBlockBytes;
}

std::string_view Q6K::gpuMatmulModule() const noexcept {
    return "matmul_q6k_vec";
}

std::string_view Q6K::gpuMatmulGemmModule() const noexcept {
    return "matmul_q6k_gemm";
}

std::size_t Q6K::gpuMatmulGemmMTile() const noexcept {
    return kGemmMTile;
}

void Q6K::dequantToF32(const void* src,
                       std::size_t nelements,
                       float*      dst) const {
    if (nelements % kBlockElements != 0) {
        throw std::runtime_error(
            "dequant Q6_K: nelements=" + std::to_string(nelements) +
            " is not a multiple of " + std::to_string(kBlockElements));
    }
    const std::size_t nblocks = nelements / kBlockElements;
    const auto* base = static_cast<const std::uint8_t*>(src);

    for (std::size_t b = 0; b < nblocks; ++b) {
        const auto* block = base + b * kBlockBytes;

        const std::uint8_t* ql = block;            // 128 bytes
        const std::uint8_t* qh = block + 128;      // 64 bytes
        const auto*         sc = reinterpret_cast<const std::int8_t*>(block + 192); // 16 bytes
        std::uint16_t dHalf;
        std::memcpy(&dHalf, block + 208, sizeof(std::uint16_t));
        const float d = halfToFloat(dHalf);

        float*       y          = dst;
        const std::uint8_t* qlp = ql;
        const std::uint8_t* qhp = qh;
        const std::int8_t*  scp = sc;

        // Two 128-element halves per super-block.
        for (int half = 0; half < 2; ++half) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const auto q1 = static_cast<std::int8_t>(
                    (qlp[l +  0] & 0x0FU) | (((qhp[l] >> 0) & 0x03U) << 4)) - 32;
                const auto q2 = static_cast<std::int8_t>(
                    (qlp[l + 32] & 0x0FU) | (((qhp[l] >> 2) & 0x03U) << 4)) - 32;
                const auto q3 = static_cast<std::int8_t>(
                    (qlp[l +  0] >> 4U)   | (((qhp[l] >> 4) & 0x03U) << 4)) - 32;
                const auto q4 = static_cast<std::int8_t>(
                    (qlp[l + 32] >> 4U)   | (((qhp[l] >> 6) & 0x03U) << 4)) - 32;
                y[l +  0] = d * static_cast<float>(scp[is + 0]) * static_cast<float>(q1);
                y[l + 32] = d * static_cast<float>(scp[is + 2]) * static_cast<float>(q2);
                y[l + 64] = d * static_cast<float>(scp[is + 4]) * static_cast<float>(q3);
                y[l + 96] = d * static_cast<float>(scp[is + 6]) * static_cast<float>(q4);
            }
            y   += 128;
            qlp += 64;
            qhp += 32;
            scp += 8;
        }

        dst += kBlockElements;
    }
}

} // namespace mimirmind::compute::quant