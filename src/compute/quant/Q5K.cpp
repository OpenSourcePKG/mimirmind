#include "compute/quant/Q5K.hpp"

#include "compute/Dequant.hpp"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace mimirmind::compute::quant {

namespace {

/// Shared with Q4_K — the same 12-byte packed scale/min layout. Kept
/// as a file-local helper here rather than depending on Q4K.hpp so
/// the two classes stay independent.
void getScaleMinK4(int j,
                   const std::uint8_t* q,
                   std::uint8_t&       outScale,
                   std::uint8_t&       outMin) noexcept {
    if (j < 4) {
        outScale = static_cast<std::uint8_t>(q[j]     & 0x3FU);
        outMin   = static_cast<std::uint8_t>(q[j + 4] & 0x3FU);
    } else {
        outScale = static_cast<std::uint8_t>(
            (q[j + 4] & 0x0FU) | ((q[j - 4] >> 6) << 4));
        outMin   = static_cast<std::uint8_t>(
            (q[j + 4] >> 4)    | ((q[j    ] >> 6) << 4));
    }
}

} // namespace

const Q5K& Q5K::instance() noexcept {
    static const Q5K inst;
    return inst;
}

model::GgmlType Q5K::ggmlType() const noexcept {
    return model::GgmlType::Q5_K;
}

std::string_view Q5K::name() const noexcept {
    return "Q5_K";
}

std::size_t Q5K::blockElements() const noexcept {
    return kBlockElements;
}

std::size_t Q5K::blockBytes() const noexcept {
    return kBlockBytes;
}

std::string_view Q5K::gpuMatmulModule() const noexcept {
    return "matmul_q5k_vec";
}

void Q5K::dequantToF32(const void* src,
                       std::size_t nelements,
                       float*      dst) const {
    if (nelements % kBlockElements != 0) {
        throw std::runtime_error(
            "dequant Q5_K: nelements=" + std::to_string(nelements) +
            " is not a multiple of " + std::to_string(kBlockElements));
    }
    const std::size_t nblocks = nelements / kBlockElements;
    const auto* base = static_cast<const std::uint8_t*>(src);

    for (std::size_t b = 0; b < nblocks; ++b) {
        const auto* block = base + b * kBlockBytes;

        std::uint16_t dHalf, dminHalf;
        std::memcpy(&dHalf,    block,     sizeof(std::uint16_t));
        std::memcpy(&dminHalf, block + 2, sizeof(std::uint16_t));
        const float d    = halfToFloat(dHalf);
        const float dmin = halfToFloat(dminHalf);

        const std::uint8_t* scales = block + 4;      // 12 bytes
        const std::uint8_t* qh     = block + 16;     // 32 bytes
        const std::uint8_t* qs     = block + 48;     // 128 bytes

        // Eight sub-blocks of 32 elements, processed two at a time.
        // The qh bit-masks u1/u2 shift by 2 per outer iteration so
        // one 32-byte qh field covers all 256 elements.
        std::uint8_t u1 = 0x01U;
        std::uint8_t u2 = 0x02U;
        for (int j = 0; j < 8; j += 2) {
            std::uint8_t sLo, mLo, sHi, mHi;
            getScaleMinK4(j,     scales, sLo, mLo);
            getScaleMinK4(j + 1, scales, sHi, mHi);
            const float d1 = d    * static_cast<float>(sLo);
            const float m1 = dmin * static_cast<float>(mLo);
            const float d2 = d    * static_cast<float>(sHi);
            const float m2 = dmin * static_cast<float>(mHi);

            const std::uint8_t* q = qs + (j / 2) * 32;
            for (int l = 0; l < 32; ++l) {
                const std::uint32_t hi = (qh[l] & u1) ? 16U : 0U;
                *dst++ = d1 * static_cast<float>((q[l] & 0x0FU) + hi) - m1;
            }
            for (int l = 0; l < 32; ++l) {
                const std::uint32_t hi = (qh[l] & u2) ? 16U : 0U;
                *dst++ = d2 * static_cast<float>((q[l] >> 4)    + hi) - m2;
            }
            u1 = static_cast<std::uint8_t>(u1 << 2);
            u2 = static_cast<std::uint8_t>(u2 << 2);
        }
    }
}

} // namespace mimirmind::compute::quant