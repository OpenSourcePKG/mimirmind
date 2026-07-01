#include "compute/quant/Q8_0.hpp"

#include "compute/Dequant.hpp"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace mimirmind::compute::quant {

const Q8_0& Q8_0::instance() noexcept {
    static const Q8_0 inst;
    return inst;
}

model::GgmlType Q8_0::ggmlType() const noexcept {
    return model::GgmlType::Q8_0;
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