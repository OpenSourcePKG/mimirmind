#include "compute/Dequant.hpp"

#include "compute/QuantType.hpp"
#include "compute/QuantTypeRegistry.hpp"
#include "core/log/Log.hpp"

#include <cstring>
#include <stdexcept>
#include <string>

namespace mimirmind::compute {

float halfToFloat(std::uint16_t h) noexcept {
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000U) << 16;
    const std::uint32_t exp  = (static_cast<std::uint32_t>(h) >> 10) & 0x1FU;
    const std::uint32_t mant = static_cast<std::uint32_t>(h) & 0x3FFU;

    std::uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;                          // +/- 0
        } else {
            // Subnormal half → normalised float.
            std::uint32_t m = mant;
            int e = -14;
            while ((m & 0x400U) == 0) {
                m <<= 1U;
                --e;
            }
            m &= 0x3FFU;
            bits = sign
                 | (static_cast<std::uint32_t>(e + 127) << 23)
                 | (m << 13);
        }
    } else if (exp == 0x1FU) {
        bits = sign | 0x7F800000U | (mant << 13);  // inf / nan
    } else {
        bits = sign
             | (static_cast<std::uint32_t>(exp + (127 - 15)) << 23)
             | (mant << 13);
    }

    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

float bf16ToFloat(std::uint16_t b) noexcept {
    const std::uint32_t bits = static_cast<std::uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

void dequantToF32(model::GgmlType type,
                  const void*     src,
                  std::size_t     nelements,
                  float*          dst) {
    if (const auto* qt = quantType(type); qt != nullptr) {
        qt->dequantToF32(src, nelements, dst);
        return;
    }
    MM_LOG_ERROR("dequant", "type {} not yet implemented",
                 model::typeInfo(type).name);
    throw std::runtime_error(
        "dequantToF32: ggml type '" +
        std::string{model::typeInfo(type).name} + "' not yet implemented");
}

} // namespace mimirmind::compute