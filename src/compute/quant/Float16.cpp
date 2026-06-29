#include "compute/quant/Float16.hpp"

#include "compute/Dequant.hpp"

#include <cstdint>

namespace mimirmind::compute::quant {

const Float16& Float16::instance() noexcept {
    static const Float16 inst;
    return inst;
}

model::GgmlType Float16::ggmlType() const noexcept {
    return model::GgmlType::F16;
}

std::string_view Float16::name() const noexcept {
    return "F16";
}

std::size_t Float16::blockElements() const noexcept {
    return 1;
}

std::size_t Float16::blockBytes() const noexcept {
    return sizeof(std::uint16_t);
}

void Float16::dequantToF32(const void* src,
                           std::size_t nelements,
                           float*      dst) const {
    const auto* in = static_cast<const std::uint16_t*>(src);
    for (std::size_t i = 0; i < nelements; ++i) {
        dst[i] = halfToFloat(in[i]);
    }
}

} // namespace mimirmind::compute::quant