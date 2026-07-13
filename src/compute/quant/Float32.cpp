#include "compute/quant/Float32.hpp"

#include <cstring>

namespace mimirmind::compute::quant {

const Float32& Float32::instance() noexcept {
    static const Float32 inst;
    return inst;
}

core::gguf::GgmlType Float32::ggmlType() const noexcept {
    return core::gguf::GgmlType::F32;
}

std::string_view Float32::name() const noexcept {
    return "F32";
}

std::size_t Float32::blockElements() const noexcept {
    return 1;
}

std::size_t Float32::blockBytes() const noexcept {
    return sizeof(float);
}

void Float32::dequantToF32(const void* src,
                           std::size_t nelements,
                           float*      dst) const {
    std::memcpy(dst, src, nelements * sizeof(float));
}

} // namespace mimirmind::compute::quant