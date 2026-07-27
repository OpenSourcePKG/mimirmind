// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

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

std::string_view Float32::gpuMatmulModule() const noexcept {
    // Dense F32 matvec. Without this GpuMatmul has no F32 entry and
    // matmulAsync falls back to a host CPU matmul — not command-list-
    // replay-safe (see kernels/matmul_f32_vec.cl for the MoE-router bug).
    return "matmul_f32_vec";
}

void Float32::dequantToF32(const void* src,
                           std::size_t nelements,
                           float*      dst) const {
    std::memcpy(dst, src, nelements * sizeof(float));
}

} // namespace mimirmind::compute::quant