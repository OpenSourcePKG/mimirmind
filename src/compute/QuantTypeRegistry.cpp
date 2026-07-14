// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "compute/QuantTypeRegistry.hpp"

#include "compute/quant/Bfloat16.hpp"
#include "compute/quant/Float16.hpp"
#include "compute/quant/Float32.hpp"
#include "compute/quant/Q4K.hpp"
#include "compute/quant/Q5K.hpp"
#include "compute/quant/Q6K.hpp"
#include "compute/quant/Q8_0.hpp"

#include <array>

namespace mimirmind::compute {

namespace {

std::array<const QuantType*, 7> makeRegistry() noexcept {
    return {
        &quant::Float32::instance(),
        &quant::Float16::instance(),
        &quant::Bfloat16::instance(),
        &quant::Q4K::instance(),
        &quant::Q5K::instance(),
        &quant::Q6K::instance(),
        &quant::Q8_0::instance(),
    };
}

const std::array<const QuantType*, 7>& registry() noexcept {
    static const auto inst = makeRegistry();
    return inst;
}

} // namespace

const QuantType* quantType(core::gguf::GgmlType type) noexcept {
    switch (type) {
        case core::gguf::GgmlType::F32:  return &quant::Float32::instance();
        case core::gguf::GgmlType::F16:  return &quant::Float16::instance();
        case core::gguf::GgmlType::BF16: return &quant::Bfloat16::instance();
        case core::gguf::GgmlType::Q4_K: return &quant::Q4K::instance();
        case core::gguf::GgmlType::Q5_K: return &quant::Q5K::instance();
        case core::gguf::GgmlType::Q6_K: return &quant::Q6K::instance();
        case core::gguf::GgmlType::Q8_0: return &quant::Q8_0::instance();

        case core::gguf::GgmlType::F64:
        case core::gguf::GgmlType::I8:
        case core::gguf::GgmlType::I16:
        case core::gguf::GgmlType::I32:
        case core::gguf::GgmlType::I64:
        case core::gguf::GgmlType::Q4_0:
        case core::gguf::GgmlType::Q4_1:
        case core::gguf::GgmlType::Q5_0:
        case core::gguf::GgmlType::Q5_1:
        case core::gguf::GgmlType::Q8_1:
        case core::gguf::GgmlType::Q2_K:
        case core::gguf::GgmlType::Q3_K:
        case core::gguf::GgmlType::Q8_K:
        case core::gguf::GgmlType::Unknown:
        default:
            return nullptr;
    }
}

std::span<const QuantType* const> allQuantTypes() noexcept {
    const auto& r = registry();
    return std::span<const QuantType* const>(r.data(), r.size());
}

} // namespace mimirmind::compute