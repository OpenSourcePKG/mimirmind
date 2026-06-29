#include "compute/QuantTypeRegistry.hpp"

#include "compute/quant/Bfloat16.hpp"
#include "compute/quant/Float16.hpp"
#include "compute/quant/Float32.hpp"
#include "compute/quant/Q4K.hpp"
#include "compute/quant/Q6K.hpp"
#include "compute/quant/Q8_0.hpp"

#include <array>

namespace mimirmind::compute {

namespace {

std::array<const QuantType*, 6> makeRegistry() noexcept {
    return {
        &quant::Float32::instance(),
        &quant::Float16::instance(),
        &quant::Bfloat16::instance(),
        &quant::Q4K::instance(),
        &quant::Q6K::instance(),
        &quant::Q8_0::instance(),
    };
}

const std::array<const QuantType*, 6>& registry() noexcept {
    static const auto inst = makeRegistry();
    return inst;
}

} // namespace

const QuantType* quantType(model::GgmlType type) noexcept {
    switch (type) {
        case model::GgmlType::F32:  return &quant::Float32::instance();
        case model::GgmlType::F16:  return &quant::Float16::instance();
        case model::GgmlType::BF16: return &quant::Bfloat16::instance();
        case model::GgmlType::Q4_K: return &quant::Q4K::instance();
        case model::GgmlType::Q6_K: return &quant::Q6K::instance();
        case model::GgmlType::Q8_0: return &quant::Q8_0::instance();

        case model::GgmlType::F64:
        case model::GgmlType::I8:
        case model::GgmlType::I16:
        case model::GgmlType::I32:
        case model::GgmlType::I64:
        case model::GgmlType::Q4_0:
        case model::GgmlType::Q4_1:
        case model::GgmlType::Q5_0:
        case model::GgmlType::Q5_1:
        case model::GgmlType::Q8_1:
        case model::GgmlType::Q2_K:
        case model::GgmlType::Q3_K:
        case model::GgmlType::Q5_K:
        case model::GgmlType::Q8_K:
        case model::GgmlType::Unknown:
        default:
            return nullptr;
    }
}

std::span<const QuantType* const> allQuantTypes() noexcept {
    const auto& r = registry();
    return std::span<const QuantType* const>(r.data(), r.size());
}

} // namespace mimirmind::compute