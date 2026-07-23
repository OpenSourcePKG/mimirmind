// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/safetensors/SafetensorsDtype.hpp"

namespace mimirmind::core::safetensors {

SafetensorsDtype dtypeFromString(std::string_view s) noexcept {
    if (s == "F32")     return SafetensorsDtype::F32;
    if (s == "F16")     return SafetensorsDtype::F16;
    if (s == "BF16")    return SafetensorsDtype::BF16;
    if (s == "F8_E4M3") return SafetensorsDtype::F8_E4M3;
    if (s == "F8_E5M2") return SafetensorsDtype::F8_E5M2;
    if (s == "U8")      return SafetensorsDtype::U8;
    if (s == "I8")      return SafetensorsDtype::I8;
    if (s == "I16")     return SafetensorsDtype::I16;
    if (s == "I32")     return SafetensorsDtype::I32;
    if (s == "I64")     return SafetensorsDtype::I64;
    if (s == "BOOL")    return SafetensorsDtype::Bool;
    return SafetensorsDtype::Unknown;
}

std::size_t dtypeWidth(SafetensorsDtype d) noexcept {
    switch (d) {
        case SafetensorsDtype::F32:     return 4;
        case SafetensorsDtype::F16:     return 2;
        case SafetensorsDtype::BF16:    return 2;
        case SafetensorsDtype::F8_E4M3: return 1;
        case SafetensorsDtype::F8_E5M2: return 1;
        case SafetensorsDtype::U8:      return 1;
        case SafetensorsDtype::I8:      return 1;
        case SafetensorsDtype::I16:     return 2;
        case SafetensorsDtype::I32:     return 4;
        case SafetensorsDtype::I64:     return 8;
        case SafetensorsDtype::Bool:    return 1;
        case SafetensorsDtype::Unknown:
        default:                        return 0;
    }
}

std::string_view dtypeName(SafetensorsDtype d) noexcept {
    switch (d) {
        case SafetensorsDtype::F32:     return "F32";
        case SafetensorsDtype::F16:     return "F16";
        case SafetensorsDtype::BF16:    return "BF16";
        case SafetensorsDtype::F8_E4M3: return "F8_E4M3";
        case SafetensorsDtype::F8_E5M2: return "F8_E5M2";
        case SafetensorsDtype::U8:      return "U8";
        case SafetensorsDtype::I8:      return "I8";
        case SafetensorsDtype::I16:     return "I16";
        case SafetensorsDtype::I32:     return "I32";
        case SafetensorsDtype::I64:     return "I64";
        case SafetensorsDtype::Bool:    return "BOOL";
        case SafetensorsDtype::Unknown:
        default:                        return "UNKNOWN";
    }
}

} // namespace mimirmind::core::safetensors