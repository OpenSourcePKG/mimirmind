#include "core/gguf/GgufTypes.hpp"

namespace mimirmind::model {

GgmlTypeInfo typeInfo(GgmlType t) noexcept {
    // Block size + type size pairs come from ggml/src/ggml.c (ggml_type_traits).
    // Quantised K-types all use a super-block of 256 elements.
    switch (t) {
        case GgmlType::F32:  return {"F32",  1,   4};
        case GgmlType::F16:  return {"F16",  1,   2};
        case GgmlType::BF16: return {"BF16", 1,   2};
        case GgmlType::F64:  return {"F64",  1,   8};
        case GgmlType::I8:   return {"I8",   1,   1};
        case GgmlType::I16:  return {"I16",  1,   2};
        case GgmlType::I32:  return {"I32",  1,   4};
        case GgmlType::I64:  return {"I64",  1,   8};

        case GgmlType::Q4_0: return {"Q4_0", 32,  18};
        case GgmlType::Q4_1: return {"Q4_1", 32,  20};
        case GgmlType::Q5_0: return {"Q5_0", 32,  22};
        case GgmlType::Q5_1: return {"Q5_1", 32,  24};
        case GgmlType::Q8_0: return {"Q8_0", 32,  34};
        case GgmlType::Q8_1: return {"Q8_1", 32,  36};

        case GgmlType::Q2_K: return {"Q2_K", 256,  82};
        case GgmlType::Q3_K: return {"Q3_K", 256, 110};
        case GgmlType::Q4_K: return {"Q4_K", 256, 144};
        case GgmlType::Q5_K: return {"Q5_K", 256, 176};
        case GgmlType::Q6_K: return {"Q6_K", 256, 210};
        case GgmlType::Q8_K: return {"Q8_K", 256, 292};

        case GgmlType::Unknown:
        default:
            return {"UNKNOWN", 0, 0};
    }
}

std::string_view valueTypeName(GgufValueType t) noexcept {
    switch (t) {
        case GgufValueType::UInt8:   return "uint8";
        case GgufValueType::Int8:    return "int8";
        case GgufValueType::UInt16:  return "uint16";
        case GgufValueType::Int16:   return "int16";
        case GgufValueType::UInt32:  return "uint32";
        case GgufValueType::Int32:   return "int32";
        case GgufValueType::Float32: return "float32";
        case GgufValueType::Bool:    return "bool";
        case GgufValueType::String:  return "string";
        case GgufValueType::Array:   return "array";
        case GgufValueType::UInt64:  return "uint64";
        case GgufValueType::Int64:   return "int64";
        case GgufValueType::Float64: return "float64";
    }
    return "unknown";
}

std::size_t valueElementWidth(GgufValueType t) noexcept {
    switch (t) {
        case GgufValueType::UInt8:
        case GgufValueType::Int8:
        case GgufValueType::Bool:
            return 1;
        case GgufValueType::UInt16:
        case GgufValueType::Int16:
            return 2;
        case GgufValueType::UInt32:
        case GgufValueType::Int32:
        case GgufValueType::Float32:
            return 4;
        case GgufValueType::UInt64:
        case GgufValueType::Int64:
        case GgufValueType::Float64:
            return 8;
        case GgufValueType::String:
        case GgufValueType::Array:
            return 0;
    }
    return 0;
}

std::size_t bytesForTensor(GgmlType t, std::uint64_t nelements) noexcept {
    const auto info = typeInfo(t);
    if (info.blockSize == 0 || info.typeSize == 0) {
        return 0;
    }
    if (nelements % info.blockSize != 0) {
        return 0;
    }
    return static_cast<std::size_t>(nelements / info.blockSize) *
           static_cast<std::size_t>(info.typeSize);
}

} // namespace mimirmind::model