#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace mimirmind::core::gguf {

/// GGUF v3 metadata value type tag (see ggml's gguf.md).
enum class GgufValueType : std::uint32_t {
    UInt8   = 0,
    Int8    = 1,
    UInt16  = 2,
    Int16   = 3,
    UInt32  = 4,
    Int32   = 5,
    Float32 = 6,
    Bool    = 7,
    String  = 8,
    Array   = 9,
    UInt64  = 10,
    Int64   = 11,
    Float64 = 12,
};

/// Subset of ggml_type that we recognise. Numeric values match
/// llama.cpp / ggml. Unrecognised types are reported as Unknown and
/// stop the load with an error.
enum class GgmlType : std::uint32_t {
    F32     = 0,
    F16     = 1,
    Q4_0    = 2,
    Q4_1    = 3,
    Q5_0    = 6,
    Q5_1    = 7,
    Q8_0    = 8,
    Q8_1    = 9,
    Q2_K    = 10,
    Q3_K    = 11,
    Q4_K    = 12,
    Q5_K    = 13,
    Q6_K    = 14,
    Q8_K    = 15,
    I8      = 24,
    I16     = 25,
    I32     = 26,
    I64     = 27,
    F64     = 28,
    BF16    = 30,
    Unknown = 0xFFFFFFFFu,
};

struct GgmlTypeInfo {
    std::string_view name;
    std::uint32_t    blockSize;   // elements per block (1 for non-quantised)
    std::uint32_t    typeSize;    // bytes per block
};

[[nodiscard]] GgmlTypeInfo     typeInfo(GgmlType t) noexcept;
[[nodiscard]] std::string_view valueTypeName(GgufValueType t) noexcept;

/// 0 means "unknown element width" — used for nested arrays / strings.
[[nodiscard]] std::size_t      valueElementWidth(GgufValueType t) noexcept;

/// Returns the tensor's byte size, or 0 if the type is Unknown or
/// nelements is not a multiple of the type's block size.
[[nodiscard]] std::size_t      bytesForTensor(GgmlType t,
                                              std::uint64_t nelements) noexcept;

} // namespace mimirmind::core::gguf