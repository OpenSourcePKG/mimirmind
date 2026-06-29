#pragma once

#include "model/GgufTypes.hpp"

#include <cstddef>
#include <string_view>

namespace mimirmind::compute {

/**
 * Per-quantisation-type behaviour as a polymorphic interface.
 *
 * One singleton subclass per GgmlType we support: F32, F16, BF16, Q4_K,
 * Q6_K, Q8_0, ... The instances are stateless — they just encapsulate the
 * block layout and the per-element dequant logic, and (optionally) name a
 * GPU matmul SPV module for the GpuMatmul dispatcher.
 *
 * Dispatch happens once, by GgmlType, via `compute::quantType(type)` from
 * QuantTypeRegistry.hpp. All call sites then go through the virtuals so
 * adding a new type is a 1-class addition (plus one switch case in the
 * registry).
 */
class QuantType {
public:
    virtual ~QuantType() = default;

    QuantType(const QuantType&)            = delete;
    QuantType& operator=(const QuantType&) = delete;
    QuantType(QuantType&&)                 = delete;
    QuantType& operator=(QuantType&&)      = delete;

    /// The GgmlType this implementation handles.
    [[nodiscard]] virtual model::GgmlType ggmlType() const noexcept = 0;

    /// Human-readable name (matches ggml's, e.g., "F32", "Q6_K").
    [[nodiscard]] virtual std::string_view name() const noexcept = 0;

    /// Elements per block. 1 for non-quantised float types.
    [[nodiscard]] virtual std::size_t blockElements() const noexcept = 0;

    /// Bytes per block in the on-disk / on-USM layout.
    [[nodiscard]] virtual std::size_t blockBytes() const noexcept = 0;

    /**
     * Convert `nelements` consecutive elements at `src` (in GGUF layout
     * for this type) into F32 at `dst`. The caller owns the buffers and
     * is responsible for `nelements` being a multiple of `blockElements()`
     * for block-quantised types.
     */
    virtual void dequantToF32(const void* src,
                              std::size_t nelements,
                              float*      dst) const = 0;

    /**
     * SPV module name for the GPU matmul kernel that consumes this
     * weight type, or empty if no GPU kernel exists. GpuMatmul uses this
     * at construction to load the corresponding SPV; at dispatch time an
     * empty name means CPU fallback.
     */
    [[nodiscard]] virtual std::string_view gpuMatmulModule() const noexcept {
        return {};
    }

protected:
    QuantType() = default;
};

} // namespace mimirmind::compute