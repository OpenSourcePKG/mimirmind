// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/gguf/GgufTypes.hpp"

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
    [[nodiscard]] virtual core::gguf::GgmlType ggmlType() const noexcept = 0;

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
     * SPV module name for the GPU matvec kernel that consumes this
     * weight type, or empty if no GPU kernel exists. GpuMatmul uses this
     * at construction to load the corresponding SPV; at dispatch time an
     * empty name means CPU fallback.
     *
     * This is the M=1 (decode) hot path.
     */
    [[nodiscard]] virtual std::string_view gpuMatmulModule() const noexcept {
        return {};
    }

    /**
     * Optional SPV module name for the GPU GEMM kernel — the batched
     * (M > 1) variant used during prefill. When empty, GpuMatmul falls
     * back to launching the matvec kernel once per row of X.
     *
     * The kernel is expected to have the signature
     *   (const float* X, const uchar* W, float* Y, int K, int N, int M)
     * with launch geometry (ceil(N/kOutputsPerGroup)*kLocalSize,
     * ceil(M/gpuMatmulGemmMTile()), 1).
     */
    [[nodiscard]] virtual std::string_view gpuMatmulGemmModule() const noexcept {
        return {};
    }

    /**
     * The M-tile size baked into the GEMM kernel's per-workgroup output
     * geometry. Only meaningful when gpuMatmulGemmModule() is non-empty.
     */
    [[nodiscard]] virtual std::size_t gpuMatmulGemmMTile() const noexcept {
        return 1;
    }

protected:
    QuantType() = default;
};

} // namespace mimirmind::compute