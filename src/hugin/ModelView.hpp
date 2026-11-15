// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/gguf/GgufReader.hpp"
#include "core/gguf/WeightsMap.hpp"

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace mimirmind::hugin {

/// Summary statistics of (a row range of) one tensor, computed in F64.
struct TensorStats {
    std::uint64_t elements{0};
    double        mean{0}, stddev{0}, min{0}, max{0}, absMax{0}, l2{0};
    std::uint64_t nanCount{0}, infCount{0};
    /// Per-row L2 norms (2-D tensors): min / mean / max over the rows.
    double        rowNormMin{0}, rowNormMean{0}, rowNormMax{0};
    std::uint64_t rows{0}, cols{0};
};

/**
 * Read-only host view of a GGUF model for Hugin's inspection tools. Opens the
 * file (mmap, no device upload), exposes tensor metadata, raw bytes,
 * dequantised rows and the base fingerprint Runes are bound to — the exact
 * value the engine's Futhark computes for a standalone GGUF load.
 *
 * Shape convention (GGUF): dims[0] is the innermost (column) dimension;
 * rows = nelements / dims[0].
 */
class ModelView {
public:
    explicit ModelView(const std::string& path);

    [[nodiscard]] const std::string& path() const noexcept { return _path; }
    [[nodiscard]] const std::string& fingerprint() const noexcept { return _fingerprint; }
    [[nodiscard]] const core::gguf::GgufReader& reader() const noexcept { return _reader; }

    /// Architecture string from `general.architecture`, or empty.
    [[nodiscard]] std::string architecture() const;

    [[nodiscard]] const core::gguf::GgufTensor* find(std::string_view name) const noexcept;
    [[nodiscard]] const core::gguf::GgufTensor& require(std::string_view name) const;
    [[nodiscard]] std::span<const std::uint8_t> bytes(const core::gguf::GgufTensor& t) const;

    /// Dequantise rows [rowBegin, rowEnd) to F32. Throws on unsupported type
    /// or an out-of-range / block-misaligned request.
    [[nodiscard]] std::vector<float> rowsF32(const core::gguf::GgufTensor& t,
                                             std::uint64_t rowBegin, std::uint64_t rowEnd) const;

    /// Stats over rows [rowBegin, rowEnd) (default: whole tensor), streamed in
    /// row chunks so large tensors do not need a full F32 copy.
    [[nodiscard]] TensorStats stats(const core::gguf::GgufTensor& t,
                                    std::optional<std::uint64_t> rowBegin = std::nullopt,
                                    std::optional<std::uint64_t> rowEnd   = std::nullopt) const;

    /// WeightsMap whose tensors point at the read-only file bytes — for a
    /// Futhark dry run (readback only; uploads must be no-ops).
    [[nodiscard]] core::gguf::WeightsMap readOnlyWeights() const;

    [[nodiscard]] static std::uint64_t colsOf(const core::gguf::GgufTensor& t) noexcept;
    [[nodiscard]] static std::uint64_t rowsOf(const core::gguf::GgufTensor& t) noexcept;

private:
    std::string            _path;
    core::gguf::GgufReader _reader;
    std::string            _fingerprint;
};

} // namespace mimirmind::hugin
