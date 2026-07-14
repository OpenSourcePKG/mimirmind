// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "compute/QuantType.hpp"

namespace mimirmind::compute::quant {

/// Trivial passthrough QuantType — `dequantToF32` is a memcpy.
class Float32 final : public QuantType {
public:
    [[nodiscard]] static const Float32& instance() noexcept;

    [[nodiscard]] core::gguf::GgmlType  ggmlType()      const noexcept override;
    [[nodiscard]] std::string_view name()          const noexcept override;
    [[nodiscard]] std::size_t      blockElements() const noexcept override;
    [[nodiscard]] std::size_t      blockBytes()    const noexcept override;

    void dequantToF32(const void* src,
                      std::size_t nelements,
                      float*      dst) const override;

private:
    Float32() = default;
};

} // namespace mimirmind::compute::quant