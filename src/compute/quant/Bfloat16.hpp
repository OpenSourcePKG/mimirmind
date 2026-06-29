#pragma once

#include "compute/QuantType.hpp"

namespace mimirmind::compute::quant {

/// bfloat16 → float32 per element (high-16 of a binary32, shifted into place).
class Bfloat16 final : public QuantType {
public:
    [[nodiscard]] static const Bfloat16& instance() noexcept;

    [[nodiscard]] model::GgmlType  ggmlType()      const noexcept override;
    [[nodiscard]] std::string_view name()          const noexcept override;
    [[nodiscard]] std::size_t      blockElements() const noexcept override;
    [[nodiscard]] std::size_t      blockBytes()    const noexcept override;

    void dequantToF32(const void* src,
                      std::size_t nelements,
                      float*      dst) const override;

private:
    Bfloat16() = default;
};

} // namespace mimirmind::compute::quant