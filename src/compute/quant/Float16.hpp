#pragma once

#include "compute/QuantType.hpp"

namespace mimirmind::compute::quant {

/// IEEE-754 binary16 → binary32 per element.
class Float16 final : public QuantType {
public:
    [[nodiscard]] static const Float16& instance() noexcept;

    [[nodiscard]] model::GgmlType  ggmlType()      const noexcept override;
    [[nodiscard]] std::string_view name()          const noexcept override;
    [[nodiscard]] std::size_t      blockElements() const noexcept override;
    [[nodiscard]] std::size_t      blockBytes()    const noexcept override;

    void dequantToF32(const void* src,
                      std::size_t nelements,
                      float*      dst) const override;

private:
    Float16() = default;
};

} // namespace mimirmind::compute::quant