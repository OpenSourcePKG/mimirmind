// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "compute/QuantType.hpp"
#include "core/gguf/GgufTypes.hpp"

#include <span>

namespace mimirmind::compute {

/**
 * Look up the singleton QuantType implementation for `type`. Returns
 * nullptr for types that are not yet implemented (e.g., Q2_K, IQ4_NL).
 *
 * The returned pointer is to a static singleton — never owns, never null
 * for the types we explicitly support.
 */
[[nodiscard]] const QuantType* quantType(core::gguf::GgmlType type) noexcept;

/**
 * All currently-registered QuantType instances. Used by GpuMatmul at
 * construction time to discover which types have GPU kernels (via
 * `QuantType::gpuMatmulModule()`).
 */
[[nodiscard]] std::span<const QuantType* const> allQuantTypes() noexcept;

} // namespace mimirmind::compute