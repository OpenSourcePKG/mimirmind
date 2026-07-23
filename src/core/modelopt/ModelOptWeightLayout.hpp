// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/modelopt/ModelOptQuant.hpp"
#include "core/safetensors/SafetensorsHeader.hpp"

#include <cstdint>

namespace mimirmind::core::modelopt {

/// Logical dimensions of an assembled ModelOpt weight, derived from the
/// packed weight tensor's shape and the scheme's packing.
struct ModelOptWeightLayout {
    ModelOptQuantScheme scheme;
    std::uint64_t       outFeatures{0}; ///< rows N  (weight shape[0])
    std::uint64_t       inFeatures{0};  ///< logical K, unpacked (packed cols x packFactor)
    std::uint16_t       groupSize{0};   ///< block-scale group (16 for NVFP4, 0 for FP8)
};

/**
 * Pure validation of a weight's tensors against its scheme descriptor.
 * Given the already-looked-up `weight` tensor and the scheme's expected
 * scale sidecars (pass nullptr for ones the scheme does not use), verify
 * dtypes, dimensionality, and that the scale shapes agree with the packed
 * weight's implied in-/out-features. Returns the logical layout.
 *
 * Free and mmap-free (operates on `SafetensorsTensor` metadata only) so it
 * is unit-testable without a real checkpoint. Throws std::runtime_error on
 * any inconsistency, naming the offending tensor.
 *
 * `groupSize` is the value from `hf_quant_config` (0 if unspecified); it is
 * cross-checked against the scheme descriptor for block-scaled schemes.
 */
[[nodiscard]] ModelOptWeightLayout validateWeightLayout(
    ModelOptQuantScheme                   scheme,
    std::uint16_t                         groupSize,
    const safetensors::SafetensorsTensor& weight,
    const safetensors::SafetensorsTensor* blockScale,
    const safetensors::SafetensorsTensor* globalScale,
    const safetensors::SafetensorsTensor* weightScale,
    const safetensors::SafetensorsTensor* inputScale);

} // namespace mimirmind::core::modelopt