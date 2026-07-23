// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/modelopt/HfQuantConfig.hpp"
#include "core/modelopt/ModelOptQuant.hpp"
#include "core/modelopt/ModelOptWeightLayout.hpp"

#include <cstdint>
#include <span>
#include <string_view>

namespace mimirmind::core::safetensors {
class SafetensorsModel;
}

namespace mimirmind::core::modelopt {

/**
 * One fully assembled ModelOpt weight: the logical layout plus zero-copy
 * byte views of the packed weight and its scale sidecars (into the owning
 * SafetensorsModel's mmaps). Which scale spans are populated depends on the
 * scheme:
 *   NVFP4 -> packedWeight (U8) + blockScale (F8_E4M3) + globalScale (F32)
 *   FP8   -> packedWeight (F8_E4M3) + weightScale (F32) + inputScale (F32)
 * The unused spans are empty.
 */
struct ModelOptWeight {
    ModelOptWeightLayout          layout;
    std::span<const std::uint8_t> packedWeight;
    std::span<const std::uint8_t> blockScale;   ///< NVFP4 per-group scale
    std::span<const std::uint8_t> globalScale;  ///< NVFP4 per-tensor F32 scalar
    std::span<const std::uint8_t> weightScale;  ///< FP8 per-tensor F32 scalar
    std::span<const std::uint8_t> inputScale;   ///< FP8 per-tensor F32 scalar
};

/**
 * Assembles ModelOpt weights from a checkpoint by joining the three layers:
 * `SafetensorsModel` (the tensors), `HfQuantConfig` (the per-module scheme),
 * and the scheme descriptors (`ModelOptQuant`). Holds references to both —
 * they must outlive the assembler.
 */
class ModelOptWeightAssembler {
public:
    ModelOptWeightAssembler(const safetensors::SafetensorsModel& model,
                            const HfQuantConfig&                 config) noexcept
        : _model(model), _config(config) {}

    /// True if `module` (a module base, e.g. "...mlp.experts.5.gate_proj",
    /// NOT a leaf tensor name) resolves to a quantised ModelOpt weight.
    [[nodiscard]] bool isQuantized(std::string_view module) const;

    /// Assemble the ModelOpt weight rooted at `module`. Throws if the module
    /// is not quantised, a required `.weight` / scale sidecar is missing, or
    /// any tensor is inconsistent with the scheme descriptor.
    [[nodiscard]] ModelOptWeight assemble(std::string_view module) const;

private:
    const safetensors::SafetensorsModel& _model;
    const HfQuantConfig&                  _config;
};

} // namespace mimirmind::core::modelopt