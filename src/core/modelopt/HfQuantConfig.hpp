// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/modelopt/ModelOptQuant.hpp"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace mimirmind::core::modelopt {

/// The quantisation of one module, as declared in `hf_quant_config.json`.
struct ModuleQuant {
    ModelOptQuantScheme scheme;
    std::uint16_t       groupSize{0}; ///< 0 when the entry omits `group_size`
};

/**
 * Parsed NVIDIA ModelOpt `hf_quant_config.json`.
 *
 * The interesting field is `quantization.quantized_layers`: a map from a
 * MODULE prefix (not a leaf tensor name) to its scheme, e.g.
 *   "model.language_model.layers.0.mlp.experts" -> W4A16_NVFP4 (group_size 16)
 *   "model.language_model.layers.0.self_attn.q_proj" -> FP8
 * A weight and all its scale sidecars (`.weight`, `.weight_scale`,
 * `.weight_scale_2`, `.input_scale`) share the module's scheme, so
 * resolution is a longest boundary-prefix match of the tensor name against
 * the module keys — see `resolve`.
 *
 * `quant_algo` at the top level is a marker (`MIXED_PRECISION` for a mixed
 * checkpoint). For a UNIFORM checkpoint ModelOpt may instead put a real
 * scheme there and omit `quantized_layers`; `resolve` handles that by
 * falling back to the top-level scheme for every non-excluded module.
 *
 * `exclude_modules` lists glob patterns (e.g. `mtp*`) kept in original
 * precision; a name matching one resolves to "not quantised" even under a
 * uniform top-level scheme.
 */
class HfQuantConfig {
public:
    /// Parse the JSON text. Throws std::runtime_error on malformed JSON, a
    /// missing/non-object `quantization`, a `quantized_layers` entry without
    /// a string `quant_algo` or naming an unsupported algo, or a
    /// `group_size` that is present but not an unsigned integer.
    [[nodiscard]] static HfQuantConfig parse(std::string_view jsonText);

    /// Resolve a tensor or module name to its quantisation, or std::nullopt
    /// when the module is not quantised (kept in its original dtype, e.g.
    /// the router gate, norms, MTP, vision tower).
    [[nodiscard]] std::optional<ModuleQuant> resolve(std::string_view name) const;

    /// Convenience: just the scheme (drops the group size). nullopt as above.
    [[nodiscard]] std::optional<ModelOptQuantScheme>
    schemeForTensor(std::string_view name) const;

    /// True if `name` matches one of the `exclude_modules` globs.
    [[nodiscard]] bool isExcluded(std::string_view name) const noexcept;

    [[nodiscard]] std::string_view topLevelAlgo() const noexcept { return _topLevelAlgo; }
    [[nodiscard]] std::string_view kvCacheAlgo()  const noexcept { return _kvCacheAlgo; }
    [[nodiscard]] bool             isMixed()      const noexcept {
        return _topLevelAlgo == "MIXED_PRECISION";
    }

    [[nodiscard]] const std::map<std::string, ModuleQuant>& quantizedModules() const noexcept {
        return _modules;
    }
    [[nodiscard]] const std::vector<std::string>& excludeModules() const noexcept {
        return _exclude;
    }

private:
    std::map<std::string, ModuleQuant> _modules;      ///< module prefix -> quant
    std::vector<std::string>           _exclude;       ///< glob patterns
    std::string                        _topLevelAlgo;  ///< e.g. "MIXED_PRECISION"
    std::string                        _kvCacheAlgo;   ///< e.g. "FP8", or empty
    ModuleQuant                        _uniform{};      ///< top-level scheme, if uniform
    bool                               _hasUniform{false};
};

} // namespace mimirmind::core::modelopt