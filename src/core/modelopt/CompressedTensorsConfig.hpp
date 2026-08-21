// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <string>
#include <vector>

namespace mimirmind::core::modelopt {

/**
 * The compressed-tensors (llm-compressor / vLLM) quantization config, parsed
 * from `config.json.quantization_config` — the alternative to the ModelOpt
 * `hf_quant_config.json` sidecar (HfQuantConfig). Format
 * `nvfp4-pack-quantized` / `mixed-precision`: `config_groups` maps regex tensor
 * `targets` to a `weights.num_bits` (4 = NVFP4 group-16, 8 = FP8-E4M3), plus a
 * top-level `ignore` list (BF16 / unquantised — e.g. the vision tower, MTP head,
 * GatedDeltaNet small projections).
 *
 * A tensor's scheme is resolved by: ignore-match -> Bf16; else the first
 * `config_group` (in key order, so a layer-specific FP8 override precedes the
 * broad NVFP4 rule) whose `targets` match -> that group's width; else Bf16.
 * `schemeForTensor` accepts an HF name with or without a trailing `.weight`
 * (matching is done on the module path).
 */
class CompressedTensorsConfig {
public:
    enum class Scheme { Bf16, Nvfp4, Fp8 };

    /// Parse from the full `config.json` text. `valid()` is false if the file
    /// has no compressed-tensors `quantization_config` (then the caller should
    /// fall back to the ModelOpt path).
    [[nodiscard]] static CompressedTensorsConfig parse(const std::string& configJson);

    [[nodiscard]] Scheme schemeForTensor(const std::string& hfWeightName) const;

    [[nodiscard]] bool valid() const noexcept { return _valid; }

private:
    struct Group {
        std::vector<std::string> targets;  ///< `re:...` regex or exact module path
        int                      numBits{0};
    };
    std::vector<Group>       _groups;
    std::vector<std::string> _ignore;
    bool                     _valid{false};
};

} // namespace mimirmind::core::modelopt
