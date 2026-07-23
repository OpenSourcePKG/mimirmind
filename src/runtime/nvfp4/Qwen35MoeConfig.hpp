// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "model/LlmConfig.hpp"

#include <string_view>

namespace mimirmind::runtime::nvfp4 {

/**
 * Parse a HuggingFace `config.json` (the `qwen3_5_moe` NVFP4 checkpoint's)
 * into a `model::LlmConfig`, producing the SAME values `LlmConfig::parseFromGguf`
 * derives from the equivalent GGUF (verified field-by-field against the
 * running unsloth GGUF's metadata). This is the config half of the NVFP4
 * load path, replacing the GGUF-metadata parse.
 *
 * Field mapping highlights (config.json under `text_config` if present):
 *   blockCount        <- num_hidden_layers (40; GGUF's 41 counts the MTP block)
 *   ropeFreqBase      <- rope_parameters.rope_theta (1e7, NOT the 1e4 default)
 *   ropeSections      <- rope_parameters.mrope_section + a trailing 0
 *   ssmInnerSize      <- linear_num_value_heads * linear_value_head_dim
 *   ssm{State,TimeStepRank,GroupCount,ConvKernel} <- linear_* keys
 *   recurrentLayerPattern <- `layer_types` ("linear_attention" => recurrent),
 *                            else synthesised from full_attention_interval
 *   nextnPredictLayers <- mtp_num_hidden_layers
 *   expert*            <- num_experts / num_experts_per_tok / *_intermediate_size
 *
 * Throws std::runtime_error on malformed JSON or a missing required key
 * (num_hidden_layers, hidden_size, num_attention_heads).
 */
[[nodiscard]] model::LlmConfig
parseQwen35MoeSafetensorsConfig(std::string_view configJson);

} // namespace mimirmind::runtime::nvfp4