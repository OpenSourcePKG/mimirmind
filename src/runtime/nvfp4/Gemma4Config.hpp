// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "model/LlmConfig.hpp"

#include <string_view>

namespace mimirmind::runtime::nvfp4 {

/**
 * Parse a Gemma-4 (dense text tower) `config.json` into an LlmConfig, so the
 * NVFP4/safetensors loader can build the same LlmConfig the GGUF path derives
 * via `LlmConfig::parseFromGguf` (verified field-for-field against the
 * gemma-4-12B GGUF: key_length 512 / key_length_swa 256, per-layer
 * head_count_kv 8/1, sliding_window_pattern, rope 1e6/1e4, softcap 30).
 *
 * The checkpoint may be a multimodal `Gemma4UnifiedForConditionalGeneration`
 * (vision + audio + text) — only the `text_config` sub-object is read; the
 * vision/audio towers are ignored. Sets `architecture = "gemma4"`; the arch
 * factory then selects Gemma4DenseBackend (dense — expertCount 0).
 */
[[nodiscard]] model::LlmConfig
parseGemma4SafetensorsConfig(std::string_view configJson);

} // namespace mimirmind::runtime::nvfp4
