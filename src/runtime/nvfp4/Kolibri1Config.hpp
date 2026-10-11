// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "model/LlmConfig.hpp"

#include <string_view>

namespace mimirmind::runtime::nvfp4 {

/**
 * Parse a Kolibri-1 (Aleph-Alpha `kolibri1`, `Kolibri1ForCausalLM`)
 * `config.json` into an LlmConfig, so the FP8/safetensors loader can build the
 * runtime model config the same way the Gemma-4 / Qwen3.5-MoE paths do.
 *
 * Kolibri-1 is a PURE-ATTENTION MoE (no GatedDeltaNet): 50 layers of GQA
 * (48 Q / 4 KV heads, head_dim 128) in a 4:1 sliding(513):full pattern
 * (`layer_types`), with per-head QK-RMSNorm and a 384-expert / top-6 sigmoid
 * router plus one always-on shared expert. Uniform head_dim ⇒ no SWA/full
 * head-dim split (unlike Gemma 4), so only `keyLength`/`valueLength` are set and
 * `keyLengthSwa` stays 0; `slidingWindowPattern` still carries the per-layer
 * sliding mask (headDim(block) returns the uniform value regardless).
 *
 * Two architectural diffs are NOT expressible in LlmConfig and are the
 * backend's responsibility (stubbed with TODOs at I-1):
 *   - full-attention layers are NoPE (RoPE applied only in sliding layers) —
 *     Kolibri1Backend must skip RoPE where slidingWindowPattern[l] == false
 *     (roadmap 5.34.5 / I-4);
 *   - the router is a SIGMOID top-6 gate (not softmax), top-6 weights are NOT
 *     renormalised (`norm_topk_prob` false), and a per-layer
 *     `moe.router.expert_bias` tensor is added to the logits for selection
 *     (DeepSeek-V3-style aux-loss-free balancing) — roadmap 5.34.6 / I-5.
 *
 * Sets `architecture = "kolibri1"`; the arch factory then selects
 * Kolibri1Backend.
 */
[[nodiscard]] model::LlmConfig
parseKolibri1SafetensorsConfig(std::string_view configJson);

/// True iff `configJson` is a Kolibri-1 checkpoint (model_type `kolibri1` or
/// `architectures[0]` contains "Kolibri"). Used by the loader to pick this
/// parser before the generic qwen35moe fallback. Pure JSON inspection — no
/// model / GPU dependencies.
[[nodiscard]] bool isKolibri1Config(const std::string& configText);

} // namespace mimirmind::runtime::nvfp4
