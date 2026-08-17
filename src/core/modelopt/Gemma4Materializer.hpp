// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/modelopt/Qwen35MoeMaterializer.hpp" // MaterializationStep + Source

#include <cstdint>
#include <vector>

namespace mimirmind::core::safetensors {
class SafetensorsModel;
}

namespace mimirmind::core::modelopt {

/// Architecture parameters to walk the Gemma-4 dense text tower. Per-layer
/// `isFullAttn` (the full-attention layers use head_dim 512 / 1 KV head / V=K,
/// so they carry NO v_proj) is derived from the config's layer_types by the
/// loader and handed in here (POD, so this stays free of model::LlmConfig).
struct Gemma4Arch {
    int               numLayers = 0;
    std::vector<bool> isFullAttn; ///< size == numLayers; true where NOT sliding
};

/**
 * Build the materialization plan for a compressed-tensors ("nvfp4-pack-
 * quantized") Gemma-4 dense text checkpoint: every GGUF tensor the
 * Gemma4DenseBackend needs, mapped from its HF source(s), with the
 * compressed-tensors NVFP4 name-triple (`.weight_packed` / `.weight_scale` /
 * `.weight_global_scale`, global stored reciprocal) recorded on each source.
 *
 * Dense only — no MoE / GatedDeltaNet / MTP. The full-attention layers omit
 * attn_v (attention_k_eq_v, V = raw K). Every `*.norm.weight` gets
 * PostTransform::AddOne (Gemma bakes the (1 + w) RMSNorm convention);
 * layer_output_scale (from `layer_scalar`) does not. lm_head is in the
 * checkpoint's `ignore` list (BF16, not quantised) → kept BF16.
 *
 * Pure: reads only tensor shapes from `model`, launches no kernels. Throws
 * (naming the tensor) if a required HF tensor is missing.
 */
[[nodiscard]] std::vector<MaterializationStep>
planGemma4Materialization(const safetensors::SafetensorsModel& model,
                          const Gemma4Arch&                    arch);

} // namespace mimirmind::core::modelopt
