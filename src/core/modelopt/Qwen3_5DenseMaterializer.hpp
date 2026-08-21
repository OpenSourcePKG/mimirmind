// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/modelopt/CompressedTensorsConfig.hpp"
#include "core/modelopt/Qwen3_5MoeMaterializer.hpp"

#include <vector>

namespace mimirmind::core::safetensors {
class SafetensorsModel;
}

namespace mimirmind::core::modelopt {

/**
 * Build the materialization plan for the DENSE qwen3_5 variant (Qwen3.8-27B),
 * a compressed-tensors (llm-compressor / vLLM) checkpoint with per-tensor
 * mixed precision: NVFP4 (group-16), per-channel FP8-E4M3, or unquantised
 * BF16, resolved per module by `CompressedTensorsConfig`.
 *
 * Shares the qwen3_5 arch walk with `planQwen3_5MoeMaterialization` (the same
 * GatedDeltaNet + periodic full-attention + norm/embed name tables), differing
 * only in the FFN — a dense SwiGLU `mlp.{gate,up,down}_proj` triple instead of
 * the router + expert banks — and in the quant format: compressed-tensors
 * names the NVFP4 sidecars explicitly (`.weight_packed` / `.weight_scale` /
 * `.weight_global_scale`, reciprocal global) and stores FP8 `weight_scale`
 * PER OUTPUT CHANNEL (`[out, 1]` BF16), not the ModelOpt per-tensor F32 scalar.
 * There are no experts, shared expert, or MTP head in the dense variant; the
 * vision tower / MTP / small GatedDeltaNet projections stay BF16 (config
 * `ignore`) and are widened to F32 like the norms.
 *
 * `arch.numExperts` and `arch.mtpLayers` are ignored (expected 0). Pure — reads
 * only tensor shapes from `model` and schemes from `cfg`.
 *
 * Throws std::runtime_error if a required HF tensor is missing, naming it.
 */
[[nodiscard]] std::vector<MaterializationStep>
planQwen3_5DenseMaterialization(const safetensors::SafetensorsModel& model,
                                const CompressedTensorsConfig&       cfg,
                                const Qwen3_5MoeArch&                arch);

} // namespace mimirmind::core::modelopt
