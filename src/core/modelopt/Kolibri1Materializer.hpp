// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/modelopt/Qwen3_5MoeMaterializer.hpp" // MaterializationStep + Source

#include <cstdint>
#include <vector>

namespace mimirmind::core::safetensors {
class SafetensorsModel;
}

namespace mimirmind::core::modelopt {

/// Architecture parameters to walk the Kolibri-1 (Aleph-Alpha `kolibri1`)
/// pure-attention MoE stack. POD, so the planner stays free of
/// model::LlmConfig; the loader fills it from the parsed config.
struct Kolibri1Arch {
    int numLayers  = 0;  ///< num_hidden_layers (50)
    int numExperts = 0;  ///< num_experts routed (384); shared expert is separate
};

/**
 * Build the materialization plan for a Kolibri-1 FP8 128x128-block checkpoint
 * (DeepSeek-V3-style `fp8` quant: `weight_block_size [128,128]`, one F32
 * `<base>.weight_scale_inv` grid per quantised 2-D weight). For every GGUF
 * tensor the (not-yet-written) Kolibri1Backend needs, map the HF source(s),
 * the dequant kind, the expert-stacking offsets and the GGUF ne-order dims.
 *
 * The quant scheme is NOT read from an hf_quant_config.json / ModelOpt
 * descriptor (Kolibri keeps the scheme inside config.json and ships no
 * per-tensor sidecar beyond `weight_scale_inv`). Classification is therefore
 * by CHECKPOINT TRUTH: a 2-D `.weight` that has a sibling `.weight_scale_inv`
 * is Fp8Block128; everything else (norms, q/k-norm, router `mlp.gate`,
 * `moe.router.expert_bias`, embeddings, `lm_head`) is an unquantised BF16
 * passthrough widened to F32 — the convention the runtime reads as
 * `const float*`. `lm_head` is computed in F32 (bf16 in the checkpoint).
 *
 * The routed experts (`mlp.experts.{E}.{gate,up,down}_proj`) are STACKED into
 * one GGUF tensor per projection per layer (`ffn_{gate,up,down}_exps.weight`,
 * ne-order `[in, out, n_expert]`). They are the only Fp8Block128 tensors the
 * loader DEFERS (empty placeholder from executeMaterialization) and fills via
 * the streaming requant bank builder, so the full BF16 expert image never
 * forms. The always-on shared expert and the attention projections are also
 * Fp8Block128 but small, so they take the ordinary dequant->BF16 path.
 *
 * RMSNorm weights are stored AS-IS (no PostTransform). Confirmed against the
 * reference oracle (aleph-alpha-inference vLLM plugin): Kolibri uses vLLM's
 * PLAIN RMSNorm (`weight * x_normed`), NOT the Gemma/Qwen (1 + weight)
 * convention — so mimirmind's `weight * x_normed` norm kernel matches without
 * an AddOne. (An earlier I-3 draft applied AddOne; removed once the oracle
 * pinned the convention — see decisions/2026-10-12-kolibri1-i4-i5-forward-spec.)
 *
 * GGUF names are provisional and finalised jointly with I-4 (the forward);
 * they mirror the HF layout so the four per-layer norms stay unambiguous:
 *   input_layernorm           -> attn_norm
 *   self_attn.q/k/v/o_proj     -> attn_q / attn_k / attn_v / attn_output
 *   self_attn.q_norm/k_norm    -> attn_q_norm / attn_k_norm
 *   post_attn_norm             -> post_attn_norm
 *   post_attention_layernorm   -> post_attention_norm
 *   post_ffn_norm              -> post_ffn_norm
 *   mlp.gate                   -> ffn_gate_inp            (router)
 *   moe.router.expert_bias     -> exp_probs_b             (sigmoid-gate bias)
 *   mlp.experts.{E}.{g,u,d}     -> ffn_{gate,up,down}_exps (stacked)
 *   mlp.shared_experts.{g,u,d}  -> ffn_{gate,up,down}_shexp
 *   model.embed_tokens         -> token_embd
 *   model.norm                 -> output_norm
 *   lm_head                    -> output
 *
 * Pure: reads only tensor shapes/dtypes from `model`, launches no kernels.
 * Throws std::runtime_error (naming the tensor) if a required HF tensor is
 * missing.
 */
[[nodiscard]] std::vector<MaterializationStep>
planKolibri1Materialization(const safetensors::SafetensorsModel& model,
                            const Kolibri1Arch&                  arch);

} // namespace mimirmind::core::modelopt
