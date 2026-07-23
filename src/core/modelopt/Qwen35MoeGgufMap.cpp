// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "core/modelopt/Qwen35MoeGgufMap.hpp"

#include <array>

namespace mimirmind::core::modelopt {

namespace {

// ---- Full-attention layer (e.g. GGUF blk.3) -----------------------------
constexpr std::array<GgufTensorSource, 8> kFullAttn = {{
    {"attn_norm.weight",           "input_layernorm.weight",          WeightXform::Direct},
    {"attn_q.weight",              "self_attn.q_proj.weight",         WeightXform::Direct}, // full gated 8192, no split
    {"attn_k.weight",              "self_attn.k_proj.weight",         WeightXform::Direct},
    {"attn_v.weight",              "self_attn.v_proj.weight",         WeightXform::Direct},
    {"attn_output.weight",         "self_attn.o_proj.weight",         WeightXform::Direct},
    {"attn_q_norm.weight",         "self_attn.q_norm.weight",         WeightXform::Direct},
    {"attn_k_norm.weight",         "self_attn.k_norm.weight",         WeightXform::Direct},
    {"post_attention_norm.weight", "post_attention_layernorm.weight", WeightXform::Direct},
}};

// ---- DeltaNet (linear_attn) layer (e.g. GGUF blk.0) ----------------------
// unsloth reuses attn_qkv/attn_gate for in_proj_qkv/in_proj_z. conv1d
// [8192,1,4] is byte-identical to GGUF [4,8192].
// ssm_alpha<-in_proj_a / ssm_beta<-in_proj_b: both are [32,2048] so this is
// not shape-checkable, but CONFIRMED by value correlation between the GGUF's
// F32 ssm_alpha/ssm_beta and the HF bf16 in_proj_a/in_proj_b (blk.0):
// alpha matches in_proj_a (mse 1.95e-4, wrong pairing 5.3e-4), beta matches
// in_proj_b (mse 3.7e-5, wrong pairing 5.1e-4) — the natural a->alpha pairing.
constexpr std::array<GgufTensorSource, 11> kDeltaNet = {{
    {"attn_norm.weight",           "input_layernorm.weight",          WeightXform::Direct},
    {"attn_qkv.weight",            "linear_attn.in_proj_qkv.weight",  WeightXform::Direct},
    {"attn_gate.weight",           "linear_attn.in_proj_z.weight",    WeightXform::Direct},
    {"ssm_alpha.weight",           "linear_attn.in_proj_a.weight",    WeightXform::Direct},
    {"ssm_beta.weight",            "linear_attn.in_proj_b.weight",    WeightXform::Direct},
    {"ssm_conv1d.weight",          "linear_attn.conv1d.weight",       WeightXform::Direct},
    {"ssm_a",                      "linear_attn.A_log",               WeightXform::Direct},
    {"ssm_dt.bias",                "linear_attn.dt_bias",             WeightXform::Direct},
    {"ssm_norm.weight",            "linear_attn.norm.weight",         WeightXform::Direct},
    {"ssm_out.weight",             "linear_attn.out_proj.weight",     WeightXform::Direct},
    {"post_attention_norm.weight", "post_attention_layernorm.weight", WeightXform::Direct},
}};

// ---- MoE block (every layer) --------------------------------------------
constexpr std::array<GgufTensorSource, 8> kMoe = {{
    {"ffn_gate_inp.weight",       "mlp.gate.weight",                     WeightXform::Direct}, // router
    {"ffn_gate_exps.weight",      "mlp.experts.{E}.gate_proj.weight",    WeightXform::StackExperts},
    {"ffn_up_exps.weight",        "mlp.experts.{E}.up_proj.weight",      WeightXform::StackExperts},
    {"ffn_down_exps.weight",      "mlp.experts.{E}.down_proj.weight",    WeightXform::StackExperts},
    {"ffn_gate_shexp.weight",     "mlp.shared_expert.gate_proj.weight",  WeightXform::Direct},
    {"ffn_up_shexp.weight",       "mlp.shared_expert.up_proj.weight",    WeightXform::Direct},
    {"ffn_down_shexp.weight",     "mlp.shared_expert.down_proj.weight",  WeightXform::Direct},
    {"ffn_gate_inp_shexp.weight", "mlp.shared_expert_gate.weight",       WeightXform::Direct},
}};

// ---- Model-level (full names) -------------------------------------------
constexpr std::array<GgufTensorSource, 3> kTopLevel = {{
    {"token_embd.weight", "model.language_model.embed_tokens.weight", WeightXform::Direct},
    {"output_norm.weight","model.language_model.norm.weight",         WeightXform::Direct},
    {"output.weight",     "lm_head.weight",                           WeightXform::Direct},
}};

} // namespace

std::span<const GgufTensorSource> qwen35moeFullAttnTensors() noexcept { return kFullAttn; }
std::span<const GgufTensorSource> qwen35moeDeltaNetTensors() noexcept { return kDeltaNet; }
std::span<const GgufTensorSource> qwen35moeMoeTensors() noexcept { return kMoe; }
std::span<const GgufTensorSource> qwen35moeTopLevelTensors() noexcept { return kTopLevel; }

bool qwen35moeIsFullAttnLayer(int layer, int interval) noexcept {
    return interval > 0 && ((layer + 1) % interval == 0);
}

std::string qwen35moeHfName(std::string_view hfSuffix, int layer, int expert) {
    std::string s = "model.language_model.layers." + std::to_string(layer) + ".";
    s.append(hfSuffix);
    if (expert >= 0) {
        const std::string ph = "{E}";
        const auto pos = s.find(ph);
        if (pos != std::string::npos) {
            s.replace(pos, ph.size(), std::to_string(expert));
        }
    }
    return s;
}

} // namespace mimirmind::core::modelopt