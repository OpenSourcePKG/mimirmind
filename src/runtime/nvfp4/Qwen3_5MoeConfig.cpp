// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/nvfp4/Qwen3_5MoeConfig.hpp"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace mimirmind::runtime::nvfp4 {

namespace {
[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("qwen35moe config: " + msg);
}
} // namespace

model::LlmConfig parseQwen3_5MoeSafetensorsConfig(std::string_view configJson) {
    nlohmann::json top =
        nlohmann::json::parse(configJson.begin(), configJson.end(), nullptr,
                              /*allow_exceptions=*/false);
    if (top.is_discarded() || !top.is_object()) {
        fail("not a JSON object");
    }
    // Multimodal checkpoints nest the text params under `text_config`.
    const nlohmann::json& c =
        (top.contains("text_config") && top["text_config"].is_object())
            ? top["text_config"]
            : top;

    auto reqU = [&](const char* key) -> std::uint32_t {
        if (!c.contains(key) || !c[key].is_number_integer()) {
            fail(std::string("missing required integer '") + key + "'");
        }
        return c[key].get<std::uint32_t>();
    };
    auto optU = [&](const char* key, std::uint32_t def) -> std::uint32_t {
        return (c.contains(key) && c[key].is_number_integer())
                   ? c[key].get<std::uint32_t>() : def;
    };
    auto optF = [&](const char* key, float def) -> float {
        return (c.contains(key) && c[key].is_number())
                   ? c[key].get<float>() : def;
    };

    model::LlmConfig cfg;
    // Wire arch id kept as the stable legacy "qwen35moe" (== HF model_type
    // qwen3_5_moe). The C++ classes were renamed Qwen35Moe* -> Qwen3_5Moe* for
    // accuracy, but this load-bearing identifier stays unchanged so deployed
    // configs and the arch-match sites keep working. Renaming the wire string
    // is a separate, Spark-verified step.
    //
    // 5.27 qwen4_exp (Qwen3.8-Flash-Next): the SAME hybrid backbone (GDN +
    // full-attn + MoE + MTP, parsed identically below) plus Hyper-Connections
    // and PLE n-gram embeddings. model_type is top-level; route it to the
    // Qwen4ExpBackend via a distinct arch id so the two new components can hook
    // in without touching the qwen35moe path.
    const std::string modelType =
        (top.contains("model_type") && top["model_type"].is_string())
            ? top["model_type"].get<std::string>() : std::string{};
    const bool isQwen4Exp = (modelType == "qwen4_exp");
    cfg.architecture     = isQwen4Exp ? "qwen4_exp" : "qwen35moe";
    cfg.blockCount       = reqU("num_hidden_layers");
    cfg.contextLength    = optU("max_position_embeddings", 262144);
    cfg.embeddingLength  = reqU("hidden_size");
    cfg.feedForwardLength = optU("intermediate_size", 0); // pure-MoE ships none
    cfg.headCount        = reqU("num_attention_heads");
    cfg.headCountKv      = optU("num_key_value_heads", cfg.headCount);
    cfg.keyLength        = optU("head_dim", 0);
    cfg.valueLength      = cfg.keyLength;
    cfg.rmsNormEps       = optF("rms_norm_eps", 1e-6F);

    // RoPE: rope_parameters.{rope_theta, mrope_section}. rope_theta is 1e7 for
    // this model — defaulting to 1e4 would corrupt every position.
    if (c.contains("rope_parameters") && c["rope_parameters"].is_object()) {
        const auto& rp = c["rope_parameters"];
        if (rp.contains("rope_theta") && rp["rope_theta"].is_number()) {
            cfg.ropeFreqBase = rp["rope_theta"].get<float>();
        }
        if (rp.contains("mrope_section") && rp["mrope_section"].is_array()) {
            for (const auto& s : rp["mrope_section"]) {
                if (s.is_number_integer()) cfg.ropeSections.push_back(s.get<std::int32_t>());
            }
            cfg.ropeSections.push_back(0); // GGUF appends a trailing 0 section
        }
    } else {
        cfg.ropeFreqBase = optF("rope_theta", 10000.0F);
    }
    cfg.ropeFreqBaseSwa = cfg.ropeFreqBase;

    // YaRN / rope_scaling long-context extension (roadmap 8.8). HF ships it as
    // a `rope_scaling` dict (top-level; some checkpoints nest it under
    // rope_parameters). Absent ⇒ base RoPE ⇒ bit-identical to today. Consumed
    // DYNAMICALLY (only when a request exceeds original_max_position) — see
    // compute/YarnRope.hpp.
    const nlohmann::json* rsp = nullptr;
    if (c.contains("rope_scaling") && c["rope_scaling"].is_object()) {
        rsp = &c["rope_scaling"];
    } else if (c.contains("rope_parameters") &&
               c["rope_parameters"].is_object() &&
               (c["rope_parameters"].contains("rope_type") ||
                c["rope_parameters"].contains("factor"))) {
        rsp = &c["rope_parameters"];
    }
    if (rsp != nullptr) {
        const auto& rs = *rsp;
        std::string type;
        if (rs.contains("rope_type") && rs["rope_type"].is_string()) {
            type = rs["rope_type"].get<std::string>();
        } else if (rs.contains("type") && rs["type"].is_string()) {
            type = rs["type"].get<std::string>();
        }
        if (!type.empty() && type != "default" && type != "none") {
            cfg.ropeScalingType = type;
            if (rs.contains("factor") && rs["factor"].is_number()) {
                cfg.ropeScalingFactor = rs["factor"].get<float>();
            }
            cfg.ropeOrigMaxPos =
                (rs.contains("original_max_position_embeddings") &&
                 rs["original_max_position_embeddings"].is_number())
                    ? rs["original_max_position_embeddings"].get<std::uint32_t>()
                    : cfg.contextLength;
            if (rs.contains("beta_fast") && rs["beta_fast"].is_number()) {
                cfg.ropeBetaFast = rs["beta_fast"].get<float>();
            }
            if (rs.contains("beta_slow") && rs["beta_slow"].is_number()) {
                cfg.ropeBetaSlow = rs["beta_slow"].get<float>();
            }
        }
    }

    // MoE.
    cfg.expertCount                   = optU("num_experts", 0);
    cfg.expertUsedCount               = optU("num_experts_per_tok", 0);
    cfg.expertFeedForwardLength       = optU("moe_intermediate_size", 0);
    cfg.expertSharedFeedForwardLength = optU("shared_expert_intermediate_size", 0);

    // GatedDeltaNet SSM. inner_size = num_value_heads * value_head_dim.
    cfg.ssmConvKernel   = optU("linear_conv_kernel_dim", 0);
    cfg.ssmStateSize    = optU("linear_key_head_dim", 0);
    cfg.ssmTimeStepRank = optU("linear_num_value_heads", 0);
    cfg.ssmGroupCount   = optU("linear_num_key_heads", 0);
    cfg.ssmInnerSize    = optU("linear_num_value_heads", 0) * optU("linear_value_head_dim", 0);

    cfg.nextnPredictLayers = optU("mtp_num_hidden_layers", 0);

    // 5.27 qwen4_exp — Hyper-Connections + PLE n-gram embeddings. Parsed into
    // LlmConfig now (default-zero for every other arch); consumed by
    // Qwen4ExpBackend in I-3 (Hyper-Connections) / I-4 (PLE). Values live in the
    // text_config (`c`), same nesting as the backbone dims above.
    if (isQwen4Exp) {
        cfg.ngramSize         = optU("ngram_size", 0);
        cfg.ngramVocabBase    = optU("ngram_vocab_size_base", 0);
        cfg.headsPerNgram     = optU("heads_per_ngram", 0);
        cfg.pleEmbedDim       = optU("ple_embed_dim", 0);
        cfg.pleConvKernelSize = optU("ple_conv_kernel_size", 0);
        cfg.splitNgramParts   = optU("split_ngram_parts", 0);
        if (c.contains("ple_embedding_dtype") && c["ple_embedding_dtype"].is_string()) {
            cfg.pleEmbeddingDtype = c["ple_embedding_dtype"].get<std::string>();
        }
        if (c.contains("ple_layer_ids") && c["ple_layer_ids"].is_array()) {
            for (const auto& id : c["ple_layer_ids"]) {
                if (id.is_number_integer()) {
                    cfg.pleLayerIds.push_back(id.get<std::uint32_t>());
                }
            }
        }
    }

    // Per-layer recurrent mask: `layer_types` is authoritative; else
    // synthesise from full_attention_interval ((b+1)%interval != 0), matching
    // the GGUF/llama.cpp qwen35moe convention.
    const std::uint32_t interval = optU("full_attention_interval", 4);
    if (c.contains("layer_types") && c["layer_types"].is_array()) {
        for (const auto& lt : c["layer_types"]) {
            cfg.recurrentLayerPattern.push_back(lt.is_string()
                                                && lt.get<std::string>() == "linear_attention");
        }
    } else if (cfg.ssmConvKernel > 0 && interval > 0) {
        for (std::uint32_t b = 0; b < cfg.blockCount; ++b) {
            cfg.recurrentLayerPattern.push_back((b + 1) % interval != 0);
        }
    }

    return cfg;
}

} // namespace mimirmind::runtime::nvfp4