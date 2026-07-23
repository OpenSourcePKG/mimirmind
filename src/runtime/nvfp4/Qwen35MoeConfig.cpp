// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/nvfp4/Qwen35MoeConfig.hpp"

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

model::LlmConfig parseQwen35MoeSafetensorsConfig(std::string_view configJson) {
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
    cfg.architecture     = "qwen35moe";
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