// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/nvfp4/Gemma4Config.hpp"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace mimirmind::runtime::nvfp4 {

namespace {
[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("gemma4 config: " + msg);
}
} // namespace

model::LlmConfig parseGemma4SafetensorsConfig(std::string_view configJson) {
    nlohmann::json top =
        nlohmann::json::parse(configJson.begin(), configJson.end(), nullptr,
                              /*allow_exceptions=*/false);
    if (top.is_discarded() || !top.is_object()) {
        fail("not a JSON object");
    }
    // Multimodal Gemma-4 nests the text params under `text_config`.
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
    cfg.architecture      = "gemma4";
    cfg.blockCount        = reqU("num_hidden_layers");
    cfg.contextLength     = optU("max_position_embeddings", 262144);
    cfg.embeddingLength   = reqU("hidden_size");
    cfg.feedForwardLength = optU("intermediate_size", 0);
    cfg.headCount         = reqU("num_attention_heads");
    cfg.rmsNormEps        = optF("rms_norm_eps", 1e-6F);
    cfg.finalLogitSoftcap = optF("final_logit_softcapping", 0.0F);
    cfg.slidingWindow     = optU("sliding_window", 0);
    cfg.sharedKvLayers    = optU("num_kv_shared_layers", 0);

    // Per-layer head_dim split: SWA layers use `head_dim` (256), full-attention
    // layers use `global_head_dim` (512). Matches the GGUF key_length (full) /
    // key_length_swa (SWA). `keyLength` holds the FULL value (headDim()'s
    // default); `keyLengthSwa` the SWA one — LlmConfig::headDim(block) picks per
    // layer from slidingWindowPattern.
    const std::uint32_t headDimSwa  = optU("head_dim", 0);
    const std::uint32_t headDimFull = optU("global_head_dim", headDimSwa);
    cfg.keyLength      = headDimFull;
    cfg.valueLength    = headDimFull;
    cfg.keyLengthSwa   = headDimSwa;
    cfg.valueLengthSwa = headDimSwa;

    // RoPE: per rope_parameters — full_attention {rope_theta 1e6, proportional},
    // sliding_attention {rope_theta 1e4, default}. ropeFreqBase drives the
    // full-attention layers, ropeFreqBaseSwa the sliding ones (GemmaBaseBackend
    // selects per layer). Proportional RoPE (partial_rotary_factor) needs the
    // rope_freqs.weight tensor, which this checkpoint does not ship — the
    // full-attention layers fall back to plain RoPE at ropeFreqBase (staged;
    // rope_freqs to be added for long-context parity).
    float ropeFull = 1000000.0F;
    float ropeSwa  = 10000.0F;
    if (c.contains("rope_parameters") && c["rope_parameters"].is_object()) {
        const auto& rp = c["rope_parameters"];
        auto theta = [&](const char* which, float def) -> float {
            if (rp.contains(which) && rp[which].is_object()) {
                const auto& o = rp[which];
                if (o.contains("rope_theta") && o["rope_theta"].is_number()) {
                    return o["rope_theta"].get<float>();
                }
            }
            return def;
        };
        ropeFull = theta("full_attention", ropeFull);
        ropeSwa  = theta("sliding_attention", ropeSwa);
        // p-RoPE: the full-attention (global) layers rotate only
        // partial_rotary_factor * head_dim dims. rope_type "proportional".
        if (rp.contains("full_attention") && rp["full_attention"].is_object()) {
            const auto& fa = rp["full_attention"];
            if (fa.contains("partial_rotary_factor") &&
                fa["partial_rotary_factor"].is_number()) {
                cfg.ropePartialRotaryFull =
                    fa["partial_rotary_factor"].get<float>();
            }
        }
    }
    cfg.ropeFreqBase    = ropeFull;
    cfg.ropeFreqBaseSwa = ropeSwa;

    // Per-layer SWA mask + KV-head count from `layer_types`. SWA layers keep
    // `num_key_value_heads` (8), full-attention layers `num_global_key_value_
    // heads` (1). The scalar headCountKv is the dominant (SWA) value; the
    // backend uses headCountKvPerLayer authoritatively.
    const std::uint32_t kvSwa  = optU("num_key_value_heads", cfg.headCount);
    const std::uint32_t kvFull = optU("num_global_key_value_heads", kvSwa);
    cfg.headCountKv = kvSwa;
    if (c.contains("layer_types") && c["layer_types"].is_array()) {
        for (const auto& lt : c["layer_types"]) {
            const bool sliding =
                lt.is_string() && lt.get<std::string>() == "sliding_attention";
            cfg.slidingWindowPattern.push_back(sliding);
            cfg.headCountKvPerLayer.push_back(sliding ? kvSwa : kvFull);
        }
    }

    return cfg;
}

} // namespace mimirmind::runtime::nvfp4
