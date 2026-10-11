// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/nvfp4/Kolibri1Config.hpp"

#include "core/log/Log.hpp"

#include <nlohmann/json.hpp>

#include <cctype>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace mimirmind::runtime::nvfp4 {

namespace {
[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("kolibri1 config: " + msg);
}
} // namespace

bool isKolibri1Config(const std::string& configText) {
    nlohmann::json j =
        nlohmann::json::parse(configText, nullptr, /*allow_exceptions=*/false);
    if (!j.is_object()) {
        return false;
    }
    auto lower = [](std::string s) {
        for (char& ch : s) ch = static_cast<char>(std::tolower(
                              static_cast<unsigned char>(ch)));
        return s;
    };
    if (j.contains("model_type") && j["model_type"].is_string()
        && lower(j["model_type"].get<std::string>()) == "kolibri1") {
        return true;
    }
    if (j.contains("architectures") && j["architectures"].is_array()) {
        for (const auto& a : j["architectures"]) {
            if (a.is_string()
                && lower(a.get<std::string>()).find("kolibri") != std::string::npos) {
                return true;
            }
        }
    }
    return false;
}

model::LlmConfig parseKolibri1SafetensorsConfig(std::string_view configJson) {
    nlohmann::json top =
        nlohmann::json::parse(configJson.begin(), configJson.end(), nullptr,
                              /*allow_exceptions=*/false);
    if (top.is_discarded() || !top.is_object()) {
        fail("not a JSON object");
    }
    // Kolibri-1 ships a flat text-only config (no `text_config` nesting), but
    // honour the nested form defensively so a future multimodal wrap still
    // parses the text tower.
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
    auto optB = [&](const char* key, bool def) -> bool {
        return (c.contains(key) && c[key].is_boolean())
                   ? c[key].get<bool>() : def;
    };
    auto optS = [&](const char* key) -> std::string {
        return (c.contains(key) && c[key].is_string())
                   ? c[key].get<std::string>() : std::string{};
    };

    model::LlmConfig cfg;
    cfg.architecture      = "kolibri1";
    cfg.blockCount        = reqU("num_hidden_layers");
    cfg.contextLength     = optU("max_position_embeddings", 262144);
    cfg.embeddingLength   = reqU("hidden_size");
    cfg.feedForwardLength = 0;  // pure-MoE: no dense FFN; experts below
    cfg.headCount         = reqU("num_attention_heads");
    cfg.headCountKv       = optU("num_key_value_heads", cfg.headCount);
    cfg.rmsNormEps        = optF("rms_norm_eps", 1e-6F);

    // Uniform head_dim across all layers (no Gemma-style SWA/full split): set
    // keyLength/valueLength only and leave keyLengthSwa == 0, so
    // LlmConfig::headDim(block) returns the uniform value even though
    // slidingWindowPattern below is populated.
    const std::uint32_t headDim = optU("head_dim", 0);
    cfg.keyLength   = headDim;
    cfg.valueLength = headDim;

    cfg.slidingWindow = optU("sliding_window", 0);

    // RoPE theta 1e4 drives the SLIDING layers. The full-attention layers are
    // NoPE (no positional encoding) — not expressible here; Kolibri1Backend
    // must skip RoPE on layers where slidingWindowPattern[l] == false (I-4).
    // ropeFreqBase is set to the same value so a backend that (wrongly) rotated
    // a full layer would at least use the right theta, but the correct forward
    // is NoPE on full.
    const float ropeTheta = optF("rope_theta", 10000.0F);
    cfg.ropeFreqBase    = ropeTheta;
    cfg.ropeFreqBaseSwa = ropeTheta;

    // Per-layer sliding mask from `layer_types` ("sliding_attention" -> true,
    // "full_attention" -> false). KV-head count is uniform (4) across layers,
    // so headCountKvPerLayer stays empty (headCountKvFor falls back to
    // headCountKv). Kolibri's pattern is 4 sliding : 1 full (full at layers
    // 4,9,...,blockCount-1).
    if (c.contains("layer_types") && c["layer_types"].is_array()) {
        for (const auto& lt : c["layer_types"]) {
            const bool sliding =
                lt.is_string() && lt.get<std::string>() == "sliding_attention";
            cfg.slidingWindowPattern.push_back(sliding);
        }
    }

    // MoE. Router MATH (sigmoid gate + per-layer expert_bias + no top-K renorm)
    // is NOT captured here — LlmConfig has no router-type field. Kolibri1Backend
    // implements the sigmoid top-6 selection with bias and the always-on shared
    // expert in I-5; these fields only size the expert tensors.
    cfg.expertCount                   = optU("num_experts", 0);
    cfg.expertUsedCount               = optU("num_experts_per_tok", 0);
    cfg.expertFeedForwardLength       = optU("moe_intermediate_size", 0);
    cfg.expertSharedFeedForwardLength = optU("shared_expert_intermediate_size", 0);

    // --- Config hardening: fail/warn on knobs the engine assumes, instead of
    // silently mis-loading (same discipline as the qwen35moe parser).
    const std::string act = optS("hidden_act");
    if (!act.empty() && act != "silu") {
        fail("hidden_act '" + act + "' is not supported (SwiGLU/SiLU is assumed)");
    }
    if (optB("attention_bias", false)) {
        fail("attention_bias=true is not supported (no Q/K/V/O bias is loaded)");
    }
    // TODO(5.34.6 / I-5): sigmoid router + no-renorm + expert_bias. Warn loudly
    // so a checkpoint that flips these away from Kolibri's values is noticed.
    if (c.contains("norm_topk_prob") && c["norm_topk_prob"].is_boolean()
        && c["norm_topk_prob"].get<bool>()) {
        MM_LOG_WARN("model",
                    "kolibri1: config sets norm_topk_prob=true, but the router "
                    "follows Kolibri's no-renormalisation convention (TODO I-5)");
    }
    // TODO(5.34.5 / I-4): full-attention layers are NoPE. Recorded here as a
    // breadcrumb; the backend owns the per-layer RoPE skip.
    MM_LOG_INFO("model",
                "kolibri1 config: {} layers ({} experts/top-{}, moe_ff {}, "
                "shared_ff {}), SWA {} (RoPE only in sliding layers; full=NoPE)",
                cfg.blockCount, cfg.expertCount, cfg.expertUsedCount,
                cfg.expertFeedForwardLength, cfg.expertSharedFeedForwardLength,
                cfg.slidingWindow);

    return cfg;
}

} // namespace mimirmind::runtime::nvfp4
