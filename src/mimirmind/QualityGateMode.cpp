// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "mimirmind/QualityGateMode.hpp"

#include "mimirmind/CliArgs.hpp"
#include "mimirmind/CliParser.hpp"

#include "core/backend/BackendRegistry.hpp"
#include "core/config/Config.hpp"
#include "core/log/Log.hpp"
#include "model/ChatTemplate.hpp"
#include "runtime/InferenceEngine.hpp"
#include "runtime/nvfp4/ModelFormatResolver.hpp"

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace mimirmind::cli {

namespace {

namespace mm = ::mimirmind;

// Curated DE goldset — diverse, prefill-exercising prompts with deterministic
// answers, so a greedy-token divergence is a real quality signal (not sampling
// noise). Kept small so the A/B stays bounded even on the 35B serving model:
// each prompt runs one baseline + one generate per case.
const std::vector<std::string>& deGoldset() {
    static const std::vector<std::string> g = {
        "Nenne die Hauptstadt von Japan. Antworte mit einem einzigen Wort.",
        "Was ist 17 multipliziert mit 23? Antworte nur mit der Zahl.",
        "Liste die ersten fuenf Primzahlen auf.",
        "Erklaere in genau zwei Saetzen, was Photosynthese ist.",
        "Uebersetze ins Englische: 'Der schnelle braune Fuchs springt "
        "ueber den faulen Hund.'",
        "Fasse in einem Satz zusammen: Die Sonne ist ein Stern im Zentrum "
        "unseres Sonnensystems, besteht vor allem aus Wasserstoff und Helium "
        "und erzeugt Energie durch Kernfusion.",
    };
    return g;
}

// Which lossy prefill flag(s) a case turns ON (baseline turns both OFF).
struct FlagCase {
    std::string name;
    bool        f32TcPrefill;
    bool        cublasFp8Prefill;
};

std::size_t commonPrefix(const std::vector<std::int32_t>& a,
                         const std::vector<std::int32_t>& b) {
    const std::size_t n = std::min(a.size(), b.size());
    std::size_t i = 0;
    while (i < n && a[i] == b[i]) ++i;
    return i;
}

} // namespace

int runQualityGate(const CliArgs& args, const mm::core::config::Config& cfg) {
    std::cout << kBanner;
    std::cout << "5.19 Increment C — lossy-tier quality gate (greedy A/B)\n";
    std::cout.flush();
    MM_LOG_INFO("main", "quality-gate starting (lossy prefill-flag A/B)");

    mm::runtime::InferenceEngine engine{cfg};

    if (args.modelPath.empty()) {
        std::cerr << "quality-gate: no model — set models[<defaultModel>].path "
                     "in the config or pass --model PATH\n";
        return 1;
    }

    // Resolve format / tokenizer sidecar from the default model entry, exactly
    // like SmokeMode / ServeMode (a bare NVFP4 directory has no extension).
    mm::core::config::ModelFormat fmt = mm::core::config::ModelFormat::Gguf;
    std::string tokenizerGguf;
    if (!cfg.models.empty()) {
        const auto& dm = cfg.defaultModelEntry();
        if (dm.path == args.modelPath) {
            fmt           = dm.format;
            tokenizerGguf = dm.tokenizerGguf;
        }
    }
    try {
        if (mm::runtime::nvfp4::resolveModelFormat(fmt, args.modelPath)
            == mm::core::config::ModelFormat::Nvfp4) {
            engine.loadModelNvfp4(args.modelPath, tokenizerGguf);
        } else {
            engine.loadModel(args.modelPath);
        }
    } catch (const std::exception& e) {
        std::cerr << "quality-gate: model load failed: " << e.what() << "\n";
        return 1;
    }

    // Mirror ServeMode / SmokeMode: kv / context overrides land AFTER load.
    {
        const auto& rt = cfg.runtime;
        if (rt.maxContextTokens.has_value() && *rt.maxContextTokens > 0) {
            engine.setMaxContextTokens(*rt.maxContextTokens);
        }
        if (rt.kvDtype.has_value()) {
            const std::string_view v{*rt.kvDtype};
            if      (v == "fp16") engine.setKvDtype(mm::runtime::KvDtype::FP16);
            else if (v == "q8_0") engine.setKvDtype(mm::runtime::KvDtype::Q8_0);
            else                  engine.setKvDtype(mm::runtime::KvDtype::F32);
        }
    }

    const bool isCuda = engine.computeContextKind() ==
                        mm::core::backend::BackendKind::Cuda;
    if (!isCuda) {
        std::cout << "\n[gate] backend is "
                  << mm::core::backend::BackendRegistry::name(
                         engine.computeContextKind())
                  << " — the lossy prefill flags are CUDA-only, their setters "
                     "are no-ops here, so there is nothing to gate.\n";
        MM_LOG_INFO("main", "quality-gate: non-CUDA backend — nothing to gate");
        return 0;
    }

    // Build the chat-templated prompt ids once per goldset entry.
    const auto& tok = engine.tokenizer();
    mm::model::ChatTemplate::Style style;
    try {
        style = mm::model::ChatTemplate::detectFromArch(
            engine.config().architecture);
    } catch (const std::exception& e) {
        std::cerr << "quality-gate: chat-template detect failed: " << e.what()
                  << "\n";
        return 1;
    }
    const auto chatStops = mm::model::ChatTemplate::stopIds(style, tok);

    struct Item {
        std::string               text;
        std::vector<std::int32_t> ids;
    };
    // Thinking OFF: for a "thinking" model (Qwen3.x) the reasoning trace
    // cascades — a one-token prefill perturbation reshuffles the whole scratch
    // pad even when the final answer is identical, which makes a full-trace
    // token compare meaningless. enable_thinking=false makes the model answer
    // directly, so the A/B measures exactly what matters: does the lossy flag
    // change the ANSWER. (The prefill perturbation itself is unchanged by the
    // thinking toggle.) Honoured by the Qwen family; a no-op elsewhere.
    std::vector<Item> items;
    for (const auto& prompt : deGoldset()) {
        const mm::model::ChatMessage msg{mm::model::ChatRole::User, prompt};
        std::vector<mm::model::ChatMessage> msgs{msg};
        items.push_back(
            {prompt, mm::model::ChatTemplate::encode(
                         style, tok, msgs, /*addGenerationPrompt=*/true,
                         /*tools=*/{}, /*enableThinking=*/std::optional<bool>{false})});
    }

    const std::size_t maxNew = args.maxNew > 0 ? args.maxNew : 32;

    auto& mmops = engine.gpuMatmul();

    // One greedy generate under an explicit flag config; a fresh KV cache so
    // the flag actually re-runs prefill (no prefix-cache skip).
    auto genUnder = [&](bool f32Tc, bool fp8,
                        std::span<const std::int32_t> ids)
        -> std::vector<std::int32_t> {
        mmops.setF32TcPrefill(f32Tc);
        mmops.setCublasFp8Prefill(fp8);
        engine.resetCache();
        mm::runtime::GenerateParams gp{};
        gp.maxNewTokens         = maxNew;
        gp.sampling.temperature = 0.0F;   // greedy argmax — deterministic
        gp.stopIds              = chatStops;
        return engine.generate(ids, gp);
    };

    // Determinism sanity: the exact-path baseline must reproduce bit-for-bit,
    // else the whole A/B is meaningless.
    {
        const auto a = genUnder(false, false, items.front().ids);
        const auto b = genUnder(false, false, items.front().ids);
        if (a != b) {
            std::cerr << "quality-gate: baseline is NON-DETERMINISTIC "
                         "(greedy generate differs run-to-run) — cannot gate. "
                         "Fix determinism first.\n";
            return 1;
        }
        std::cout << "\n[gate] determinism check passed (exact-path baseline "
                     "reproduces). maxNew=" << maxNew << ", goldset="
                  << items.size() << " DE prompts, thinking=off "
                     "(comparing final answers).\n";
    }

    // Exact-path baseline once per prompt (deterministic, verified above);
    // every case compares its candidate against it.
    std::vector<std::vector<std::int32_t>> baseline;
    baseline.reserve(items.size());
    for (const auto& it : items) baseline.push_back(genUnder(false, false, it.ids));

    const std::vector<FlagCase> cases = {
        {"F32_TC_PREFILL",     true,  false},
        {"CUBLAS_FP8_PREFILL", false, true },
        {"BOTH",               true,  true },
    };

    int exitCode = 0;
    for (const auto& c : cases) {
        std::cout << "\n========== case " << c.name << " ==========\n";
        std::size_t identical = 0;
        for (std::size_t i = 0; i < items.size(); ++i) {
            const auto& it   = items[i];
            const auto& ref  = baseline[i];                            // exact
            const auto cand = genUnder(c.f32TcPrefill, c.cublasFp8Prefill,
                                       it.ids);
            const std::size_t cp = commonPrefix(ref, cand);
            const bool same = (ref == cand);
            if (same) {
                ++identical;
            } else {
                std::cout << "  DIVERGE @tok " << cp << " — prompt: \""
                          << it.text << "\"\n";
                std::cout << "    ref : "
                          << tok.decode(ref, /*skipSpecial=*/true) << "\n";
                std::cout << "    cand: "
                          << tok.decode(cand, /*skipSpecial=*/true) << "\n";
            }
        }
        const bool exact = (identical == items.size());
        std::cout << "  -> " << identical << "/" << items.size()
                  << " token-exact — "
                  << (exact ? "TOKEN-EXACT (safe to promote apply:true)"
                            : "DIVERGES (human semantic review of the text "
                              "above required before apply:true)")
                  << "\n";
        if (!exact) exitCode = 2;
    }

    std::cout << "\n[gate] "
              << (exitCode == 0
                      ? "ALL cases TOKEN-EXACT — the lossy flags do not change "
                        "greedy output on this goldset; safe to flip apply:true."
                      : "some cases DIVERGE — review the ref/cand text above; "
                        "do not auto-promote apply:true.")
              << "\n";
    MM_LOG_INFO("main", "quality-gate done — exit {}", exitCode);
    return exitCode;
}

} // namespace mimirmind::cli
