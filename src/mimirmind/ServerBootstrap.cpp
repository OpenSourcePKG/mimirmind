// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "mimirmind/ServerBootstrap.hpp"

#include "core/config/Config.hpp"
#include "core/log/Log.hpp"
#include "model/Tokenizer.hpp"
#include "runtime/InferenceEngine.hpp"
#include "runtime/spec/Drafter.hpp"
#include "runtime/spec/ModelDrafter.hpp"
#include "runtime/spec/NGramDrafter.hpp"

#include <cstddef>
#include <exception>
#include <string>

namespace mimirmind::cli {

ServerBootstrap::SpeculativeSetup ServerBootstrap::buildSpeculative(
    const core::config::Config&    cfg,
    const runtime::InferenceEngine& targetEngine) {
    // M9.11.1 — Optional speculative decoding. Two drafter variants:
    //   * `speculative.drafter == "model"` loads a second, smaller
    //     InferenceEngine and wraps it in `ModelDrafter`. Requires a
    //     vocab-compatible model resolved via `speculative.draft`.
    //   * `speculative.drafter == "ngram"` uses in-context Prompt-Lookup
    //     Decoding — no second model, no vocab check, zero USM cost.
    // Both variants only kick in when `speculative.enabled` is true.
    SpeculativeSetup setup{};
    if (!cfg.speculative.enabled) {
        return setup;
    }

    using DrafterKind = ::mimirmind::core::config::SpeculativeSettings::Drafter;
    if (cfg.speculative.drafter == DrafterKind::NGram) {
        ::mimirmind::runtime::NGramDrafter::Config nc{};
        nc.minK = static_cast<std::size_t>(cfg.speculative.ngramMinK);
        nc.maxK = static_cast<std::size_t>(cfg.speculative.ngramMaxK);
        setup.drafter = std::make_unique<::mimirmind::runtime::NGramDrafter>(nc);
        MM_LOG_INFO("main",
                    "serve: speculative decoding ready — "
                    "drafter=ngram minK={} maxK={}",
                    nc.minK, nc.maxK);
    } else if (!cfg.speculative.draft.empty()) {
        std::string draftPath;
        try {
            draftPath = cfg.model(cfg.speculative.draft).path;
        } catch (const std::exception& e) {
            MM_LOG_WARN("main",
                        "serve: speculative.draft='{}' unresolved ({}) — "
                        "speculative decoding disabled",
                        cfg.speculative.draft, e.what());
        }
        if (!draftPath.empty()) {
            MM_LOG_INFO("main",
                        "serve: loading draft model '{}'", draftPath);
            try {
                setup.draftEngine =
                    std::make_unique<::mimirmind::runtime::InferenceEngine>(cfg);
                setup.draftEngine->loadModel(draftPath);

                // Vocab compatibility. Modified rejection sampling
                // only works when draft token-id N and target token-
                // id N mean the same subword. vocabSize alone
                // doesn't guarantee it, but a mismatch there is a
                // hard disqualification. bos/eos must match too
                // because we replay the same prompt-id stream
                // through both engines.
                const auto& tTok = targetEngine.tokenizer();
                const auto& dTok = setup.draftEngine->tokenizer();
                const bool sizeMatch = tTok.vocabSize() == dTok.vocabSize();
                const bool bosMatch  = tTok.bosId()     == dTok.bosId();
                const bool eosMatch  = tTok.eosId()     == dTok.eosId();
                if (!sizeMatch || !bosMatch || !eosMatch) {
                    MM_LOG_WARN("main",
                                "serve: draft model vocab incompatible with "
                                "target — disabling speculative decoding. "
                                "target(vocab={}, bos={}, eos={}) vs "
                                "draft(vocab={}, bos={}, eos={})",
                                tTok.vocabSize(), tTok.bosId(), tTok.eosId(),
                                dTok.vocabSize(), dTok.bosId(), dTok.eosId());
                    setup.draftEngine.reset();
                } else {
                    setup.drafter =
                        std::make_unique<::mimirmind::runtime::ModelDrafter>(
                            *setup.draftEngine);
                    MM_LOG_INFO("main",
                                "serve: speculative decoding ready — "
                                "drafter=model target arch={} d_model={}, "
                                "draft arch={} d_model={} "
                                "(shared vocab_size={}, bos={}, eos={})",
                                targetEngine.config().architecture,
                                targetEngine.config().embeddingLength,
                                setup.draftEngine->config().architecture,
                                setup.draftEngine->config().embeddingLength,
                                tTok.vocabSize(), tTok.bosId(), tTok.eosId());
                }
            } catch (const std::exception& e) {
                MM_LOG_WARN("main",
                            "serve: draft model load failed ({}) — "
                            "speculative decoding disabled", e.what());
                setup.draftEngine.reset();
            }
        }
    } else {
        MM_LOG_WARN("main",
                    "serve: speculative.enabled=true with drafter='model' "
                    "but speculative.draft is empty — speculative "
                    "decoding disabled");
    }
    return setup;
}

} // namespace mimirmind::cli
