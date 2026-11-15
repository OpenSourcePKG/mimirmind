// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "hugin/McpServer.hpp"

#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace mimirmind::runtime {
class InferenceEngine;
}

namespace mimirmind::hugin {

/**
 * Hugin's forward-based tools (roadmap 7.2 Inc-2). Every tool runs the
 * engine's OWN forward (`InferenceEngine::forwardVerify`, linked from the
 * same object files as the serving binary) — there is no second forward
 * implementation in Hugin. The engine is loaded lazily on the first forward
 * tool call, on whatever backend `MIMIRMIND_BACKEND` / auto-select picks
 * (CPU on a box without a visible GPU).
 *
 *  - tokenize        ids + pieces of a text
 *  - next_token      top-k next-token distribution after a text
 *  - perplexity      per-token log-probs, mean NLL, perplexity
 *  - logit_lens      per block: the model's own final norm + lm_head applied
 *                    to the residual stream -> top-k tokens
 *  - activations     per block: residual-stream norms/statistics, update size
 *                    vs the previous block, strongest dimensions
 *  - rune_compare    base vs the same model with runes applied by Futhark at
 *                    load: per-position KL, top-1 agreement, perplexity delta
 *
 * Every forward starts from an empty KV cache (resetCache) — tools are
 * stateless with respect to each other.
 */
class ForwardTools {
public:
    /// `modelPath` is asked on each call so open_model (InspectTools) also
    /// retargets the forward tools; a changed path reloads the engine.
    explicit ForwardTools(std::function<std::string()> modelPath);
    ~ForwardTools();

    ForwardTools(const ForwardTools&)            = delete;
    ForwardTools& operator=(const ForwardTools&) = delete;

    void registerOn(McpServer& server);

private:
    struct Loaded;

    runtime::InferenceEngine& base();
    runtime::InferenceEngine& withRunes(const std::vector<std::string>& runes);
    std::unique_ptr<Loaded>   load(const std::string& path, const std::vector<std::string>& runes) const;

    std::function<std::string()> _modelPath;
    std::unique_ptr<Loaded>      _base;
    std::unique_ptr<Loaded>      _variant;
};

} // namespace mimirmind::hugin
