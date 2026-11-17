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
 *  - attention_pattern  per-head attention weights of one block, rebuilt from
 *                    the backend's post-RoPE Q/K stage tap, self-checked
 *                    against the kernel's own attention output
 *  - eval            forward-only goldset eval (exact continuation or
 *                    multiple choice), optionally base vs rune with flips
 *  - expert_usage    MoE routing actually taken per block (stage tap of the
 *                    selected expert ids/weights): usage histogram, load
 *                    entropy, per-token picks
 *
 * Every forward starts from an empty KV cache (resetCache) — tools are
 * stateless with respect to each other.
 */
class ForwardTools {
public:
    /// What the forward tools run on: a GGUF file or an NVFP4 checkpoint
    /// directory (+ optional tokenizer GGUF).
    struct Target {
        std::string path;
        std::string tokenizerGguf;
    };

    /// `target` is asked on each call so open_model (InspectTools) also
    /// retargets the forward tools; a changed target reloads the engine.
    explicit ForwardTools(std::function<Target()> target);
    ~ForwardTools();

    ForwardTools(const ForwardTools&)            = delete;
    ForwardTools& operator=(const ForwardTools&) = delete;

    void registerOn(McpServer& server);

    /// The base engine (lazy load / reload on a changed target). Shared with
    /// SurgeryTools so both tool groups run on one resident model.
    runtime::InferenceEngine& engine() { return base(); }

    /// The same model with `runes` applied by Futhark at load (one variant
    /// engine is kept besides the base; a different rune list reloads it).
    runtime::InferenceEngine& engineWithRunes(const std::vector<std::string>& runes) {
        return withRunes(runes);
    }

    /// Drop the rune variant engine (e.g. after a staged rune file changed).
    void dropVariant() noexcept;

    /// Keep at most ONE engine resident (--single-engine): loading the rune
    /// variant releases the base and vice versa, for models too large to hold
    /// base + variant at once (Kolibri-1 78B on a 128 GB GB10). References
    /// returned by engine() / engineWithRunes() are invalidated by the next
    /// call to the other one.
    void setSingleEngine(bool on) noexcept { _singleEngine = on; }
    [[nodiscard]] bool singleEngine() const noexcept { return _singleEngine; }

    /// Identity of the current target (path + tokenizer); session state that
    /// is only valid for one model (steering vectors) keys on it.
    [[nodiscard]] std::string targetKey() const;

    /// The current target (as InspectTools / open_model set it).
    [[nodiscard]] Target target() const { return _target(); }

private:
    struct Loaded;

    runtime::InferenceEngine& base();
    runtime::InferenceEngine& withRunes(const std::vector<std::string>& runes);
    std::unique_ptr<Loaded>   load(const Target& t, const std::vector<std::string>& runes) const;

    std::function<Target()>      _target;
    std::unique_ptr<Loaded>      _base;
    std::unique_ptr<Loaded>      _variant;
    bool                         _singleEngine{false};
};

} // namespace mimirmind::hugin
