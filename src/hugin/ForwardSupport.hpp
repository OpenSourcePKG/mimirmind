// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "runtime/InferenceEngine.hpp"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

/**
 * Helpers shared by Hugin's forward-based tool groups (ForwardTools,
 * SurgeryTools): argument parsing, the stateless forward, log-probs and
 * top-k rendering. Free functions so tool lambdas call them unqualified via
 * `using namespace forward_support;`.
 */
namespace mimirmind::hugin::forward_support {

using Engine = runtime::InferenceEngine;

/// JSON-schema object with `properties` / `required`.
nlohmann::json schema(nlohmann::json properties, nlohmann::json required = nlohmann::json::array());

/// text / tokens / add_bos input properties, plus `extra`.
nlohmann::json textProps(nlohmann::json extra = nlohmann::json::object());

/// Token ids from `tokens` or `text` (+ add_bos); throws when empty.
std::vector<std::int32_t> inputTokens(const Engine& e, const nlohmann::json& a);

/// Prompt text -> ids with special markup parsed (`<|im_start|>` etc. become
/// their single token id, as the engine's chat-template path encodes them), so
/// a rendered chat prompt reaches the forward exactly as in serving. Plain text
/// without special markup encodes as before.
std::vector<std::int32_t> encodeText(const Engine& e, std::string_view text, bool addBos);

/// Same, for a text under a different key (e.g. "clean" / "corrupt").
std::vector<std::int32_t> textTokens(const Engine& e, const std::string& text, bool addBos);

std::string piece(const Engine& e, std::int32_t id);

/// One stateless forward: empty cache in, empty cache out.
std::vector<std::vector<float>> forward(Engine& e, std::span<const std::int32_t> ids,
                                        Engine::ForwardProbe* probe = nullptr);

/// Final soft-cap + log-softmax of one logits row.
std::vector<double> logProbs(const Engine& e, std::vector<float> logits);

nlohmann::json topJson(const Engine& e, const std::vector<double>& logp, std::size_t k);

/// Requested `blocks` (default: all), validated, sorted, unique.
std::vector<std::uint32_t> blockList(const Engine& e, const nlohmann::json& a);

/// `position` (negative counts from the end; default last) validated against n.
std::size_t positionArg(const nlohmann::json& a, std::size_t n);

/// Per-token log-probs of ids[1..] under `logits` (position i predicts i+1).
std::vector<double> tokenLogProbs(const Engine& e, const std::vector<std::vector<float>>& logits,
                                  std::span<const std::int32_t> ids);

std::size_t nonFiniteCount(const std::vector<double>& v);

double perplexityOf(const std::vector<double>& lps);

/// Teacher-forced score of `cont` after `prompt` in one forward.
struct ContinuationScore {
    double      logprob{0};
    bool        greedyMatch{true};   // every continuation token is the argmax
    std::size_t firstRank{0};
    std::size_t tokens{0};
};

/// Installs per-forward interventions (7.3 previews) into a fresh probe for
/// an n-token forward. Null = plain forward.
using ProbeSetup = std::function<void(Engine::ForwardProbe& probe, std::size_t n)>;

ContinuationScore scoreContinuation(Engine& e, const std::vector<std::int32_t>& prompt,
                                    const std::vector<std::int32_t>& cont,
                                    const ProbeSetup* setup = nullptr);

/// Greedy continuation of `prompt`, stopping at EOS / the chat template's stop
/// ids. Without `setup` it runs the engine's KV-cached generate() (the serving
/// decode path, fast); with `setup` every step is a stateless forward with the
/// interventions installed (slow, but the edits apply to every token).
std::vector<std::int32_t> generateGreedy(Engine& e, const std::vector<std::int32_t>& prompt,
                                         std::size_t maxTokens, const ProbeSetup* setup = nullptr);

/// Goldset items: inline `items` array or a .json (array) / .jsonl file at `path`.
nlohmann::json loadGoldset(const nlohmann::json& a);

/// One eval item: {"prompt", "expect"} (exact continuation, greedy match),
/// {"prompt", "choices":[...], "answer": idx} (multiple choice by logprob) or
/// {"prompt", "generate": {...}} (greedy free generation checked by
/// GenerationCheck: language, code-mixing, regex facts; result carries
/// "checks" and "text", and no logprob).
nlohmann::json evalItem(Engine& e, const nlohmann::json& item, bool addBos,
                        const ProbeSetup* setup = nullptr);

/// Whole goldset on one engine; per-item results land in rows[i]["base_or_run"].
/// Summary: items, accuracy, mean_logprob (finite scored items; generate
/// items have no logprob and are left out), non_finite_items, generate_items.
/// `setup` (optional) applies the same interventions to every forward.
nlohmann::json evalRun(Engine& e, const nlohmann::json& items, bool addBos, nlohmann::json& rows,
                       const ProbeSetup* setup = nullptr);

/// Per-check regressions of generate items between two evalRun row sets: every
/// check that passed on `baseRows` and fails on `runRows`, as
/// [{index, id, check, base_text, text}]. Checks a base answer already failed
/// are not counted, so a known base error never blocks an edit.
nlohmann::json checkRegressions(const nlohmann::json& baseRows, const nlohmann::json& runRows);

} // namespace mimirmind::hugin::forward_support
