// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "model/XlmRobertaTokenizer.hpp"
#include "runtime/encoder/EncoderModel.hpp"
#include "runtime/encoder/EncoderRunner.hpp"

#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace mimirmind::compute {
class ComputeOps;
class ComputeMatmul;
} // namespace mimirmind::compute

namespace mimirmind::runtime::encoder {

/**
 * A loaded cross-encoder reranker: tokenizer + weights + forward, wrapped as
 * a single serving unit behind /v1/rerank. Given a query and N documents it
 * returns a relevance score per document (the classifier logit) and, by
 * default, the descending ranking.
 *
 * Owns the EncoderModel and the XLM-R tokenizer; borrows the shared
 * ComputeOps / ComputeMatmul (same handles the chat engines use — one GPU
 * context per process). Not thread-safe: the caller serialises calls per
 * engine (the server holds a per-target mutex), same as the chat path.
 */
class RerankEngine {
public:
    /// `modelDir` holds config.json, model.safetensors (dense F32 XLM-R) and
    /// sentencepiece.bpe.model. Loads everything up front. Throws on any
    /// missing/malformed file.
    RerankEngine(std::string_view modelDir,
                 compute::ComputeOps& ops,
                 compute::ComputeMatmul& matmul);

    struct Result {
        std::size_t index;   // position in the input `documents`
        float       score;   // relevance logit (higher = more relevant)
    };

    /// Score every document against the query. If `sortDescending`, the result
    /// is ranked best-first; otherwise it preserves input order. `topN` caps
    /// the returned count (applied after sorting).
    [[nodiscard]] std::vector<Result>
    rerank(std::string_view query,
           std::span<const std::string> documents,
           bool sortDescending = true,
           std::optional<std::size_t> topN = std::nullopt) const;

    /// Single (query, document) relevance logit.
    [[nodiscard]] float scorePair(std::string_view query,
                                  std::string_view document) const;

private:
    EncoderModel               _model;
    model::XlmRobertaTokenizer _tokenizer;
    mutable EncoderRunner      _runner;
};

} // namespace mimirmind::runtime::encoder
