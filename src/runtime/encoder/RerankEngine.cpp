// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/encoder/RerankEngine.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace mimirmind::runtime::encoder {

RerankEngine::RerankEngine(std::string_view modelDir,
                           compute::ComputeOps& ops,
                           compute::ComputeMatmul& matmul)
    : _runner{_model, ops, matmul} {
    // BF16 linear weights -> tensor-core GEMM at M>1 (fast batched rerank);
    // ranking is robust to the bf16 rounding, activations stay F32.
    _model.load(modelDir, ops, core::gguf::GgmlType::BF16);
    const std::filesystem::path spm =
        std::filesystem::path{modelDir} / "sentencepiece.bpe.model";
    _tokenizer.load(spm.string());
}

float RerankEngine::scorePair(std::string_view query,
                              std::string_view document) const {
    const std::vector<std::int32_t> ids =
        _tokenizer.encodePair(query, document);
    return _runner.score(ids);
}

std::vector<RerankEngine::Result>
RerankEngine::rerank(std::string_view query,
                     std::span<const std::string> documents,
                     bool sortDescending,
                     std::optional<std::size_t> topN) const {
    // Tokenize every (query, document) pair, then score the whole pool in ONE
    // batched forward (padded, per-sequence attention masking) — far faster
    // than one forward per document.
    std::vector<std::vector<std::int32_t>> pairs;
    pairs.reserve(documents.size());
    for (const auto& doc : documents) {
        pairs.push_back(_tokenizer.encodePair(query, doc));
    }

    const std::vector<std::vector<float>> logits =
        _runner.forwardLogitsBatch(std::span<const std::vector<std::int32_t>>{pairs});

    std::vector<Result> results;
    results.reserve(documents.size());
    for (std::size_t i = 0; i < documents.size(); ++i) {
        const float score = (i < logits.size() && !logits[i].empty())
                                ? logits[i].front() : 0.0F;
        results.push_back({i, score});
    }

    if (sortDescending) {
        std::stable_sort(results.begin(), results.end(),
                         [](const Result& a, const Result& b) {
                             return a.score > b.score;
                         });
    }
    if (topN.has_value() && *topN < results.size()) {
        results.resize(*topN);
    }
    return results;
}

} // namespace mimirmind::runtime::encoder
