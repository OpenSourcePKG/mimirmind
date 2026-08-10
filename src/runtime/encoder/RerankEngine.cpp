// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/encoder/RerankEngine.hpp"

#include <algorithm>
#include <filesystem>

namespace mimirmind::runtime::encoder {

RerankEngine::RerankEngine(std::string_view modelDir,
                           compute::ComputeOps& ops,
                           compute::ComputeMatmul& matmul)
    : _runner{_model, ops, matmul} {
    _model.load(modelDir, ops);
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
    std::vector<Result> results;
    results.reserve(documents.size());
    for (std::size_t i = 0; i < documents.size(); ++i) {
        results.push_back({i, scorePair(query, documents[i])});
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
