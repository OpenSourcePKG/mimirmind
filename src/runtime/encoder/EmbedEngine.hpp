// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "model/XlmRobertaTokenizer.hpp"
#include "runtime/encoder/EncoderModel.hpp"
#include "runtime/encoder/EncoderRunner.hpp"

#include <cstddef>
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
 * A loaded bi-encoder embedding model: tokenizer + weights + forward, wrapped
 * as a single serving unit behind /v1/embeddings. Given a text it returns a
 * `hidden`-dim unit vector (bge-style CLS pooling + L2 normalize) — the shared
 * EncoderRunner forward, minus the classifier head.
 *
 * The sibling of RerankEngine: same EncoderModel + XLM-R tokenizer, same shared
 * ComputeOps / ComputeMatmul, but loads a checkpoint WITHOUT a classifier head
 * (EncoderModel::hasClassifier() is false) and pools/normalizes instead of
 * scoring. Not thread-safe: the caller serialises calls per engine.
 */
class EmbedEngine {
public:
    /// `modelDir` holds config.json, model.safetensors (dense F32 XLM-R
    /// embedding checkpoint, e.g. bge-m3) and sentencepiece.bpe.model. Loads
    /// everything up front. Throws on any missing/malformed file.
    EmbedEngine(std::string_view modelDir,
                compute::ComputeOps& ops,
                compute::ComputeMatmul& matmul);

    /// Embed one text → a `hidden`-dim L2-normalized vector (CLS-pooled).
    [[nodiscard]] std::vector<float> embed(std::string_view text) const;

    /// Embed each text independently → one unit vector per input, input order
    /// preserved. (One forward per text for now; batched pooling is a later
    /// increment, mirroring forwardLogitsBatch.)
    [[nodiscard]] std::vector<std::vector<float>>
    embedBatch(std::span<const std::string> texts) const;

    /// Embedding dimensionality (= encoder hidden size).
    [[nodiscard]] std::size_t dim() const noexcept { return _model.config().hidden; }

private:
    EncoderModel               _model;
    model::XlmRobertaTokenizer _tokenizer;
    mutable EncoderRunner      _runner;
};

} // namespace mimirmind::runtime::encoder
