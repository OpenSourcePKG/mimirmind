// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "model/XlmRobertaTokenizer.hpp"
#include "runtime/encoder/DecisionHead.hpp"
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
 * A "System-One" typed-decision serving unit (8.23) behind POST /v1/decide.
 * Wraps the same frozen bge-m3 encoder as EmbedEngine (tokenizer + EncoderModel
 * + shared EncoderRunner) plus one or more trained DecisionHeads. Given a text
 * it produces a typed answer per requested question in a single encoder forward
 * pass + a tiny host-side matvec per head — ~ms, not an LLM turn.
 *
 * The encoder produces the identical L2-normalized CLS embedding that
 * /v1/embeddings returns, so a head trained on /v1/embeddings features is
 * evaluated on exactly that feature at runtime (offline/online parity).
 *
 * Heads are discovered under `headsDir`: every immediate subdirectory that
 * contains a `head.json` is loaded as one DecisionHead, keyed by its `name`.
 * An engine with zero heads still loads and serves (every question resolves to
 * "no head" so the caller falls back to the LLM) — this is the substrate that
 * the offline-trained heads drop into. Not thread-safe: the caller serialises
 * calls per engine (the shared EncoderRunner has mutable scratch).
 */
class DecideEngine {
public:
    /// One resolved question: the head's typed decision, or `hasHead=false`
    /// when no head is loaded for that question name (caller → LLM fallback).
    struct Answer {
        std::string           question;
        bool                  hasHead{false};
        DecisionHead::Result  result;   // meaningful only when hasHead
    };

    /// `encoderDir` holds the dense F32 XLM-R embedding checkpoint (config.json,
    /// model.safetensors, sentencepiece.bpe.model — e.g. bge-m3), exactly like
    /// EmbedEngine. `headsDir` is scanned for head subdirectories; a missing or
    /// empty headsDir yields a head-less engine (valid, serves fallback-only).
    /// Throws on a missing/malformed encoder or a malformed head.
    DecideEngine(std::string_view encoderDir,
                 std::string_view headsDir,
                 compute::ComputeOps& ops,
                 compute::ComputeMatmul& matmul);

    /// Answer each requested question for `text`. When `questions` is empty,
    /// every loaded head is evaluated. Unknown question names come back with
    /// `hasHead=false`. One encoder forward is shared across all heads.
    [[nodiscard]] std::vector<Answer>
    decide(std::string_view text, std::span<const std::string> questions) const;

    /// The loaded head names (question keys), for /v1/models + diagnostics.
    [[nodiscard]] std::vector<std::string> headNames() const;

    /// Number of loaded heads.
    [[nodiscard]] std::size_t headCount() const noexcept { return _heads.size(); }

    /// Embedding dimensionality (= encoder hidden size).
    [[nodiscard]] std::size_t dim() const noexcept { return _model.config().hidden; }

    /// Device bytes held by the encoder weights (heads are host-side and tiny;
    /// not counted). For /v1/system/memory per-model attribution.
    [[nodiscard]] std::size_t weightBytes() const noexcept { return _model.weightBytes(); }

private:
    [[nodiscard]] const DecisionHead* findHead(std::string_view name) const;

    EncoderModel               _model;
    model::XlmRobertaTokenizer _tokenizer;
    mutable EncoderRunner      _runner;
    std::vector<DecisionHead>  _heads;
};

} // namespace mimirmind::runtime::encoder
