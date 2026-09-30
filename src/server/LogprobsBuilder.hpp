// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "runtime/TokenLogprobs.hpp"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <vector>

namespace mimirmind::model {
class Tokenizer;
}

namespace mimirmind::server {

/// Builds the OpenAI `logprobs` object (8.19.14 part B) from the generated
/// token stream + its captured per-token logprobs. Emitted over the GENERATED
/// tokens (the model's raw output, like vLLM); token pieces + UTF-8 bytes are
/// decoded here so the serving layer stays tokenizer-agnostic. Extracted
/// verbatim from the former free helpers (roadmap 8.30.11.2).
class LogprobsBuilder {
public:
    /// One `content[]` entry: {token, logprob, bytes, top_logprobs[]}. Shared by
    /// the blocking path (whole-stream build) and the streaming path (per-token).
    [[nodiscard]] static nlohmann::json entryJson(
        std::int32_t                  tokId,
        const runtime::TokenLogprobs& lp,
        const model::Tokenizer&       tok);

    /// The full `{content:[...]}` object over all captured tokens (blocking).
    [[nodiscard]] static nlohmann::json buildJson(
        const std::vector<std::int32_t>&           generated,
        const std::vector<runtime::TokenLogprobs>& lp,
        const model::Tokenizer&                    tok);
};

} // namespace mimirmind::server
