// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstdint>
#include <vector>

namespace mimirmind::runtime {

/// 8.19.14 part B — per-token log-probability capture for the OpenAI
/// `logprobs` / `top_logprobs` response fields.
///
/// Computed on the serving decode path (ServingSession) from the RAW lm-head
/// logits row via a full-vocab log-softmax — i.e. the model's own probability
/// for the token it actually emitted, independent of temperature / penalties /
/// grammar masking (the chosen token is always grammar-allowed, so its raw
/// logit survives an in-place mask). Tokenizer-agnostic: token ids only; the
/// server layer decodes ids -> text/bytes for the JSON response.
///
/// Opt-in per request: only slots whose SamplingParams::logprobsTopN >= 0
/// populate this (captured=true); every other slot leaves captured=false and
/// the greedy/argmax fast path is untouched.
struct TopLogprob {
    std::int32_t token{-1};
    float        logprob{0.0F};
};

struct TokenLogprobs {
    std::int32_t             token{-1};     // the emitted token id
    float                    logprob{0.0F}; // log P(token) over the full vocab
    std::vector<TopLogprob>  top;           // top-N alternatives, prob-desc
    bool                     captured{false};
};

} // namespace mimirmind::runtime
