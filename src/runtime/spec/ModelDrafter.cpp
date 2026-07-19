// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/spec/ModelDrafter.hpp"

#include "runtime/InferenceEngine.hpp"

namespace mimirmind::runtime {

ModelDrafter::ModelDrafter(InferenceEngine& draft) : _draft{draft} {}

compute::SpeculativeBatch
ModelDrafter::propose(std::span<const std::int32_t>  context,
                      std::size_t                    N,
                      const compute::SamplingParams& sampling) {
    return _sampler.speculate(_draft, context, N, sampling);
}

void ModelDrafter::warmPrefix(std::span<const std::int32_t> promptPlusFirstToken) {
    // One-token draft.generate() primes the draft's prefix cache so the
    // first propose() call only prefills the divergent suffix. The
    // returned token is discarded — only the resulting KV state matters.
    GenerateParams primeParams{};
    primeParams.maxNewTokens = 1;
    (void)_draft.generate(promptPlusFirstToken, primeParams);
}

} // namespace mimirmind::runtime