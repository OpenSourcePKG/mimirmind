#include "compute/SpeculativeSampler.hpp"

#include "runtime/InferenceEngine.hpp"

namespace mimirmind::compute {

SpeculativeBatch
SpeculativeSampler::speculate(runtime::InferenceEngine&      draft,
                              std::span<const std::int32_t>  promptSoFar,
                              std::size_t                    N,
                              const SamplingParams&          sampling)
{
    SpeculativeBatch out;
    if (N == 0) {
        return out;
    }

    runtime::GenerateParams params{};
    params.maxNewTokens = N;
    params.sampling     = sampling;

    runtime::GenerateStats stats{};
    auto drafted = draft.generate(promptSoFar, params,
                                  /*onToken=*/{},
                                  /*outStats=*/&stats);

    out.tokens = std::move(drafted);
    // Greedy placeholder — M9.11.4 replaces this with the real
    // p_draft(tokens[i]) once the draft-side sampler exposes logits.
    out.probs.assign(out.tokens.size(), 1.0F);
    out.hitStop = stats.hitStop;
    return out;
}

} // namespace mimirmind::compute