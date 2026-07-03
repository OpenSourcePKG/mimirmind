#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

namespace mimirmind::compute {

/**
 * Sampling knobs for one decode step.
 *
 * Default is greedy/argmax (temperature == 0). Bit-identical to a naked
 * argmax over logits, so engines that don't set sampling params keep
 * deterministic behavior.
 *
 * Order of operations when temperature > 0:
 *   apply penalties over recent tokens  →  logits / T  →  softmax
 *      →  top-k truncate  →  top-p truncate  →  draw
 *
 * Penalty semantics for each token id `t` that occurs in the recent
 * window (M7f):
 *   if logits[t] > 0: logits[t] /= repetitionPenalty
 *   else            : logits[t] *= repetitionPenalty
 *   logits[t] -= frequencyPenalty * count(t in window)
 *   logits[t] -= presencePenalty  (if t appears at all)
 *
 * `frequencyPenalty` and `presencePenalty` match OpenAI's semantics
 * (subtractive). `repetitionPenalty` matches llama.cpp's convention
 * (multiplicative). All three defaults are neutral so a caller that
 * sets only OpenAI's params keeps predictable behaviour; the server
 * layer picks the operator-chosen defaults on top.
 *
 * Stable, matches the llama.cpp convention used by `llama_sample_*`.
 */
struct SamplingParams {
    /// 0.0 (or negative) => deterministic argmax. Otherwise divide logits.
    float        temperature{0.0F};

    /// 0 => disabled (keep full vocab). Otherwise keep only the top-K most
    /// probable tokens after temperature scaling.
    std::size_t  topK{0};

    /// 1.0 => disabled. Otherwise keep the smallest set of top-sorted
    /// tokens whose cumulative probability >= topP, then renormalize.
    float        topP{1.0F};

    /// Seed for the RNG. 0 => non-deterministic (std::random_device).
    std::uint64_t seed{0};

    /// OpenAI-compatible frequency penalty. Range typically [-2, 2].
    /// 0 => disabled. Positive discourages repeat tokens proportional
    /// to how often they appeared in the recent window.
    float        frequencyPenalty{0.0F};

    /// OpenAI-compatible presence penalty. Range typically [-2, 2].
    /// 0 => disabled. Positive discourages any token that appeared at
    /// all in the recent window, regardless of count.
    float        presencePenalty{0.0F};

    /// llama.cpp-style repetition penalty. Range typically [1.0, 1.3].
    /// 1.0 => disabled (multiplicative identity). Values > 1 shrink
    /// logits of repeated tokens, values < 1 amplify them.
    float        repetitionPenalty{1.0F};

    /// Number of most-recent tokens the penalties look at. 0 => disabled
    /// (no penalty applied even if the penalty fields are non-neutral).
    /// llama.cpp default is 64.
    std::uint32_t penaltyWindow{0};
};

/**
 * Stateful token sampler with a seedable RNG and reusable scratch buffers
 * (avoids per-call vocab-sized allocations). One instance per engine.
 *
 * Not thread-safe: the RNG and scratch are mutated by sample().
 */
class Sampler {
public:
    Sampler();
    explicit Sampler(std::uint64_t seed);

    /// Reset the generator to a specific seed. seed == 0 reseeds from
    /// std::random_device (non-deterministic).
    void reseed(std::uint64_t seed);

    /**
     * Draw one token id from `logits`.
     *
     * Greedy path (temperature <= 0 OR topK == 1): plain argmax, no RNG
     * use, bit-identical across runs. Sampling path: apply penalties over
     * the last `params.penaltyWindow` entries of `recentTokens`, then
     * temperature scale, stable softmax, top-K cutoff, top-P (nucleus)
     * cutoff, multinomial draw.
     *
     * `recentTokens` is treated as ordered-oldest-to-newest; only its tail
     * of at most `params.penaltyWindow` ids contributes to the count.
     * Empty span or `penaltyWindow == 0` skips the penalty pass entirely.
     * The caller's `logits` is never mutated — when penalties apply the
     * sampler copies into a scratch buffer.
     */
    std::int32_t sample(std::span<const float>        logits,
                        std::span<const std::int32_t> recentTokens,
                        const SamplingParams&         params);

    /// Backwards-compatible overload without recent-tokens context.
    /// Semantically identical to sample(logits, {}, params).
    std::int32_t sample(std::span<const float> logits,
                        const SamplingParams&  params) {
        return sample(logits, std::span<const std::int32_t>{}, params);
    }

private:
    /// Apply penalties to `_penaltyLogits`. Precondition: caller has
    /// copied the raw logits into `_penaltyLogits`. See class docstring
    /// for the multiplicative + subtractive composition rules.
    void applyPenalties(std::span<const std::int32_t> recentTokens,
                        const SamplingParams&         params);

    std::mt19937_64            _rng;
    std::vector<std::int32_t>  _idx;           // sorted indices
    std::vector<float>         _probs;         // working probability buffer
    std::vector<std::uint32_t> _penaltyCount;  // reused per sample()
    std::vector<float>         _penaltyLogits; // scratch for mutated logits
};

} // namespace mimirmind::compute