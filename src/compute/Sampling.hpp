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
 *   logits / T  →  softmax  →  top-k truncate  →  top-p truncate  →  draw
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
     * use, bit-identical across runs. Sampling path: temperature scale,
     * stable softmax, top-K cutoff, top-P (nucleus) cutoff, multinomial.
     */
    std::int32_t sample(std::span<const float> logits,
                        const SamplingParams&  params);

private:
    std::mt19937_64           _rng;
    std::vector<std::int32_t> _idx;        // sorted indices
    std::vector<float>        _probs;      // working probability buffer
};

} // namespace mimirmind::compute