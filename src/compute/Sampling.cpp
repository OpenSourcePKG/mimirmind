#include "compute/Sampling.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mimirmind::compute {

namespace {

[[nodiscard]] std::int32_t argmaxRow(std::span<const float> logits) {
    if (logits.empty()) {
        throw std::runtime_error("Sampler: empty logits row");
    }
    std::int32_t best  = 0;
    float        bestV = logits[0];
    for (std::size_t i = 1; i < logits.size(); ++i) {
        if (logits[i] > bestV) {
            bestV = logits[i];
            best  = static_cast<std::int32_t>(i);
        }
    }
    return best;
}

} // namespace

Sampler::Sampler() : Sampler{0} {}

Sampler::Sampler(std::uint64_t seed) {
    reseed(seed);
}

void Sampler::reseed(std::uint64_t seed) {
    if (seed == 0) {
        std::random_device rd;
        const auto a = static_cast<std::uint64_t>(rd());
        const auto b = static_cast<std::uint64_t>(rd());
        _rng.seed((a << 32) ^ b);
    } else {
        _rng.seed(seed);
    }
}

std::int32_t Sampler::sample(std::span<const float> logits,
                             const SamplingParams&  params) {
    if (logits.empty()) {
        throw std::runtime_error("Sampler::sample: empty logits");
    }

    // Greedy: deterministic, no RNG use, bit-identical to argmax.
    if (params.temperature <= 0.0F || params.topK == 1) {
        return argmaxRow(logits);
    }

    const std::size_t V    = logits.size();
    const float       invT = 1.0F / params.temperature;

    // Sort indices by raw logit descending. Ordering by logit equals
    // ordering by probability since exp is monotonic.
    _idx.resize(V);
    for (std::size_t i = 0; i < V; ++i) {
        _idx[i] = static_cast<std::int32_t>(i);
    }
    std::sort(_idx.begin(), _idx.end(),
              [&](std::int32_t a, std::int32_t b) {
                  return logits[static_cast<std::size_t>(a)] >
                         logits[static_cast<std::size_t>(b)];
              });

    // Top-K cutoff (after the sort the first kept entries are exactly
    // the K most probable — no extra work).
    std::size_t kept = V;
    if (params.topK > 0 && params.topK < V) {
        kept = params.topK;
    }

    // Numerically-stable softmax over the kept (already sorted) entries.
    // Subtract max before exp; the max is the first sorted entry.
    const float maxLogit = logits[static_cast<std::size_t>(_idx[0])] * invT;
    _probs.resize(kept);
    double sum = 0.0;
    for (std::size_t i = 0; i < kept; ++i) {
        const float scaled = logits[static_cast<std::size_t>(_idx[i])] * invT;
        const float p      = std::exp(scaled - maxLogit);
        _probs[i] = p;
        sum += static_cast<double>(p);
    }
    if (sum <= 0.0) {
        // All masked / numerical underflow — fall back to argmax.
        return _idx[0];
    }
    const float invSum = static_cast<float>(1.0 / sum);
    for (auto& p : _probs) {
        p *= invSum;
    }

    // Top-P (nucleus): walk the sorted probs, accumulate, drop the tail.
    // Always keep at least one entry so we can still draw.
    if (params.topP > 0.0F && params.topP < 1.0F) {
        double cum  = 0.0;
        std::size_t cut = kept;
        for (std::size_t i = 0; i < kept; ++i) {
            cum += static_cast<double>(_probs[i]);
            if (cum >= static_cast<double>(params.topP)) {
                cut = i + 1;
                break;
            }
        }
        if (cut < kept) {
            kept = cut;
            // Renormalize over the kept head.
            double cumKept = 0.0;
            for (std::size_t i = 0; i < kept; ++i) {
                cumKept += static_cast<double>(_probs[i]);
            }
            if (cumKept <= 0.0) {
                return _idx[0];
            }
            const float invCum = static_cast<float>(1.0 / cumKept);
            for (std::size_t i = 0; i < kept; ++i) {
                _probs[i] *= invCum;
            }
        }
    }

    // Multinomial draw from the sorted, renormalized head.
    std::uniform_real_distribution<float> u01{0.0F, 1.0F};
    const float u = u01(_rng);
    float cum = 0.0F;
    for (std::size_t i = 0; i < kept; ++i) {
        cum += _probs[i];
        if (u < cum) {
            return _idx[i];
        }
    }
    // Numerical slop at the tail — return the last kept.
    return _idx[kept - 1];
}

} // namespace mimirmind::compute