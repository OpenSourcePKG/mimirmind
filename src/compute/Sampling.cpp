// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

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

[[nodiscard]] bool penaltiesActive(const SamplingParams& p) {
    return p.penaltyWindow > 0
        && (p.repetitionPenalty != 1.0F
            || p.frequencyPenalty  != 0.0F
            || p.presencePenalty   != 0.0F);
}

/// Apply Gemma-4 final-logit softcap in place: `cap * tanh(l / cap)`.
/// Monotonic in `l`, so argmax is preserved; sampling distributions
/// downstream are strictly softer for `|l| > cap`. Caller must guarantee
/// `cap > 0`.
void applyFinalLogitSoftcapInPlace(std::span<float> logits, float cap) {
    const float invCap = 1.0F / cap;
    for (float& l : logits) {
        l = cap * std::tanh(l * invCap);
    }
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

void Sampler::applyPenalties(std::span<const std::int32_t> recentTokens,
                             const SamplingParams&         params) {
    // Precondition — `_penaltyLogits` already sized and filled by caller.
    if (recentTokens.empty()) {
        return;
    }

    // Restrict window to the tail of `recentTokens`. Callers may pass
    // the whole history; the sampler owns the "how far back" policy.
    const std::size_t n      = recentTokens.size();
    const std::size_t win    = std::min<std::size_t>(params.penaltyWindow, n);
    const auto        window = recentTokens.subspan(n - win);

    // Reuse the count buffer across calls; size to vocab.
    const std::size_t V = _penaltyLogits.size();
    _penaltyCount.assign(V, 0U);

    for (const std::int32_t t : window) {
        if (t < 0) {
            continue; // -1 or sentinel — skip
        }
        const std::size_t ut = static_cast<std::size_t>(t);
        if (ut >= V) {
            continue; // out of range — model/vocab mismatch, skip defensively
        }
        _penaltyCount[ut] += 1U;
    }

    const float rep  = params.repetitionPenalty;
    const float freq = params.frequencyPenalty;
    const float pres = params.presencePenalty;

    for (std::size_t i = 0; i < V; ++i) {
        const std::uint32_t c = _penaltyCount[i];
        if (c == 0U) {
            continue;
        }
        float l = _penaltyLogits[i];
        // llama.cpp-style multiplicative repetition penalty first, so
        // subsequent subtractive penalties compose linearly.
        if (rep != 1.0F) {
            l = (l > 0.0F) ? (l / rep) : (l * rep);
        }
        if (freq != 0.0F) {
            l -= freq * static_cast<float>(c);
        }
        if (pres != 0.0F) {
            l -= pres;
        }
        _penaltyLogits[i] = l;
    }
}

std::int32_t Sampler::sample(std::span<const float>        logits,
                             std::span<const std::int32_t> recentTokens,
                             const SamplingParams&         params) {
    if (logits.empty()) {
        throw std::runtime_error("Sampler::sample: empty logits");
    }

    const bool softcapOn = params.finalLogitSoftcap > 0.0F;
    const bool penaltyOn = penaltiesActive(params)
                           && !recentTokens.empty();
    // Greedy: deterministic argmax, no RNG use.
    const bool greedy    = (params.temperature <= 0.0F || params.topK == 1);

    // Fast greedy path: softcap is `cap * tanh(l / cap)` — strictly
    // monotonic, hence argmax-invariant — so when no penalties are active
    // the copy + full-vocab tanh would not change which index wins.
    // Argmax the raw logits directly and skip the ~1 MB scratch copy plus
    // V transcendentals that would otherwise run on every greedy token
    // (Gemma 4 ships softcap on by default). Penalties genuinely reorder,
    // so they still force the scratch path below.
    if (greedy && !penaltyOn) {
        return argmaxRow(logits);
    }

    // Route through the scratch buffer whenever softcap or penalties
    // would mutate the values. Softcap runs FIRST — matches llama.cpp's
    // gemma4.cpp placement (softcap inside the compute graph, penalties
    // in the sampler). Order matters: `penalty(softcap(x))` differs from
    // `softcap(penalty(x))` for repetition and frequency penalties, and
    // the target parity we care about is `penalty(softcap(x))`.
    std::span<const float> effLogits = logits;
    if (softcapOn || penaltyOn) {
        _penaltyLogits.assign(logits.begin(), logits.end());
        if (softcapOn) {
            applyFinalLogitSoftcapInPlace(std::span<float>{_penaltyLogits},
                                          params.finalLogitSoftcap);
        }
        if (penaltyOn) {
            applyPenalties(recentTokens, params);
        }
        effLogits = std::span<const float>{_penaltyLogits};
    }

    // Greedy with penalties active — argmax over the penalised logits.
    if (greedy) {
        return argmaxRow(effLogits);
    }

    const std::size_t V    = effLogits.size();
    const float       invT = 1.0F / params.temperature;

    // Order indices by logit descending (equals ordering by probability
    // since exp is monotonic). Only the top `kept` entries are ever read
    // downstream (softmax, top-P, draw all walk `[0, kept)`), so when
    // top-K bounds `kept < V` we select just those with partial_sort —
    // O(V log kept) instead of the full O(V log V) sort over a ~256k
    // Gemma-4 vocab on every sampled token. Tie order among exactly-equal
    // logits is unspecified either way (both algorithms are unstable).
    _idx.resize(V);
    for (std::size_t i = 0; i < V; ++i) {
        _idx[i] = static_cast<std::int32_t>(i);
    }
    const auto byLogitDesc = [&](std::int32_t a, std::int32_t b) {
        return effLogits[static_cast<std::size_t>(a)] >
               effLogits[static_cast<std::size_t>(b)];
    };

    // Top-K cutoff — compute first so the sort can stop at `kept`.
    std::size_t kept = V;
    if (params.topK > 0 && params.topK < V) {
        kept = params.topK;
    }

    if (kept < V) {
        std::partial_sort(_idx.begin(), _idx.begin() + kept, _idx.end(),
                          byLogitDesc);
    } else {
        // Nucleus (top-P) over the full vocab needs every entry ordered.
        std::sort(_idx.begin(), _idx.end(), byLogitDesc);
    }

    // Numerically-stable softmax over the kept (already sorted) entries.
    // Subtract max before exp; the max is the first sorted entry.
    const float maxLogit =
        effLogits[static_cast<std::size_t>(_idx[0])] * invT;
    _probs.resize(kept);
    double sum = 0.0;
    for (std::size_t i = 0; i < kept; ++i) {
        const float scaled =
            effLogits[static_cast<std::size_t>(_idx[i])] * invT;
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