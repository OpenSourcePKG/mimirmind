// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/spec/NGramDrafter.hpp"

#include "core/log/Log.hpp"

#include <algorithm>
#include <stdexcept>

namespace mimirmind::runtime {

NGramDrafter::NGramDrafter(Config cfg) : _cfg{cfg} {
    if (_cfg.minK == 0) {
        throw std::invalid_argument("NGramDrafter: minK must be >= 1");
    }
    if (_cfg.maxK < _cfg.minK) {
        throw std::invalid_argument("NGramDrafter: maxK must be >= minK");
    }
    MM_LOG_INFO("spec-dec",
                "NGramDrafter ready — minK={} maxK={}",
                _cfg.minK, _cfg.maxK);
}

compute::SpeculativeBatch
NGramDrafter::propose(std::span<const std::int32_t>  context,
                      std::size_t                    N,
                      const compute::SamplingParams& /*sampling*/) {
    compute::SpeculativeBatch out;
    if (N == 0) {
        return out;
    }
    out.tokens = lookup(context, N, _cfg.maxK, _cfg.minK);
    out.probs.assign(out.tokens.size(), 1.0F);
    return out;
}

std::vector<std::int32_t>
NGramDrafter::lookup(std::span<const std::int32_t> context,
                     std::size_t                   N,
                     std::size_t                   maxK,
                     std::size_t                   minK) {
    if (N == 0 || minK == 0 || maxK < minK) {
        return {};
    }
    const std::size_t len = context.size();
    // Need at least `minK` tokens for a needle plus one earlier
    // follow-up token — i.e. `len >= minK + 1` — otherwise no lookup
    // can succeed at any k.
    if (len < minK + 1) {
        return {};
    }
    // Clamp the top of the k-sweep to what fits with at least one
    // usable follow-up slot: needle occupies `[len-k, len-1]`, match
    // position and follow-up must live in `[0, len-k-1]`, so
    // `len >= k + 1` i.e. `k <= len - 1`.
    const std::size_t topK = std::min(maxK, len - 1);

    for (std::size_t k = topK; k >= minK; --k) {
        const std::size_t needleStart = len - k;
        // Backward scan over match positions before the needle. A
        // useful match satisfies `pos + k < needleStart` so there is
        // at least one follow-up token that does not overlap the
        // needle. Positions closer to the needle (`pos + k == needleStart`
        // or larger) get skipped as no-follow-up.
        for (std::size_t pos = needleStart; pos-- > 0; ) {
            bool match = true;
            for (std::size_t i = 0; i < k; ++i) {
                if (context[pos + i] != context[needleStart + i]) {
                    match = false;
                    break;
                }
            }
            if (!match) {
                continue;
            }
            const std::size_t followStart = pos + k;
            if (followStart >= needleStart) {
                // Overlap with the needle — no clean follow-up. Keep
                // scanning earlier match positions at this k.
                continue;
            }
            const std::size_t maxFollow = needleStart - followStart;
            const std::size_t take      = std::min(N, maxFollow);
            return {context.begin() + static_cast<std::ptrdiff_t>(followStart),
                    context.begin() + static_cast<std::ptrdiff_t>(followStart + take)};
        }
        if (k == minK) {
            break;
        }
    }
    return {};
}

} // namespace mimirmind::runtime