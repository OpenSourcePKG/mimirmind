// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/serving/ChunkedPrefillScheduler.hpp"

#include <algorithm>
#include <stdexcept>

namespace mimirmind::runtime::serving {

ChunkedPrefillScheduler::ChunkedPrefillScheduler(std::int32_t tokenBudget)
    : _tokenBudget(tokenBudget)
{
    if (tokenBudget <= 0) {
        throw std::invalid_argument{
            "ChunkedPrefillScheduler: tokenBudget must be > 0"};
    }
}

BatchSchedule ChunkedPrefillScheduler::schedule(
    std::span<const RequestSlice> requests) const noexcept
{
    BatchSchedule out;
    std::int32_t remaining = _tokenBudget;

    // Pass 1 — decodes first. One token per request, input-order.
    // Excess decodes (more than budget) are silently dropped from
    // this iteration; the RequestScheduler's C5 preemption policy
    // handles memory-pressure separately.
    for (const auto& r : requests) {
        if (remaining <= 0) break;
        if (!r.is_decoding) continue;
        out.decodes.push_back({r.request_id});
        remaining -= 1;
    }

    // Pass 2 — fill remaining budget with prefill chunks. Input-order.
    // Long prompts naturally split across iterations because each
    // call sees a fresh snapshot of tokens_pending.
    for (const auto& r : requests) {
        if (remaining <= 0) break;
        if (r.is_decoding) continue;
        if (r.tokens_pending <= 0) continue;  // defensive: nothing to prefill
        const std::int32_t chunk = std::min(remaining, r.tokens_pending);
        out.prefills.push_back({r.request_id, chunk});
        remaining -= chunk;
    }

    out.total_tokens_scheduled = _tokenBudget - remaining;
    return out;
}

} // namespace mimirmind::runtime::serving