// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace mimirmind::runtime::serving {

/**
 * Scheduling-relevant snapshot of one active request. The scheduler is
 * **stateless** — the RequestScheduler (Sub-Step C1) owns the full
 * request lifecycle and passes an updated slice snapshot into
 * `schedule()` every iteration.
 */
struct RequestSlice {
    /// Opaque identifier the scheduler echoes back in its output —
    /// caller resolves it against the full request-state map.
    std::uint64_t request_id{0};

    /// Prompt tokens NOT yet prefilled. Drops to 0 once prefill
    /// completes, then `is_decoding` flips to true.
    std::int32_t tokens_pending{0};

    /// Decoded output tokens so far. Not consumed by C3, kept for
    /// future policies (e.g. decode-priority-by-age).
    std::int32_t tokens_decoded{0};

    /// True once the request has fully prefilled and is now decoding
    /// one token per iteration.
    bool is_decoding{false};
};

/**
 * One decode-token assignment for the upcoming batch iteration.
 * Implies exactly one query token for this request (the standard
 * decode step). Caller advances `tokens_decoded += 1` after the
 * forward pass returns.
 */
struct DecodeAssignment {
    std::uint64_t request_id{0};
};

/**
 * One prefill assignment. `chunk_size` prompt tokens from the request
 * will be processed this iteration; if `chunk_size <
 * slice.tokens_pending` the caller reduces `tokens_pending` by
 * `chunk_size` and reschedules the remainder next iteration. Long
 * prompts naturally split across iterations without any extra
 * bookkeeping in this class.
 */
struct PrefillAssignment {
    std::uint64_t request_id{0};
    std::int32_t  chunk_size{0};
};

/**
 * Combined schedule for one forward-pass iteration. `decodes` and
 * `prefills` are packed into a single batched forward pass by the
 * kernel-launch layer (Phase B PagedAttention + Cat-A batch-oblivious
 * kernels).
 */
struct BatchSchedule {
    std::vector<DecodeAssignment>  decodes;
    std::vector<PrefillAssignment> prefills;

    /// Sum of scheduled tokens = decodes.size() + Σ prefill.chunk_size.
    /// Always ≤ `ChunkedPrefillScheduler::tokenBudget()`.
    std::int32_t total_tokens_scheduled{0};
};

/**
 * Chunked-Prefill scheduler — Sub-Step C3 of M-Cuda.Batch.
 *
 * vLLM V1 / Sarathi-Serve pattern: unified prefill+decode iteration
 * with a fixed token budget per iteration. Decodes take priority
 * (one-token-per-request → predictable low latency), remaining budget
 * is filled with prefill chunks. Long prompts split across iterations
 * — a 3k-token prompt at `tokenBudget=512` takes 6 iterations to
 * prefill, and decode-requests keep making progress in every one of
 * those 6 iterations.
 *
 * ---- Design rules committed for Bragi-v1 -------------------------
 *
 * 1. **Decode-first priority.** All eligible decodes are scheduled
 *    before any prefill work. Cap = min(remaining_budget, decode_count).
 * 2. **Prefill packing in input order.** No priority sort — caller
 *    (RequestScheduler in C1) controls fairness via the input order.
 * 3. **Chunk = min(remaining_budget, tokens_pending).** Simple.
 * 4. **Stateless.** Same input → same output; no hidden state.
 *
 * ---- Non-goals in C3 --------------------------------------------
 *
 * - **Preemption.** When decodes > budget, the excess is dropped
 *   from this iteration's schedule. Real Prod behaviour (release KV
 *   blocks, mark as `Preempted`, re-enqueue for RECOMPUTE) lives in
 *   Sub-Step C5. C3 is the *math*, C5 is the *policy*.
 * - **Priority / SLA / age-based reordering.** Fair-input-order for
 *   Bragi-v1. Priority policies are follow-up milestones.
 * - **Cross-iteration memoisation.** Every call is independent.
 *
 * ---- Threading --------------------------------------------------
 *
 * `schedule()` is `const` and pure — safe to call concurrently on
 * disjoint inputs. In practice the scheduler owns exactly one
 * instance called serially on its event-loop thread.
 */
class ChunkedPrefillScheduler {
public:
    /**
     * vLLM V1 default. Configurable per instance (e.g. lower on
     * memory-constrained hosts where an iteration must fit in
     * pinned-workspace budgets — Phase C wiring will pull this from
     * `serving.token_budget` in `config.json`).
     */
    static constexpr std::int32_t kDefaultTokenBudget = 512;

    /**
     * `tokenBudget` MUST be > 0. Zero would deadlock (no forward-pass
     * progress ever schedulable); negative is nonsensical. Ctor
     * throws `std::invalid_argument` on either.
     */
    explicit ChunkedPrefillScheduler(std::int32_t tokenBudget = kDefaultTokenBudget);

    ChunkedPrefillScheduler(const ChunkedPrefillScheduler&)            = default;
    ChunkedPrefillScheduler& operator=(const ChunkedPrefillScheduler&) = default;
    ChunkedPrefillScheduler(ChunkedPrefillScheduler&&) noexcept        = default;
    ChunkedPrefillScheduler& operator=(ChunkedPrefillScheduler&&) noexcept = default;

    [[nodiscard]] std::int32_t tokenBudget() const noexcept { return _tokenBudget; }

    /**
     * Build the batch schedule for one forward-pass iteration.
     * Never throws; returns an empty schedule on an empty input.
     */
    [[nodiscard]] BatchSchedule schedule(std::span<const RequestSlice> requests) const noexcept;

private:
    std::int32_t _tokenBudget;
};

} // namespace mimirmind::runtime::serving