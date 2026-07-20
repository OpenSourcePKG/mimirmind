// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "runtime/serving/ChunkedPrefillScheduler.hpp"
#include "runtime/serving/PagedKvSequence.hpp"    // pulls in PagedKvBlockAllocator.hpp

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace mimirmind::runtime::serving {

/**
 * Per-request lifecycle state — Sub-Step C2 of M-Cuda.Batch.
 *
 * Transitions:
 *   Waiting  → Prefilling  (admitted into active set by tick())
 *   Prefilling → Decoding  (tokens_pending reaches 0)
 *   Decoding → Completed   (EOS reached OR tokens_decoded == max_tokens)
 *   Prefilling → Preempted (memory pressure — preemptOne() called)
 *   Decoding → Preempted   (memory pressure — preemptOne() called)
 *   Preempted → Waiting    (re-enqueued for RECOMPUTE — automatic)
 *
 * Completed is a terminal state until the caller explicitly removes
 * the entry via `drainCompleted()`.
 */
enum class RequestState : std::uint8_t {
    Waiting    = 0,
    Prefilling = 1,
    Decoding   = 2,
    Completed  = 3,
    Preempted  = 4,
};

/**
 * Full lifecycle state owned by `RequestScheduler`. RequestSlice
 * (from ChunkedPrefillScheduler) is a stripped-down snapshot the
 * scheduling algorithm consumes; this struct carries the full
 * picture the event loop needs.
 */
struct RequestStateData {
    std::uint64_t request_id{0};
    RequestState  state{RequestState::Waiting};
    std::int32_t  prompt_length{0};   // total prompt tokens (frozen at admit)
    std::int32_t  tokens_pending{0};  // prompt tokens NOT yet prefilled
    std::int32_t  tokens_decoded{0};  // output tokens generated so far
    std::int32_t  max_tokens{0};      // decode cap from caller

    /**
     * Monotonic admission counter — used by preemption to pick LIFO
     * targets (most-recently admitted has least invested KV, cheapest
     * to recompute). Assigned by admit() in the same order as
     * request_id.
     */
    std::uint64_t admit_seq{0};

    /**
     * Optional per-request paged-KV block-table + token-history.
     * Populated only when the scheduler ctor received a non-null
     * `PagedKvBlockAllocator*`. Ownership stays with this struct;
     * `optional`'s destructor releases the sequence's blocks back to
     * the shared pool via `PagedKvSequence::~PagedKvSequence()` when
     * the request is drained or the scheduler destructs.
     */
    std::optional<PagedKvSequence> kv_sequence;

    /**
     * Set to true the first time preemptOne() moves this request to
     * Preempted. Used only to decide whether a Completed transition
     * should bump the monotonic total_preempted_since_start counter
     * (only requests that WERE preempted at some point count). Not
     * user-facing.
     */
    bool ever_preempted{false};

    RequestStateData()                                       = default;

    // Non-copyable because PagedKvSequence is non-copyable. Move-only
    // is fine — the scheduler stores instances in a std::vector that
    // may reallocate on admit(); the move-ctor keeps the allocator
    // pointer intact because the allocator lives outside the sequence.
    RequestStateData(const RequestStateData&)            = delete;
    RequestStateData& operator=(const RequestStateData&) = delete;
    RequestStateData(RequestStateData&&) noexcept        = default;
    RequestStateData& operator=(RequestStateData&&) noexcept = default;
};

/**
 * Aggregated snapshot of the serving loop's state for
 * `/v1/system/info.serving` (Phase E3) and any operator-facing
 * dashboard. Produced by `RequestScheduler::snapshotMetrics()`.
 *
 * Instantaneous counters describe *right now*; totals are monotonic
 * across the scheduler's lifetime. Pool-side fields are 0 when the
 * scheduler runs in state-machine-only mode (no allocator).
 */
struct ServingMetrics {
    // ---- Instantaneous (right now) --------------------------------
    std::size_t   num_waiting{0};
    std::size_t   num_active{0};             // Prefilling + Decoding
    std::size_t   num_preempted{0};
    std::size_t   num_completed_current{0};  // Completed but not yet drained

    // ---- Monotonic totals since scheduler-start -------------------
    std::uint64_t total_admitted{0};
    std::uint64_t total_completed{0};
    std::uint64_t total_preempted{0};        // count of preemption events, not requests

    // ---- Pool utilisation (0 fields when no allocator) ------------
    std::size_t   block_pool_total{0};
    std::size_t   block_pool_free{0};
    std::size_t   block_pool_used{0};
    double        block_pool_utilization{0.0};   // used / total, [0.0, 1.0]
};

/**
 * Request lifecycle manager for the M-Cuda.Batch serving loop —
 * Sub-Steps C1 (event-loop class) + C2 (state machine).
 *
 * ---- Loop shape --------------------------------------------------
 *
 * Every event-loop iteration is a three-step dance:
 *
 *   1. `tick()`             — admit queued requests up to maxActive,
 *                             build the batch schedule via
 *                             ChunkedPrefillScheduler.
 *   2. Caller executes it   — Phase D's InferenceEngine::generateServing
 *                             launches PagedAttentionV1/V2 + the
 *                             batched Cat-A kernels, collects new
 *                             tokens + EOS flags.
 *   3. `commitProgress()`   — feed results back, advance state
 *                             (Prefilling → Decoding when pending=0,
 *                             Decoding → Completed on EOS/max_tokens).
 *
 * `preemptOne()` is called between (1) and (2) when memory pressure
 * requires shrinking the active set — the returned request-id
 * transitions to Preempted, its KV blocks get released by the caller
 * (Sub-Step C5 policy, out of scope for the skeleton), then the
 * scheduler automatically re-enqueues it as Waiting for RECOMPUTE on
 * the next tick.
 *
 * ---- Skeleton scope ---------------------------------------------
 *
 * The skeleton owns only the **state machine** — RequestState
 * transitions + request-registry + integration with C3. It does NOT:
 *   - own PagedKvSequence instances (follow-up commit — bind to
 *     Allocator ref, create per-request Sequence on admission,
 *     release on Completed/Preempted)
 *   - launch kernels (Phase D)
 *   - talk to HTTP (Phase D)
 *   - implement C5 preemption *policy* — only exposes preemptOne()
 *     as the mechanism the eventual policy will call
 *
 * ---- Threading --------------------------------------------------
 *
 * NOT thread-safe. Same discipline as PagedKvBlockAllocator /
 * PagedAttentionV1/V2: the serving-loop thread owns the instance and
 * serialises all mutations. Cross-thread concurrent calls are UB.
 */
class RequestScheduler {
public:
    /**
     * `tokenBudget` forwards to the owned ChunkedPrefillScheduler.
     * `maxActiveRequests` bounds the concurrent (Prefilling + Decoding)
     * count — admissions beyond this cap stay in Waiting until an
     * existing request Completes or Preempts.
     *
     * Both parameters MUST be > 0; ctor throws `std::invalid_argument`
     * otherwise.
     *
     * `allocator` is optional. When null (default), the scheduler
     * operates in **state-machine-only mode**: admissions create
     * request entries without paged-KV sequences and `feedPrefillTokens`
     * / `feedDecodeToken` become no-ops. When non-null, the allocator
     * MUST outlive the scheduler; every admitted request gets its own
     * `PagedKvSequence` bound to this allocator, and preemption /
     * completion release the sequence's blocks back to the pool.
     */
    RequestScheduler(std::int32_t            tokenBudget,
                     std::size_t             maxActiveRequests,
                     PagedKvBlockAllocator*  allocator = nullptr);

    RequestScheduler(const RequestScheduler&)            = delete;
    RequestScheduler& operator=(const RequestScheduler&) = delete;
    RequestScheduler(RequestScheduler&&) noexcept        = default;
    RequestScheduler& operator=(RequestScheduler&&) noexcept = default;

    // ---- Admission --------------------------------------------------

    /**
     * Enqueue a new request. Returns its freshly-assigned id
     * (monotonic, starts at 1 — 0 is the "no request" sentinel used
     * by `preemptOne()`). Never throws.
     *
     * `promptLength` and `maxTokens` MUST be > 0; the method returns
     * 0 without side-effects on invalid input (defensive; callers
     * should validate at the HTTP boundary).
     */
    [[nodiscard]] std::uint64_t admit(std::int32_t promptLength,
                                      std::int32_t maxTokens) noexcept;

    // ---- Event-loop step -------------------------------------------

    /**
     * Advance any Waiting → Prefilling under the maxActive cap and
     * build the batch schedule for this iteration. Caller executes
     * the returned schedule then calls `commitProgress()`.
     */
    [[nodiscard]] BatchSchedule tick() noexcept;

    /**
     * Report execution results to advance state. `executed` is the
     * schedule handed out by `tick()`; `reachedEos` is a parallel
     * inline-bitset to `executed.decodes` (same length + same order)
     * where non-zero means the decode produced an EOS token.
     *
     * We deliberately take `std::span<const std::uint8_t>` rather than
     * `std::span<const bool>` — `std::vector<bool>` is the C++ bitset
     * specialisation and cannot round-trip through a bool-span. Using
     * uint8_t lets callers pick whatever container fits.
     *
     * State advances per assignment:
     *   PrefillAssignment{r, chunk} →
     *       r.tokens_pending -= chunk; if 0 → Prefilling → Decoding
     *   DecodeAssignment{r} + reachedEos[i]==0 →
     *       r.tokens_decoded += 1; if == max_tokens → Decoding → Completed
     *   DecodeAssignment{r} + reachedEos[i]!=0 →
     *       Decoding → Completed (regardless of max_tokens)
     *
     * `reachedEos.size()` MUST equal `executed.decodes.size()` — the
     * caller is responsible for that invariant. Mismatched sizes are
     * treated as no-EOS (defensive) so a caller bug degrades to
     * "no requests complete this iter" rather than corrupting state.
     */
    void commitProgress(const BatchSchedule&           executed,
                        std::span<const std::uint8_t>  reachedEos) noexcept;

    // ---- Preemption (mechanism only — C5 policy is separate) -------

    /**
     * Pick the newest active request (LIFO — most recent admit_seq
     * among Prefilling+Decoding) and transition it to Preempted.
     * Returns the request-id, or 0 if no active requests exist.
     *
     * The Preempted entry does NOT auto-re-enqueue synchronously;
     * that happens at the top of the next `tick()`. This split
     * keeps the caller in control of KV-block release ordering
     * (release blocks THEN tick, so freed blocks are available for
     * the newly-promoted requests in the same iteration).
     */
    std::uint64_t preemptOne() noexcept;

    // ---- Cleanup ----------------------------------------------------

    /**
     * Remove all Completed entries and return their ids. Caller uses
     * this to reap the request-registry after the HTTP response has
     * been sent. Ids are returned in admission order. Each drained
     * entry's `kv_sequence` destructor releases its blocks back to
     * the shared pool.
     */
    std::vector<std::uint64_t> drainCompleted() noexcept;

    // ---- KV wire-up (Phase D calls these) --------------------------

    /**
     * Append prefilled prompt tokens to a request's KV sequence.
     * Called by Phase D (`InferenceEngine::generateServing`) after
     * each prefill chunk to keep the block-table + hash-chain in sync
     * with the physical KV-cache writes.
     *
     * Returns true on success; false if:
     *   - the id is unknown or already drained, OR
     *   - the scheduler ctor received no allocator (state-machine
     *     mode — nothing to feed), OR
     *   - any per-token append hit pool exhaustion. **Partial appends
     *     are NOT rolled back** — the caller should preempt the
     *     affected request via `preemptOne()` (or preempt-by-id in a
     *     future policy) to recover the partial KV.
     *
     * Never throws.
     */
    bool feedPrefillTokens(std::uint64_t                  id,
                           std::span<const std::int32_t>  tokens) noexcept;

    /**
     * Append one decoded token. Called by Phase D after each decode
     * forward-pass. Same failure semantics as feedPrefillTokens.
     */
    bool feedDecodeToken(std::uint64_t id, std::int32_t token) noexcept;

    /**
     * Read-only accessor to the allocator this scheduler was
     * constructed with. Returns nullptr when in state-machine-only
     * mode. Useful for `/v1/system/info.serving` metrics that want
     * to surface `block_pool_utilization`.
     */
    [[nodiscard]] const PagedKvBlockAllocator* allocator() const noexcept {
        return _allocator;
    }

    /**
     * One-shot metrics aggregator for the serving-info endpoint /
     * dashboards. Reads all state without mutating; O(N) in the
     * current request-count.
     *
     * Pool-utilisation fields are populated only when the scheduler
     * has an allocator. Monotonic totals reflect the scheduler's
     * lifetime — they are NOT reset by `drainCompleted()`.
     */
    [[nodiscard]] ServingMetrics snapshotMetrics() const noexcept;

    // ---- Inspection (feeds `/v1/system/info.serving` Phase E3) -----

    [[nodiscard]] std::size_t numWaiting()    const noexcept;
    [[nodiscard]] std::size_t numActive()     const noexcept;  // Prefilling + Decoding
    [[nodiscard]] std::size_t numCompleted()  const noexcept;
    [[nodiscard]] std::size_t numPreempted()  const noexcept;

    [[nodiscard]] std::size_t maxActiveRequests() const noexcept {
        return _maxActiveRequests;
    }
    [[nodiscard]] std::int32_t tokenBudget()      const noexcept {
        return _prefillScheduler.tokenBudget();
    }

    /**
     * Read-only lookup by id. Returns `nullptr` if the id is unknown
     * (never admitted, or already drained). Useful for the future
     * Phase D wire-up: after `tick()`, the InferenceEngine needs to
     * know each active request's tokens_decoded / tokens_pending to
     * source query tokens from its per-request buffer.
     */
    [[nodiscard]] const RequestStateData* find(std::uint64_t id) const noexcept;

private:
    /**
     * Build a RequestSlice snapshot of every active (Prefilling +
     * Decoding) request for the ChunkedPrefillScheduler.
     */
    [[nodiscard]] std::vector<RequestSlice> snapshotActiveSlices() const;

    /**
     * Re-enqueue Preempted entries as Waiting. Called at the top of
     * every `tick()` so a request preempted in iteration N is
     * eligible for re-admission (and RECOMPUTE) in iteration N+1.
     */
    void reenqueuePreempted() noexcept;

    /**
     * Promote Waiting requests to Prefilling until either the
     * Waiting queue is empty or the active count reaches
     * `maxActiveRequests`.
     */
    void promoteWaitingToActive() noexcept;

    /**
     * Locate a request by id and return a mutable pointer or
     * nullptr. O(N) linear scan — fine for the maxActive <= 64
     * bound.
     */
    RequestStateData* findMut(std::uint64_t id) noexcept;

    ChunkedPrefillScheduler        _prefillScheduler;
    std::size_t                    _maxActiveRequests;
    PagedKvBlockAllocator*         _allocator{nullptr};
    std::uint64_t                  _nextRequestId{1};
    std::uint64_t                  _nextAdmitSeq{1};
    std::vector<RequestStateData>  _requests;

    // Monotonic counters — never decremented, never reset (persistent
    // across drainCompleted). Surface via snapshotMetrics().
    std::uint64_t                  _totalAdmitted{0};
    std::uint64_t                  _totalCompleted{0};
    std::uint64_t                  _totalPreempted{0};
};

} // namespace mimirmind::runtime::serving