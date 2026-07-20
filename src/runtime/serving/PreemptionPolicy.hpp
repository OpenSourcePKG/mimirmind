// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstddef>

namespace mimirmind::runtime::serving {

/**
 * Advisory policy that tells the M-Cuda.Batch serving loop **when to
 * preempt**. Sub-Step C5 of the milestone. Pairs with
 * `RequestScheduler::preemptOne()` (mechanism) — this class is the
 * *policy* that decides whether to call it.
 *
 * ---- Design ------------------------------------------------------
 *
 * Reactive threshold on the paged-KV block-pool free-ratio. The
 * scheduler loop calls `shouldPreempt(...)` after each `tick()` /
 * before launching the batch; if true, it calls `preemptOne()`,
 * then re-checks. That per-tick single-preemption loop naturally
 * bounds thrashing without needing time-based hysteresis: one
 * preempted request frees blocks, next tick re-evaluates, and if
 * still below threshold, preempts again.
 *
 * The policy is **stateless** — same input → same output. Real
 * threshold tuning happens once we have Spark load-test numbers.
 * Bragi-v1 default is 5 % free (`kDefaultFreeBlockThreshold`), a
 * conservative starting point matching vLLM's `gpu_memory_utilization`
 * fallback heuristic.
 *
 * ---- Sole-active guard -------------------------------------------
 *
 * Preemption requires `numActive > 1`. If only one request is
 * active, preempting it makes zero forward progress — the caller
 * would evict the KV, re-admit the same request next tick, re-fill
 * KV, and hit the same block-shortage. This is the classic
 * livelock case; the guard prevents it. When `numActive == 1` and
 * the pool is exhausted, the correct behaviour is to fail the
 * request with OOM, not to thrash — that failure path lives in
 * the caller (Phase D wire-up), not here.
 *
 * ---- Non-goals in C5 (v1) ---------------------------------------
 *
 * - **Predictive preemption.** No look-ahead across future waiting
 *   requests. Reactive threshold only.
 * - **SLA / priority-aware victim choice.** `preemptOne()` picks
 *   LIFO (least KV invested). Priority-based victim selection is a
 *   follow-up if fairness ever becomes an issue.
 * - **Time-based hysteresis.** The single-preemption-per-tick loop
 *   is the hysteresis mechanism. Adding time-window smoothing would
 *   require state, which we deliberately avoid.
 * - **Multi-preempt-per-tick.** Bounded to 1 for v1 to keep the
 *   feedback loop tight and observable in the perf-ledger.
 *
 * ---- Threading --------------------------------------------------
 *
 * `shouldPreempt` is const, pure, and reads no shared state — safe
 * to call concurrently on disjoint inputs. In practice the
 * scheduler owns one instance called serially on its event-loop
 * thread.
 */
class PreemptionPolicy {
public:
    /**
     * vLLM's `gpu_memory_utilization=0.95` maps to 5 % headroom
     * before pressure — this is the equivalent "start pushing back"
     * point for the paged block-pool.
     */
    static constexpr double kDefaultFreeBlockThreshold = 0.05;

    /**
     * `freeBlockThreshold` MUST be in `[0.0, 1.0]`. Ctor throws
     * `std::invalid_argument` outside that range.
     *
     * `0.0` effectively disables the policy — `freeRatio < 0.0` is
     * impossible so `shouldPreempt` always returns false. Useful for
     * benchmarks or single-tenant modes where preemption is
     * undesirable.
     *
     * `1.0` triggers preemption whenever any block is used — nonsense
     * in Prod but accepted for stress-tests.
     */
    explicit PreemptionPolicy(double freeBlockThreshold = kDefaultFreeBlockThreshold);

    [[nodiscard]] double freeBlockThreshold() const noexcept { return _threshold; }

    /**
     * Advisory: should the caller invoke `RequestScheduler::preemptOne()`?
     *
     * Returns true when ALL of:
     *   - `totalBlocks > 0` (defensive)
     *   - `numActive > 1` (sole-active guard — see class doc)
     *   - `freeBlocks / totalBlocks < freeBlockThreshold`
     *
     * Otherwise false.
     */
    [[nodiscard]] bool shouldPreempt(std::size_t freeBlocks,
                                     std::size_t totalBlocks,
                                     std::size_t numActive) const noexcept;

private:
    double _threshold;
};

} // namespace mimirmind::runtime::serving