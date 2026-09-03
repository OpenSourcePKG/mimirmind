// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "compute/Sampling.hpp"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace mimirmind::runtime {
class InferenceEngine;
}

namespace mimirmind::runtime::serving {

/**
 * Thrown (via `ServingRequest::overloaded`) when the batcher rejects a submit
 * because the in-flight bound is reached. The HTTP layer maps this to a 503
 * (retryable overload) rather than a 500 (server bug).
 */
struct ServingOverloadedError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

/**
 * Thrown (via `ServingRequest::tenantQuotaExceeded`) when the batcher rejects a
 * submit because the requesting tenant already holds its per-tenant in-flight
 * quota — while the server as a whole may still have spare capacity. Distinct
 * from `ServingOverloadedError` so the HTTP layer can map it to 429 (rate limit
 * scoped to the caller) rather than 503 (whole-server overload).
 */
struct ServingTenantQuotaError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

/**
 * A single in-flight serving request. Shared between the submitting HTTP
 * thread and the batcher worker: the worker appends generated token ids
 * under `mtx` and signals `cv`; the HTTP thread waits on `cv` to block for
 * completion or to stream tokens as they arrive.
 *
 * M-Cuda.Batch D2e.2.
 */
struct ServingRequest {
    std::mutex               mtx;
    std::condition_variable  cv;
    std::vector<std::int32_t> tokens;    // generated ids, appended by worker
    bool                     done{false};
    bool                     cancelled{false};
    bool                     overloaded{false};  // rejected: in-flight bound hit
    bool                     tenantQuotaExceeded{false}; // rejected: per-tenant cap
    std::string              error;      // non-empty => request failed
    // Owning tenant label (API-key tenantId; empty when auth is off). Set at
    // submit() and immutable thereafter, so the worker and the per-tenant
    // admission scan can read it without extra synchronisation beyond `_mtx`.
    std::string              tenantId;

    /// Block until the request completes; returns the full token stream.
    /// On failure the stream may be partial and `error` is set.
    std::vector<std::int32_t> waitAll();

    /// Streaming: wait until token index `next` is available or the request
    /// finishes. Returns true and writes `out` if a token at `next` exists;
    /// returns false when the request is done and no token `next` remains.
    bool waitToken(std::size_t next, std::int32_t& out);
};

/**
 * Continuous-batching serving loop — Sub-Step D2e.2 of M-Cuda.Batch.
 *
 * Owns a single worker thread that drives `InferenceEngine::stepServing`
 * over a fixed set of `maxBatch` physical slots. HTTP threads `submit()`
 * prompts asynchronously; the worker admits waiting prompts into free slots
 * (lowest index first, so the active set stays a tight prefix), steps every
 * active slot ONE token per iteration — at the slot's OWN position, so a
 * freshly admitted prompt prefills at position 0 while others decode at
 * position 500 in the SAME batched forward — and frees a slot the moment its
 * request hits EOS / a stop id / its token budget.
 *
 * v1 uses token-by-token prefill (one prompt token per slot per iteration):
 * correct and genuinely continuous. Chunked multi-token prefill is a later
 * optimization (needs a var-length attention kernel).
 *
 * CUDA + qwen35moe only (the engine's stepServing throws otherwise).
 */
class ContinuousBatcher {
public:
    /// `engine` must outlive the batcher. `maxBatch` physical slots,
    /// `maxContext` tokens (prompt + generated) per slot. `eosId` ends a
    /// request when generated (pass negative to disable the implicit EOS
    /// stop). The serving state is built eagerly here.
    /// `maxInflight` bounds the total accepted-but-unfinished requests
    /// (running slots + waiting queue). Beyond it, `submit()` rejects with
    /// `ServingRequest::overloaded` so the HTTP layer sheds load with a 503
    /// instead of growing an unbounded queue. Clamped up to `maxBatch` so the
    /// bound can never sit below the running capacity.
    /// `maxInflightPerTenant` optionally caps the accepted-but-unfinished
    /// requests a SINGLE tenant (API-key `tenantId`) may hold at once, so no
    /// one caller can monopolise the whole `maxInflight` budget and starve
    /// co-tenants. 0 disables per-tenant limiting (any tenant may use the whole
    /// global budget). Requests with an empty tenant label (auth off) are never
    /// per-tenant limited. A submit over the per-tenant cap is rejected via
    /// `ServingRequest::tenantQuotaExceeded` (HTTP 429), distinct from the
    /// whole-server overload (HTTP 503).
    ContinuousBatcher(InferenceEngine& engine, std::size_t maxBatch,
                      std::size_t maxContext, std::int32_t eosId,
                      std::size_t maxInflight,
                      std::size_t maxInflightPerTenant = 0);
    ~ContinuousBatcher();

    ContinuousBatcher(const ContinuousBatcher&)            = delete;
    ContinuousBatcher& operator=(const ContinuousBatcher&) = delete;
    ContinuousBatcher(ContinuousBatcher&&)                 = delete;
    ContinuousBatcher& operator=(ContinuousBatcher&&)      = delete;

    /// Enqueue a request. `prompt` must be non-empty; `maxNew` caps the
    /// generated tokens; `stopIds` are extra single-token stops (checked in
    /// addition to the ctor eosId). Returns immediately with a handle the
    /// caller waits/streams on. Thread-safe.
    /// `tenantId` is the owning API-key tenant (empty when auth is off); it
    /// feeds the optional per-tenant admission cap and is stored on the returned
    /// handle. A per-tenant-quota rejection sets `tenantQuotaExceeded` on the
    /// handle (mapped to 429) rather than `overloaded` (503).
    /// `sampling` (8.19.5) carries the request's temperature/top_p/top_k/seed
    /// into the slot at admission; the default is greedy argmax, matching the
    /// pre-sampling behaviour.
    std::shared_ptr<ServingRequest> submit(std::vector<std::int32_t> prompt,
                                           std::size_t               maxNew,
                                           std::vector<std::int32_t> stopIds,
                                           std::string               tenantId = {},
                                           compute::SamplingParams   sampling = {});

    /// Request early termination (e.g. the streaming client disconnected).
    /// The worker retires the request at the next iteration and frees its
    /// slot; already-produced tokens stay in the handle. Idempotent.
    void cancel(const std::shared_ptr<ServingRequest>& req);

    [[nodiscard]] std::size_t maxBatch()   const noexcept { return _maxBatch; }
    [[nodiscard]] std::size_t maxContext() const noexcept { return _maxContext; }
    [[nodiscard]] std::size_t maxInflight() const noexcept { return _maxInflight; }
    [[nodiscard]] std::size_t maxInflightPerTenant() const noexcept {
        return _maxInflightPerTenant;
    }

    /// Current accepted-but-unfinished requests (running slots + waiting).
    /// Thread-safe. A cheap pre-admission proxy for the HTTP layer to shed a
    /// streaming request with a clean 503 before it commits to an SSE body;
    /// `submit()` remains the authoritative hard cap.
    [[nodiscard]] std::size_t inflight() const;
    [[nodiscard]] bool        atCapacity() const { return inflight() >= _maxInflight; }

    /// Accepted-but-unfinished requests currently owned by `tenantId` (running
    /// slots + waiting queue). Thread-safe. Returns 0 for an empty tenant label.
    [[nodiscard]] std::size_t inflightForTenant(std::string_view tenantId) const;

    /// Cheap pre-admission proxy mirroring `atCapacity()` but scoped to one
    /// tenant, so the HTTP layer can shed a per-tenant-quota request with a
    /// clean 429 before committing to an SSE body. Always false when per-tenant
    /// limiting is disabled (`_maxInflightPerTenant == 0`) or the tenant label
    /// is empty; `submit()` remains the authoritative hard cap for the race.
    [[nodiscard]] bool atCapacityForTenant(std::string_view tenantId) const;

private:
    struct Pending {
        std::shared_ptr<ServingRequest> req;
        std::vector<std::int32_t>       prompt;
        std::size_t                     maxNew{0};
        std::vector<std::int32_t>       stopIds;
        compute::SamplingParams         sampling{};  // 8.19.5, set at submit()
    };

    struct Slot {
        bool                            occupied{false};
        std::shared_ptr<ServingRequest> req;
        std::vector<std::int32_t>       prompt;
        std::size_t                     promptLen{0};
        std::size_t                     pos{0};       // next input index / position
        std::int32_t                    lastTok{0};
        std::size_t                     maxNew{0};
        std::size_t                     produced{0};  // generated tokens so far
        std::vector<std::int32_t>       stopIds;
        // 5.21-III true mixed step: how many prompt tokens have been prefilled so
        // far (chunk by chunk, interleaved with other slots' decode). == promptLen
        // means the slot is fully prefilled and now decoding. Only used when
        // `_mixedStep` (else prefill is eager and this stays 0).
        std::size_t                     prefillPos{0};
    };

    void workerLoop();
    [[nodiscard]] bool isStop(std::int32_t tok, const Slot& s) const;

    /// Count `tenantId`'s accepted-but-unfinished requests (waiting queue +
    /// occupied slots). Caller MUST hold `_mtx`. Returns 0 for an empty label.
    [[nodiscard]] std::size_t countTenantLocked(std::string_view tenantId) const;

    /// Increment A — prefill a just-admitted slot's whole prompt as chunked
    /// T>1 forwards (see InferenceEngine::prefillSlot), then move it into
    /// decode state at pos == promptLen with its first generated token
    /// pushed. Runs the GPU work without holding `_mtx`. A prefill failure
    /// fails only that request. Only called when `_prefillChunk > 0`.
    void prefillSlotAdmitted(std::size_t slot);

    /// 5.21-III MULTI-SLOT — prefill a CONTIGUOUS run of just-admitted slots
    /// (`run` sorted ascending, run[k]==run[0]+k) by batching their prompts
    /// into ragged `prefillSlotsBatched` forwards (packed up to `_prefillMaxRows`
    /// tokens/forward), then committing each into decode. A slot whose prompt
    /// exceeds the per-forward budget falls back to single-slot chunked prefill.
    /// Runs the GPU work without holding `_mtx`. Only called when
    /// `_mixedBatchPrefill` is on. A prefill failure fails only the involved
    /// requests, leaving the rest of the batch's slots retired cleanly.
    void prefillSlotRunBatched(std::span<const std::size_t> run);

    /// Commit a freshly-prefilled slot into decode state at pos == promptLen
    /// with its first generated token `firstTok` pushed (stop-condition checks
    /// included). Caller must NOT hold `_mtx`. Shared by the single-slot and
    /// multi-slot prefill paths.
    void commitPrefilledSlot(std::size_t slot,
                             const std::shared_ptr<ServingRequest>& req,
                             std::int32_t firstTok);

    /// 5.21-III TRUE MIXED STEP — one ragged forward over the active prefix
    /// [0,nActive) folding each slot's next PREFILL chunk (seqT>1) OR its DECODE
    /// token (seqT=1) into the same `runVarlenPrefill` call (vLLM-style prefill
    /// riding along with decode). Advances prefill positions, distributes decode
    /// tokens, retires finished requests. Runs the GPU forward without `_mtx`.
    /// Returns true if a slot was freed (submitters should be woken). Only called
    /// when `_mixedStep` is on and at least one slot is still prefilling.
    bool runMixedStep();

    InferenceEngine& _engine;
    std::size_t      _maxBatch;
    std::size_t      _maxContext;
    std::int32_t     _eosId;
    std::size_t      _maxInflight;
    // Per-tenant accepted-but-unfinished cap; 0 => per-tenant limiting off.
    std::size_t      _maxInflightPerTenant;
    // Chunked-prefill chunk size C (InferenceEngine::servingPrefillChunk());
    // 0 => disabled, prompts ingest token-by-token via the decode loop.
    std::size_t      _prefillChunk{0};
    // 5.21 Tier 1 (prefill/decode interleave): cap how many waiting requests are
    // admitted+prefilled per worker iteration. Each admitted slot is prefilled
    // to completion (single-slot, batch=1) before the iteration's decode step,
    // so a burst of N concurrent prompts would otherwise prefill all N (serial,
    // batch=1) before any decode runs -> in-flight decode stalls for the whole
    // burst. Capping admission to K per iteration interleaves a batched decode
    // step between prefills so in-flight sequences keep decoding while new ones
    // prefill. 0 => unlimited (legacy: admit all free slots at once).
    // MIMIRMIND_PREFILL_ADMIT_PER_ITER=<K>. Safe (no partial slot ever enters
    // the decode batch); overlap only — does NOT batch the prefills themselves.
    std::size_t      _admitPerIter{0};
    // 5.21-III MULTI-SLOT — batch newly-admitted slots' prefill into ONE ragged
    // forward (unlike admit-per-iter, this batches the prefill compute itself).
    // MIMIRMIND_PREFILL_MIXED_BATCH=1; default OFF. Needs chunked prefill.
    bool             _mixedBatchPrefill{false};
    // Per-mixed-forward token budget (InferenceEngine::servingPrefillMaxRows()).
    std::size_t      _prefillMaxRows{0};
    // 5.21-III TRUE MIXED STEP — fold each slot's prefill chunk OR decode token
    // into ONE ragged forward per iteration (prefill overlaps decode, vLLM-style).
    // Supersedes eager prefill: newly-admitted slots start prefilling in-band.
    // DEFAULT ON (server-decides, no user toggle): Spark A/B 2026-08-24 validated
    // green across burst/trickle/decode-heavy — +19% agg / -16% wall / -43% ttft
    // under load, and bit-identical pure-decode (inert once nothing prefills →
    // decode-v2 unchanged). Auto-disabled when the substrate has no chunked prefill
    // (prefillMaxRows==0). Rollback: MIMIRMIND_MIXED_STEP=0.
    //
    // 2026-08-24 REVERTED TO DEFAULT-OFF pending investigation: under real prod
    // load (full serve set, mixed multi-model traffic) mixedStep=1 triggered a
    // cudaStreamSynchronize illegal memory access that poisoned the shared CUDA
    // context -> garbage output from OTHER co-resident models (gemma4). The
    // isolated A/B sweeps (single model, controlled load) never hit it. Keep the
    // feature env-opt-in (MIMIRMIND_MIXED_STEP=1) until the illegal access is
    // root-caused and fixed; prod must run the known-good decode-v2 scheduler.
    bool             _mixedStep{false};
    // 5.21-III PRESSURE GATE — only fold prefill into the decode forward when the
    // wait queue exceeds free-slot capacity (a real backlog). With no backlog,
    // newly-admitted slots take the eager chunked-prefill path so live decoders
    // keep the fast decode-v2 step. Default OFF: the 2026-08-24 Spark A/B showed
    // unconditional folding wins e2e (wall/agg/ttft) at every tested workload —
    // at conc<=maxBatch the gate reverts to prod and DROPS the +19%/-43% win,
    // while only recovering the decode_tok_s_est ACCOUNTING artifact. Kept as an
    // opt-in (MIMIRMIND_MIXED_STEP_PRESSURE=1) for the untested trickle-arrival
    // regime (steady decode pool + occasional new prefill), which the burst
    // sweep does not exercise.
    bool             _mixedStepPressureGate{false};

    std::vector<Slot>                     _slots;
    std::deque<Pending>                   _waiting;
    mutable std::mutex                    _mtx;
    std::condition_variable               _cv;      // wakes worker on submit/stop
    std::thread                           _worker;
    bool                                  _running{false};
};

} // namespace mimirmind::runtime::serving
