// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/serving/ContinuousBatcher.hpp"

#include "runtime/InferenceEngine.hpp"
#include "core/log/Log.hpp"

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace mimirmind::runtime::serving {

std::vector<std::int32_t> ServingRequest::waitAll() {
    std::unique_lock<std::mutex> lk(mtx);
    cv.wait(lk, [this] { return done; });
    return tokens;
}

bool ServingRequest::waitToken(std::size_t next, std::int32_t& out) {
    std::unique_lock<std::mutex> lk(mtx);
    cv.wait(lk, [this, next] { return tokens.size() > next || done; });
    if (tokens.size() > next) {
        out = tokens[next];
        return true;
    }
    return false;  // done and no further token
}

ContinuousBatcher::ContinuousBatcher(InferenceEngine& engine,
                                     std::size_t maxBatch,
                                     std::size_t maxContext,
                                     std::int32_t eosId,
                                     std::size_t maxInflight,
                                     std::size_t maxInflightPerTenant)
    : _engine(engine),
      _maxBatch(maxBatch),
      _maxContext(maxContext),
      _eosId(eosId),
      // Never let the admission bound sit below the running capacity — that
      // would reject requests that could start immediately. A zero config
      // value means "no extra queue", i.e. exactly maxBatch in flight.
      _maxInflight(std::max(maxInflight, maxBatch)),
      _maxInflightPerTenant(maxInflightPerTenant) {
    if (maxBatch == 0 || maxContext == 0) {
        throw std::runtime_error("ContinuousBatcher: maxBatch/maxContext > 0");
    }
    // Build the persistent per-slot serving state up front (qwen35moe paged
    // pool or, on L0/Xe-LPG, the Gemma 4 MoE slab substrate; throws for a
    // backend that supports neither, before any request is accepted).
    _engine.ensureServingState(maxBatch, maxContext);
    _prefillChunk = _engine.servingPrefillChunk();
    if (const char* ap = std::getenv("MIMIRMIND_PREFILL_ADMIT_PER_ITER")) {
        const long v = std::strtol(ap, nullptr, 10);
        if (v > 0) _admitPerIter = static_cast<std::size_t>(v);
    }
    // 5.21-III MULTI-SLOT — batch newly-admitted slots' prefill into ONE ragged
    // forward (the 6.6x continuous-prefill-batching wall-fix) instead of one
    // batch=1 forward per slot. Default OFF (env opt-in) while validating.
    if (const char* mb = std::getenv("MIMIRMIND_PREFILL_MIXED_BATCH")) {
        _mixedBatchPrefill = (mb[0] == '1' && mb[1] == '\0');
    }
    _prefillMaxRows = _engine.servingPrefillMaxRows();
    if (_mixedBatchPrefill && _prefillMaxRows == 0) {
        _mixedBatchPrefill = false;   // chunked prefill disabled -> no mixed path
    }
    // 5.21-III TRUE MIXED STEP — supersedes eager batched prefill: prefill chunks
    // fold into the same forward as decode. Requires chunked prefill (nRowMax>0).
    // DEFAULT ON (validated green on Spark 2026-08-24); MIMIRMIND_MIXED_STEP=0
    // rolls back to the pre-overlap eager/decode scheduler.
    if (const char* ms = std::getenv("MIMIRMIND_MIXED_STEP")) {
        _mixedStep = (ms[0] == '1' && ms[1] == '\0');
    }
    if (_mixedStep && _prefillMaxRows == 0) {
        _mixedStep = false;
    }
    // 5.21-III PRESSURE GATE — the mixed step folds decode tokens into the heavy
    // (MoE-GEMM-dominated) prefill forward, which costs decode throughput when
    // there is no prefill backlog to overlap with. Gate the in-band strategy on
    // real pressure: only fold when the wait queue exceeds free-slot capacity;
    // otherwise admit via eager chunked prefill (dedicated forward) so running
    // decoders stay on the fast decode-v2 path. Default OFF — the Spark A/B found
    // unconditional folding wins e2e everywhere tested; the gate only recovers an
    // accounting metric while dropping the win at conc<=maxBatch. Opt-in via
    // MIMIRMIND_MIXED_STEP_PRESSURE=1 for the untested trickle-arrival regime.
    if (const char* pg = std::getenv("MIMIRMIND_MIXED_STEP_PRESSURE")) {
        _mixedStepPressureGate = (pg[0] == '1' && pg[1] == '\0');
    }
    _slots.resize(maxBatch);
    _running = true;
    _worker  = std::thread(&ContinuousBatcher::workerLoop, this);
    MM_LOG_INFO("serving",
                "ContinuousBatcher: started (maxBatch={} maxContext={} "
                "maxInflight={} maxInflightPerTenant={} eosId={} prefillChunk={} "
                "mixedBatchPrefill={} mixedStep={} mixedStepPressureGate={} "
                "prefillMaxRows={})",
                maxBatch, maxContext, _maxInflight,
                _maxInflightPerTenant, eosId, _prefillChunk,
                _mixedBatchPrefill ? 1 : 0, _mixedStep ? 1 : 0,
                (_mixedStep && _mixedStepPressureGate) ? 1 : 0, _prefillMaxRows);
}

ContinuousBatcher::~ContinuousBatcher() {
    {
        std::lock_guard<std::mutex> lk(_mtx);
        _running = false;
    }
    _cv.notify_all();
    if (_worker.joinable()) {
        _worker.join();
    }
}

bool ContinuousBatcher::isStop(std::int32_t tok, const Slot& s) const {
    return std::find(s.stopIds.begin(), s.stopIds.end(), tok) != s.stopIds.end();
}

std::shared_ptr<ServingRequest> ContinuousBatcher::submit(
        std::vector<std::int32_t> prompt, std::size_t maxNew,
        std::vector<std::int32_t> stopIds, std::string tenantId) {
    auto req = std::make_shared<ServingRequest>();
    req->tenantId = std::move(tenantId);
    if (prompt.empty()) {
        req->error = "empty prompt";
        req->done  = true;
        return req;
    }
    if (prompt.size() >= _maxContext) {
        req->error = "prompt exceeds serving maxContext";
        req->done  = true;
        return req;
    }
    {
        std::lock_guard<std::mutex> lk(_mtx);
        if (!_running) {
            req->error     = "batcher stopped";
            req->done      = true;
            req->cancelled = true;
            return req;
        }
        // Bounded admission — the authoritative hard cap. Count running slots
        // plus the waiting queue; beyond `_maxInflight` shed load instead of
        // growing an unbounded queue (which would balloon memory and push tail
        // latency to minutes under a spike). The HTTP layer turns `overloaded`
        // into a 503 + Retry-After.
        std::size_t inFlight = _waiting.size();
        for (const auto& s : _slots) {
            if (s.occupied) ++inFlight;
        }
        if (inFlight >= _maxInflight) {
            req->error      = "server overloaded: request queue full";
            req->overloaded = true;
            req->done       = true;
            MM_LOG_WARN("serving",
                        "ContinuousBatcher: rejecting submit — {} in flight >= "
                        "maxInflight {} (load shedding)",
                        inFlight, _maxInflight);
            return req;
        }
        // Per-tenant fairness — the authoritative hard cap for one caller.
        // Reject before the whole server is full so a single tenant flooding
        // requests cannot starve co-tenants of the shared slot budget. Scoped
        // by tenantId (empty label / disabled cap => never limited). This is a
        // 429 (your quota), not a 503 (server overload) — a distinct handle
        // flag so the HTTP layer can signal the two cases apart.
        if (_maxInflightPerTenant > 0 && !req->tenantId.empty()) {
            const std::size_t tenantInFlight = countTenantLocked(req->tenantId);
            if (tenantInFlight >= _maxInflightPerTenant) {
                req->error               = "tenant quota exceeded: too many "
                                           "concurrent requests for this key";
                req->tenantQuotaExceeded = true;
                req->done                = true;
                MM_LOG_WARN("serving",
                            "ContinuousBatcher: rejecting submit — tenant '{}' "
                            "has {} in flight >= maxInflightPerTenant {} "
                            "(per-tenant load shedding)",
                            req->tenantId, tenantInFlight,
                            _maxInflightPerTenant);
                return req;
            }
        }
        Pending p;
        p.req     = req;
        p.prompt  = std::move(prompt);
        p.maxNew  = maxNew;
        p.stopIds = std::move(stopIds);
        _waiting.push_back(std::move(p));
    }
    _cv.notify_all();
    return req;
}

std::size_t ContinuousBatcher::inflight() const {
    std::lock_guard<std::mutex> lk(_mtx);
    std::size_t n = _waiting.size();
    for (const auto& s : _slots) {
        if (s.occupied) ++n;
    }
    return n;
}

std::size_t ContinuousBatcher::countTenantLocked(
        std::string_view tenantId) const {
    // Derive per-tenant occupancy from live state rather than a maintained
    // counter: the same tenant label rides on every waiting Pending and every
    // occupied Slot via its ServingRequest, and requests leave flight from six
    // different sites (prefill-stop, decode-stop, cancel, step-fail, shutdown,
    // reject). A scan here mirrors the global inflight() and cannot drift out
    // of sync the way a hand-decremented counter across those sites would. n is
    // bounded by maxInflight (tens), so the O(n) scan is trivial off the
    // once-per-request submit path.
    if (tenantId.empty()) {
        return 0;
    }
    std::size_t n = 0;
    for (const auto& p : _waiting) {
        if (p.req && p.req->tenantId == tenantId) ++n;
    }
    for (const auto& s : _slots) {
        if (s.occupied && s.req && s.req->tenantId == tenantId) ++n;
    }
    return n;
}

std::size_t ContinuousBatcher::inflightForTenant(
        std::string_view tenantId) const {
    std::lock_guard<std::mutex> lk(_mtx);
    return countTenantLocked(tenantId);
}

bool ContinuousBatcher::atCapacityForTenant(std::string_view tenantId) const {
    if (_maxInflightPerTenant == 0 || tenantId.empty()) {
        return false;
    }
    return inflightForTenant(tenantId) >= _maxInflightPerTenant;
}

void ContinuousBatcher::cancel(const std::shared_ptr<ServingRequest>& req) {
    if (!req) return;
    std::lock_guard<std::mutex> rl(req->mtx);
    req->cancelled = true;
    // The worker checks `cancelled` each iteration; wake it in case it is
    // idle-waiting (no active slots) so a still-queued request is retired.
    _cv.notify_all();
}

void ContinuousBatcher::prefillSlotAdmitted(std::size_t slot) {
    // Snapshot the prompt + request handle under a short lock; the GPU prefill
    // runs without holding _mtx so submitters/streamers never block on it.
    std::vector<std::int32_t>       prompt;
    std::shared_ptr<ServingRequest> req;
    {
        std::lock_guard<std::mutex> lk(_mtx);
        if (!_slots[slot].occupied) {
            return;   // cancelled/retired before we got here
        }
        prompt = _slots[slot].prompt;   // copy: read outside the lock
        req    = _slots[slot].req;
    }
    const std::size_t L = prompt.size();
    if (L == 0) {
        return;
    }

    // Ingest the prompt as chunked T>1 forwards; the final chunk yields the
    // first generated token (identical to the token-by-token step at pos ==
    // promptLen-1).
    std::int32_t firstTok = -1;
    try {
        const std::size_t C = _prefillChunk;
        for (std::size_t p = 0; p < L; p += C) {
            const std::size_t t    = std::min(C, L - p);
            const bool        last = (p + t == L);
            const std::span<const std::int32_t> chunk{prompt.data() + p, t};
            const std::int32_t tk = _engine.prefillSlot(slot, chunk, p, last);
            if (last) {
                firstTok = tk;
            }
        }
    } catch (const std::exception& ex) {
        std::lock_guard<std::mutex> lk(_mtx);
        Slot& s = _slots[slot];
        if (s.occupied && s.req == req) {
            {
                std::lock_guard<std::mutex> rl(s.req->mtx);
                s.req->error = std::string("prefill failed: ") + ex.what();
                s.req->done  = true;
                s.req->cv.notify_all();
            }
            s.occupied = false;
            s.req.reset();
            s.prompt.clear();
        }
        return;
    }

    commitPrefilledSlot(slot, req, firstTok);
}

void ContinuousBatcher::commitPrefilledSlot(
        std::size_t slot, const std::shared_ptr<ServingRequest>& req,
        std::int32_t firstTok) {
    // Commit: the slot enters decode at pos == promptLen with firstTok already
    // produced. (Only the worker thread admits/frees slots, so the slot cannot
    // have been re-used mid-prefill; the guards are defensive.)
    std::lock_guard<std::mutex> lk(_mtx);
    Slot& s = _slots[slot];
    if (!s.occupied || s.req != req) {
        return;
    }
    s.pos      = s.promptLen;
    // Eager prefill drives the slot through the `pos` state machine, not the
    // in-band `prefillPos` one. Mark the whole prompt as ingested so the true
    // mixed step (runMixedStep) treats this slot as decode-only and never
    // re-prefills it — the two prefill strategies are chosen per-slot at
    // admission (pressure gate) and must not both fire for one slot.
    s.prefillPos = s.promptLen;
    s.lastTok  = firstTok;
    s.produced = 1;

    bool cancelled = false;
    {
        std::lock_guard<std::mutex> rl(s.req->mtx);
        cancelled = s.req->cancelled;
        if (!cancelled) {
            s.req->tokens.push_back(firstTok);
            s.req->cv.notify_all();
        }
    }
    const bool ctxFull = (s.pos + 1 >= _maxContext);
    const bool stop =
        cancelled || (_eosId >= 0 && firstTok == _eosId) || isStop(firstTok, s) ||
        s.produced >= s.maxNew || ctxFull;
    if (stop) {
        {
            std::lock_guard<std::mutex> rl(s.req->mtx);
            s.req->done = true;
            s.req->cv.notify_all();
        }
        s.occupied = false;
        s.req.reset();
        s.prompt.clear();
    }
}

void ContinuousBatcher::prefillSlotRunBatched(std::span<const std::size_t> run) {
    // Snapshot each slot's prompt + request under a short lock; GPU prefill runs
    // without _mtx. Slots in `run` are contiguous & ascending (run[k]==run[0]+k)
    // — required so runVarlenPrefill's state/block-table slices address as
    // base + firstSlot*stride + seq*stride.
    const std::size_t R = run.size();
    if (R == 0) {
        return;
    }
    std::vector<std::vector<std::int32_t>>       prompts(R);
    std::vector<std::shared_ptr<ServingRequest>> reqs(R);
    {
        std::lock_guard<std::mutex> lk(_mtx);
        for (std::size_t k = 0; k < R; ++k) {
            const std::size_t slot = run[k];
            if (_slots[slot].occupied) {
                prompts[k] = _slots[slot].prompt;   // copy: read outside the lock
                reqs[k]    = _slots[slot].req;
            }
        }
    }

    // Fail one slot's request with an error message (mirrors the single-slot
    // catch path). Caller holds no lock.
    auto failSlot = [this](std::size_t slot,
                           const std::shared_ptr<ServingRequest>& req,
                           const std::string& msg) {
        std::lock_guard<std::mutex> lk(_mtx);
        Slot& s = _slots[slot];
        if (s.occupied && s.req == req) {
            {
                std::lock_guard<std::mutex> rl(s.req->mtx);
                s.req->error = msg;
                s.req->done  = true;
                s.req->cv.notify_all();
            }
            s.occupied = false;
            s.req.reset();
            s.prompt.clear();
        }
    };

    // Greedily pack a batch of consecutive slots whose prompts each fit the
    // per-forward budget and whose summed tokens stay within _prefillMaxRows,
    // then prefill the whole batch in ONE ragged forward. A single prompt larger
    // than the budget can't be one-shot batched → prefill it alone via the
    // chunked single-slot path. (Sequential prompt ingestion within a slot is
    // still one forward per chunk; the win is batching ACROSS slots.)
    std::size_t k = 0;
    while (k < R) {
        // Skip slots that vanished before we snapshotted them.
        if (reqs[k] == nullptr || prompts[k].empty()) { ++k; continue; }

        const std::size_t Lk = prompts[k].size();
        if (Lk > _prefillMaxRows) {
            // Oversized prompt: fall back to the single-slot chunked prefill,
            // which internally chunks to _prefillChunk-sized forwards.
            prefillSlotAdmitted(run[k]);
            ++k;
            continue;
        }

        // Grow a contiguous batch [k, j) within the token budget.
        std::size_t j = k, tokBudget = 0;
        std::vector<std::span<const std::int32_t>> chunks;
        std::vector<std::size_t>                   startPos;
        while (j < R && reqs[j] != nullptr && !prompts[j].empty()) {
            const std::size_t Lj = prompts[j].size();
            if (Lj > _prefillMaxRows) break;                 // handled alone above
            if (tokBudget + Lj > _prefillMaxRows) break;     // budget full
            chunks.emplace_back(prompts[j].data(), Lj);
            startPos.push_back(0);                            // whole prompt, pos 0
            tokBudget += Lj;
            ++j;
        }

        const std::size_t N = chunks.size();
        std::vector<std::int32_t> firstTok(N, -1);
        bool ok = true;
        try {
            _engine.prefillSlotsBatched(run[k], chunks, startPos,
                                        /*produceToken=*/true, firstTok);
        } catch (const std::exception& ex) {
            ok = false;
            const std::string msg = std::string("prefill failed: ") + ex.what();
            for (std::size_t b = 0; b < N; ++b) failSlot(run[k + b], reqs[k + b], msg);
        }
        if (ok) {
            for (std::size_t b = 0; b < N; ++b) {
                commitPrefilledSlot(run[k + b], reqs[k + b], firstTok[b]);
            }
        }
        k = j;
    }
}

bool ContinuousBatcher::runMixedStep() {
    // ONE ragged forward over the active prefix [0,nActive) that folds each
    // slot's next PREFILL chunk (seqT>1, startPos=prefillPos) OR its DECODE token
    // (seqT=1, startPos=pos). runVarlenPrefill derives per-slot rope/KV/state and
    // returns each slot's last-row greedy token; we use it for decode slots and
    // for prefill slots that just completed their prompt, and discard it for
    // still-prefilling slots.

    // Retire-with-error helper (inline; runMixedStep is off the worker's `finish`).
    auto failReq = [](const std::shared_ptr<ServingRequest>& req, std::string err) {
        std::lock_guard<std::mutex> rl(req->mtx);
        if (!err.empty()) req->error = std::move(err);
        req->done = true;
        req->cv.notify_all();
    };

    std::size_t nActive = 0;
    std::vector<std::vector<std::int32_t>>       chunkStore;   // stable token storage
    std::vector<std::span<const std::int32_t>>   chunks;
    std::vector<std::size_t>                     startPos;
    std::vector<std::shared_ptr<ServingRequest>> reqs;
    std::vector<char> prefilling, finishesPrefill;             // per slot flags
    std::vector<std::size_t> newPrefillPos;

    {
        std::lock_guard<std::mutex> lk(_mtx);
        for (std::size_t i = 0; i < _maxBatch; ++i) {
            if (_slots[i].occupied) nActive = i + 1;
        }
        if (nActive == 0) {
            return false;
        }
        chunkStore.resize(nActive);
        chunks.resize(nActive);
        startPos.assign(nActive, 0);
        reqs.resize(nActive);
        prefilling.assign(nActive, 0);
        finishesPrefill.assign(nActive, 0);
        newPrefillPos.assign(nActive, 0);

        // Reserve 1 row per active slot; hand the remaining budget to prefill
        // chunks in slot order. Guarantees every slot contributes >=1 row and the
        // summed tokens never exceed _prefillMaxRows.
        std::size_t extraBudget =
            (_prefillMaxRows > nActive) ? _prefillMaxRows - nActive : 0;

        for (std::size_t i = 0; i < nActive; ++i) {
            Slot& s = _slots[i];
            if (!s.occupied) {
                chunkStore[i] = {0};   // hole: dummy 1-token row, result discarded
                chunks[i]     = std::span<const std::int32_t>(chunkStore[i]);
                continue;
            }
            reqs[i] = s.req;
            if (s.prefillPos < s.promptLen) {
                const std::size_t remain = s.promptLen - s.prefillPos;
                const std::size_t want   = std::min(_prefillChunk, remain);   // >=1
                const std::size_t take   = std::min(want - 1, extraBudget);
                extraBudget -= take;
                const std::size_t c = 1 + take;
                chunkStore[i].assign(s.prompt.begin() + s.prefillPos,
                                     s.prompt.begin() + s.prefillPos + c);
                chunks[i]          = std::span<const std::int32_t>(chunkStore[i]);
                startPos[i]        = s.prefillPos;
                prefilling[i]      = 1;
                newPrefillPos[i]   = s.prefillPos + c;
                finishesPrefill[i] = (newPrefillPos[i] == s.promptLen) ? 1 : 0;
            } else {
                chunkStore[i] = { s.lastTok };   // decode: single token at pos
                chunks[i]     = std::span<const std::int32_t>(chunkStore[i]);
                startPos[i]   = s.pos;
            }
        }
    }

    std::vector<std::int32_t> outTok(nActive, -1);
    try {
        _engine.prefillSlotsBatched(/*firstSlot=*/0, chunks, startPos,
                                    /*produceToken=*/true, outTok);
    } catch (const std::exception& ex) {
        // A forward failure poisons the whole batch — fail every active request.
        std::lock_guard<std::mutex> lk(_mtx);
        for (std::size_t i = 0; i < nActive; ++i) {
            if (_slots[i].occupied) {
                failReq(_slots[i].req, std::string("mixed step failed: ") + ex.what());
                _slots[i].occupied = false;
                _slots[i].req.reset();
                _slots[i].prompt.clear();
            }
        }
        return true;
    }

    bool freedSlot = false;
    {
        std::lock_guard<std::mutex> lk(_mtx);
        for (std::size_t i = 0; i < nActive; ++i) {
            Slot& s = _slots[i];
            if (!s.occupied || s.req != reqs[i]) {
                continue;   // hole, or retired/reused since snapshot
            }

            bool cancelled = false;
            {
                std::lock_guard<std::mutex> rl(s.req->mtx);
                cancelled = s.req->cancelled;
            }
            if (cancelled) {
                failReq(s.req, "");
                s.occupied = false; s.req.reset(); s.prompt.clear();
                freedSlot = true;
                continue;
            }

            bool decodeAdvance = false;   // decode slots ++pos after emitting
            if (prefilling[i]) {
                s.prefillPos = newPrefillPos[i];
                if (!finishesPrefill[i]) {
                    continue;             // still prefilling — no token this step
                }
                s.pos = s.promptLen;      // first generated token lands at promptLen
            } else {
                decodeAdvance = true;
            }

            const std::int32_t t = outTok[i];
            {
                std::lock_guard<std::mutex> rl(s.req->mtx);
                s.req->tokens.push_back(t);
                s.req->cv.notify_all();
            }
            s.lastTok = t;
            ++s.produced;
            // ctxFull: decode feeds the next token at pos+1 (needs +2 headroom);
            // a just-finished prefill feeds it at promptLen (needs +1).
            const bool ctxFull = decodeAdvance ? (s.pos + 2 >= _maxContext)
                                               : (s.pos + 1 >= _maxContext);
            const bool stop =
                (_eosId >= 0 && t == _eosId) || isStop(t, s) ||
                s.produced >= s.maxNew || ctxFull;
            if (stop) {
                {
                    std::lock_guard<std::mutex> rl(s.req->mtx);
                    s.req->done = true;
                    s.req->cv.notify_all();
                }
                s.occupied = false; s.req.reset(); s.prompt.clear();
                freedSlot = true;
                continue;
            }
            if (decodeAdvance) {
                ++s.pos;
            }
        }
    }
    return freedSlot;
}

void ContinuousBatcher::workerLoop() {
    using Step = InferenceEngine::ServingSlotStep;

    // Fail-and-drain helper: marks a request done, optionally with an error.
    auto finish = [](const std::shared_ptr<ServingRequest>& req,
                     std::string err) {
        std::lock_guard<std::mutex> rl(req->mtx);
        if (!err.empty()) req->error = std::move(err);
        req->done = true;
        req->cv.notify_all();
    };

    std::vector<Step>         steps;
    std::vector<std::int32_t> toks;

    while (true) {
        std::size_t nActive = 0;
        std::vector<std::size_t> toPrefill;   // slots admitted this iter (chunked)
        {
            std::unique_lock<std::mutex> lk(_mtx);

            if (!_running) {
                // Shutdown: cancel everything still in flight.
                for (auto& p : _waiting) {
                    p.req->cancelled = true;
                    finish(p.req, "batcher stopped");
                }
                _waiting.clear();
                for (auto& s : _slots) {
                    if (s.occupied) {
                        s.req->cancelled = true;
                        finish(s.req, "batcher stopped");
                        s.occupied = false;
                        s.req.reset();
                    }
                }
                return;
            }

            // 5.21-III PRESSURE GATE — decide this iteration's prefill strategy
            // for the true mixed step. Pressure = the wait queue still exceeds
            // free-slot capacity (a real backlog worth overlapping). Under
            // pressure, admit slots for in-band mixed prefill (folded with
            // decode); with no backlog, admit them for eager chunked prefill so
            // running decoders keep the fast decode-v2 path. The choice is fixed
            // per slot at admission — an in-flight prefill always finishes on the
            // strategy it started with (the `pos` vs `prefillPos` state machines
            // are not interchangeable mid-prompt).
            bool prefillPressure = true;
            if (_mixedStep && _mixedStepPressureGate) {
                std::size_t nFree = 0;
                for (std::size_t i = 0; i < _maxBatch; ++i) {
                    if (!_slots[i].occupied) ++nFree;
                }
                prefillPressure = (_waiting.size() > nFree);
            }

            // Admit waiting requests into free slots (lowest index first so
            // the active set stays a tight prefix). 5.21 Tier 1: cap admits per
            // iteration so a burst does not prefill all slots (serial, batch=1)
            // before any decode step runs — interleaves decode with prefill.
            std::size_t admitted = 0;
            for (std::size_t i = 0; i < _maxBatch && !_waiting.empty(); ++i) {
                if (_admitPerIter != 0 && admitted >= _admitPerIter) break;
                if (_slots[i].occupied) continue;
                Pending p = std::move(_waiting.front());
                _waiting.pop_front();
                Slot& s     = _slots[i];
                s.occupied  = true;
                s.req       = std::move(p.req);
                s.prompt    = std::move(p.prompt);
                s.promptLen = s.prompt.size();
                s.pos       = 0;
                s.lastTok   = 0;
                s.maxNew    = p.maxNew;
                s.produced  = 0;
                s.stopIds   = std::move(p.stopIds);
                s.prefillPos = 0;   // 5.21-III mixed step: start prefilling in-band
                // Chunked prefill: the whole prompt ingests as one or more
                // T>1 forwards (prefillSlotAdmitted splits it into
                // _prefillChunk-sized chunks, each carrying KV + recurrent
                // state forward), replacing token-by-token prompt ingestion.
                // The TRUE mixed step (_mixedStep) instead prefills in-band with
                // decode — no eager prefill, so it skips toPrefill. Under the
                // pressure gate, a mixed-step slot admitted with no backlog still
                // takes the eager path (keeps live decoders on the fast path).
                const bool eagerPrefill =
                    !_mixedStep || (_mixedStepPressureGate && !prefillPressure);
                if (_prefillChunk > 0 && s.promptLen > 0 && eagerPrefill) {
                    toPrefill.push_back(i);
                }
                ++admitted;
            }

            for (std::size_t i = 0; i < _maxBatch; ++i) {
                if (_slots[i].occupied) nActive = i + 1;
            }

            if (nActive == 0) {
                // Idle: wait for a submit (or shutdown).
                _cv.wait(lk, [this] { return !_running || !_waiting.empty(); });
                continue;
            }
        }

        // --- Chunked prefill of newly-admitted slots (Increment A) --------
        // Each slot's whole prompt ingests as T>1 forwards over its own slot
        // (GPU work, no lock), then the slot enters decode at pos==promptLen
        // with its first generated token already pushed. When disabled
        // (_prefillChunk==0) toPrefill is empty and prompts ingest
        // token-by-token through the decode step below (legacy path).
        // Batch the ragged prefill forward when explicitly enabled, OR when the
        // true mixed step diverted a no-backlog burst to eager prefill: those
        // slots must still prefill in a dedicated BATCHED forward (not one serial
        // single-slot forward each), else the pressure gate would regress prefill
        // throughput back to the pre-5.21 serialization.
        if (_mixedBatchPrefill || _mixedStep) {
            // 5.21-III MULTI-SLOT — batch each maximal CONTIGUOUS run of admitted
            // slots into ragged forwards. toPrefill is ascending (admit loop
            // iterates slot index ascending), so a run breaks only at a gap.
            std::size_t r0 = 0;
            while (r0 < toPrefill.size()) {
                std::size_t r1 = r0 + 1;
                while (r1 < toPrefill.size() &&
                       toPrefill[r1] == toPrefill[r1 - 1] + 1) {
                    ++r1;
                }
                prefillSlotRunBatched(
                    std::span<const std::size_t>{toPrefill.data() + r0, r1 - r0});
                r0 = r1;
            }
        } else {
            for (std::size_t si : toPrefill) {
                prefillSlotAdmitted(si);
            }
        }

        // --- 5.21-III TRUE MIXED STEP ------------------------------------
        // When enabled, prefill was NOT done eagerly above (toPrefill empty). If
        // any slot still has prompt tokens to ingest, run ONE ragged forward
        // that folds every slot's next prefill chunk OR decode token together,
        // then loop. Pure-decode iterations (nothing prefilling) fall through to
        // the optimized decode step below (paged decode-v2).
        if (_mixedStep) {
            bool anyPrefilling = false;
            {
                std::lock_guard<std::mutex> lk(_mtx);
                for (std::size_t i = 0; i < _maxBatch; ++i) {
                    if (_slots[i].occupied &&
                        _slots[i].prefillPos < _slots[i].promptLen) {
                        anyPrefilling = true;
                        break;
                    }
                }
            }
            if (anyPrefilling) {
                const bool freed = runMixedStep();
                if (freed) _cv.notify_all();
                continue;   // iteration fully handled by the mixed forward
            }
        }

        // Snapshot per-slot step inputs under the lock; run the forward
        // outside it so submitters never block on a decode iteration.
        // Re-taken after prefill: a slot may have retired (eos on the first
        // token / cancel) and prefilled slots now sit at pos==promptLen.
        {
            std::unique_lock<std::mutex> lk(_mtx);
            nActive = 0;
            for (std::size_t i = 0; i < _maxBatch; ++i) {
                if (_slots[i].occupied) nActive = i + 1;
            }
            if (nActive == 0) {
                continue;   // everything retired during prefill — loop back
            }
            steps.resize(nActive);
            for (std::size_t i = 0; i < nActive; ++i) {
                Step st{};
                st.slot = static_cast<std::uint32_t>(i);
                if (_slots[i].occupied) {
                    const Slot& s = _slots[i];
                    const std::size_t p = s.pos;
                    st.token    = (p < s.promptLen) ? s.prompt[p] : s.lastTok;
                    st.pos      = static_cast<std::int32_t>(p);
                    st.seqStart = (p == 0);
                } else {
                    // Idle hole inside the prefix: fresh 1-token dummy.
                    st.token = 0; st.pos = 0; st.seqStart = true;
                }
                steps[i] = st;
            }
        }

        toks.assign(nActive, 0);
        try {
            _engine.stepServing(steps, toks);
        } catch (const std::exception& ex) {
            // A forward failure poisons the whole batch — fail every active
            // request rather than silently losing tokens.
            std::lock_guard<std::mutex> lk(_mtx);
            for (std::size_t i = 0; i < nActive; ++i) {
                if (_slots[i].occupied) {
                    finish(_slots[i].req, std::string("stepServing failed: ") + ex.what());
                    _slots[i].occupied = false;
                    _slots[i].req.reset();
                }
            }
            continue;
        }

        // Distribute sampled tokens + retire finished requests.
        bool freedSlot = false;
        {
            std::lock_guard<std::mutex> lk(_mtx);
            for (std::size_t i = 0; i < nActive; ++i) {
                if (!_slots[i].occupied) continue;
                Slot& s = _slots[i];

                // Client-requested cancellation: retire + free immediately.
                bool cancelled = false;
                {
                    std::lock_guard<std::mutex> rl(s.req->mtx);
                    cancelled = s.req->cancelled;
                }
                if (cancelled) {
                    finish(s.req, "");
                    s.occupied = false;
                    s.req.reset();
                    s.prompt.clear();
                    freedSlot = true;
                    continue;
                }

                const bool isGen = (s.pos + 1 >= s.promptLen);
                if (isGen) {
                    const std::int32_t t = toks[i];
                    {
                        std::lock_guard<std::mutex> rl(s.req->mtx);
                        s.req->tokens.push_back(t);
                        s.req->cv.notify_all();
                    }
                    s.lastTok = t;
                    ++s.produced;
                    const bool ctxFull = (s.pos + 2 >= _maxContext);
                    const bool stop =
                        (_eosId >= 0 && t == _eosId) || isStop(t, s) ||
                        s.produced >= s.maxNew || ctxFull;
                    if (stop) {
                        {
                            std::lock_guard<std::mutex> rl(s.req->mtx);
                            s.req->done = true;
                            s.req->cv.notify_all();
                        }
                        s.occupied = false;
                        s.req.reset();
                        s.prompt.clear();
                        freedSlot = true;
                        continue;
                    }
                }
                ++s.pos;
            }
        }
        if (freedSlot) {
            _cv.notify_all();  // a slot opened — let waiting requests admit
        }
    }
}

} // namespace mimirmind::runtime::serving
