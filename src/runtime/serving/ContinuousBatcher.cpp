// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/serving/ContinuousBatcher.hpp"

#include "runtime/InferenceEngine.hpp"
#include "core/log/Log.hpp"

#include <algorithm>
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
    _slots.resize(maxBatch);
    _running = true;
    _worker  = std::thread(&ContinuousBatcher::workerLoop, this);
    MM_LOG_INFO("serving",
                "ContinuousBatcher: started (maxBatch={} maxContext={} "
                "maxInflight={} maxInflightPerTenant={} eosId={} prefillChunk={})",
                maxBatch, maxContext, _maxInflight,
                _maxInflightPerTenant, eosId, _prefillChunk);
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

    // Commit: the slot enters decode at pos == promptLen with firstTok already
    // produced. (Only the worker thread admits/frees slots, so the slot cannot
    // have been re-used mid-prefill; the guards are defensive.)
    std::lock_guard<std::mutex> lk(_mtx);
    Slot& s = _slots[slot];
    if (!s.occupied || s.req != req) {
        return;
    }
    s.pos      = s.promptLen;
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

            // Admit waiting requests into free slots (lowest index first so
            // the active set stays a tight prefix).
            for (std::size_t i = 0; i < _maxBatch && !_waiting.empty(); ++i) {
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
                // Chunked prefill: the whole prompt ingests as one or more
                // T>1 forwards (prefillSlotAdmitted splits it into
                // _prefillChunk-sized chunks, each carrying KV + recurrent
                // state forward), replacing token-by-token prompt ingestion.
                if (_prefillChunk > 0 && s.promptLen > 0) {
                    toPrefill.push_back(i);
                }
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
        for (std::size_t si : toPrefill) {
            prefillSlotAdmitted(si);
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
