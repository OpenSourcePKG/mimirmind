// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/serving/ContinuousBatcher.hpp"

#include "runtime/InferenceEngine.hpp"
#include "core/log/Log.hpp"

#include <algorithm>
#include <utility>

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
                                     std::int32_t eosId)
    : _engine(engine),
      _maxBatch(maxBatch),
      _maxContext(maxContext),
      _eosId(eosId) {
    if (maxBatch == 0 || maxContext == 0) {
        throw std::runtime_error("ContinuousBatcher: maxBatch/maxContext > 0");
    }
    // Build the persistent per-slot serving state up front (throws for a
    // non-qwen35moe engine, before any request is accepted).
    _engine.ensureServingState(maxBatch, maxContext);
    _slots.resize(maxBatch);
    _running = true;
    _worker  = std::thread(&ContinuousBatcher::workerLoop, this);
    MM_LOG_INFO("serving",
                "ContinuousBatcher: started (maxBatch={} maxContext={} eosId={})",
                maxBatch, maxContext, eosId);
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
        std::vector<std::int32_t> stopIds) {
    auto req = std::make_shared<ServingRequest>();
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
            }

            for (std::size_t i = 0; i < _maxBatch; ++i) {
                if (_slots[i].occupied) nActive = i + 1;
            }

            if (nActive == 0) {
                // Idle: wait for a submit (or shutdown).
                _cv.wait(lk, [this] { return !_running || !_waiting.empty(); });
                continue;
            }

            // Snapshot per-slot step inputs under the lock; run the forward
            // outside it so submitters never block on a decode iteration.
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
