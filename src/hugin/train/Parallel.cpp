// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "hugin/train/Parallel.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <exception>
#include <thread>
#include <vector>

namespace mimirmind::hugin::train {

namespace {
std::atomic<std::size_t> g_threads{0};
} // namespace

void Parallel::setThreads(std::size_t n) noexcept { g_threads.store(n); }

std::size_t Parallel::threads() noexcept {
    const std::size_t n = g_threads.load();
    if (n > 0) return n;
    const std::size_t hw = std::thread::hardware_concurrency();
    return std::max<std::size_t>(1, hw / 2);
}

namespace {

/// Persistent workers: a Kolibri train step issues hundreds of thousands of
/// small forRange calls (one per routed-expert matmul), so spawning threads
/// per call cost more than the math. The caller runs chunk 0 itself; a call
/// from inside a worker, or while another thread holds the pool, runs inline.
class Pool {
public:
    static Pool& instance() {
        static Pool p;
        return p;
    }

    /// False when the pool is busy (concurrent caller) or the call is nested
    /// inside a pool job: run inline then.
    bool run(std::size_t workers, std::size_t n, std::size_t chunk,
             const std::function<void(std::size_t, std::size_t)>& fn) {
        if (t_inWorker) return false;   // nested: also the caller while it runs chunk 0
        std::unique_lock<std::mutex> owner(_callMutex, std::try_to_lock);
        if (!owner.owns_lock()) return false;
        ensure(workers - 1);
        std::vector<std::exception_ptr> errors(workers);
        {
            std::lock_guard<std::mutex> lk(_m);
            _fn      = &fn;
            _n       = n;
            _chunk   = chunk;
            _active  = workers - 1;
            _errors  = &errors;
            _pending = workers - 1;
            ++_generation;
        }
        _cv.notify_all();
        t_inWorker = true;
        try {
            fn(0, std::min(n, chunk));
        } catch (...) {
            errors[0] = std::current_exception();
        }
        t_inWorker = false;
        {
            std::unique_lock<std::mutex> lk(_m);
            _done.wait(lk, [this] { return _pending == 0; });
            _fn = nullptr;
        }
        for (const auto& e : errors) {
            if (e) std::rethrow_exception(e);
        }
        return true;
    }

    Pool(const Pool&)            = delete;
    Pool& operator=(const Pool&) = delete;

private:
    Pool() = default;
    ~Pool() {
        {
            std::lock_guard<std::mutex> lk(_m);
            _stop = true;
            ++_generation;
        }
        _cv.notify_all();
        for (auto& t : _threads) t.join();
    }

    /// Called before the generation of the new job is published: a new worker
    /// starts from the current generation, so it cannot miss that job.
    void ensure(std::size_t count) {
        std::size_t gen = 0;
        {
            std::lock_guard<std::mutex> lk(_m);
            gen = _generation;
        }
        while (_threads.size() < count) {
            const std::size_t id = _threads.size() + 1;   // chunk index; 0 is the caller
            _threads.emplace_back([this, id, gen] { loop(id, gen); });
        }
    }

    void loop(std::size_t id, std::size_t seen) {
        t_inWorker = true;
        for (;;) {
            const std::function<void(std::size_t, std::size_t)>* fn = nullptr;
            std::size_t n = 0, chunk = 0;
            std::vector<std::exception_ptr>* errors = nullptr;
            {
                std::unique_lock<std::mutex> lk(_m);
                _cv.wait(lk, [&] { return _stop || _generation != seen; });
                if (_stop) return;
                seen = _generation;
                if (id > _active) continue;   // not needed for this call
                fn     = _fn;
                n      = _n;
                chunk  = _chunk;
                errors = _errors;
            }
            const std::size_t b = id * chunk, e = std::min(n, b + chunk);
            if (b < e) {
                try {
                    (*fn)(b, e);
                } catch (...) {
                    (*errors)[id] = std::current_exception();
                }
            }
            {
                std::lock_guard<std::mutex> lk(_m);
                if (--_pending == 0) _done.notify_one();
            }
        }
    }

    static thread_local bool t_inWorker;

    std::mutex                _callMutex;
    std::mutex                _m;
    std::condition_variable   _cv, _done;
    std::vector<std::thread>  _threads;
    const std::function<void(std::size_t, std::size_t)>* _fn{nullptr};
    std::size_t               _n{0}, _chunk{0}, _active{0}, _pending{0}, _generation{0};
    std::vector<std::exception_ptr>* _errors{nullptr};
    bool                      _stop{false};
};

thread_local bool Pool::t_inWorker = false;

} // namespace

void Parallel::forRange(std::size_t n, std::size_t minChunk,
                        const std::function<void(std::size_t, std::size_t)>& fn) {
    if (n == 0) return;
    const std::size_t maxByWork = std::max<std::size_t>(1, n / std::max<std::size_t>(1, minChunk));
    const std::size_t workers   = std::min(threads(), maxByWork);
    if (workers <= 1) {
        fn(0, n);
        return;
    }
    const std::size_t chunk = (n + workers - 1) / workers;
    if (!Pool::instance().run(workers, n, chunk, fn)) fn(0, n);
}

} // namespace mimirmind::hugin::train
