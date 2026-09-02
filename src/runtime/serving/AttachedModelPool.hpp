// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mimirmind::runtime::serving {

/**
 * Worker-side materialize/evict cache for M-Munin.3 (per-request model
 * switch). Holds at most `capacity` (K) materialized model payloads at once
 * and switches on demand: a request for a model that is not resident
 * materializes it (via the supplied factory — which, in production, attaches
 * to Munin and runs the NVFP4/GGUF materialization), evicting the LRU
 * *unpinned* slot first when the cache is full. This is the standard
 * LRU-multiplex pattern (Ray Serve model multiplexing; vLLM sleep/wake is the
 * single-process cousin). See decisions/2026-08-22-m-munin3-per-request-
 * model-switch.md.
 *
 * `Payload` is the materialized model bundle (in production: the
 * InferenceEngine + its MuninClient + importer, kept alive together). The
 * pool is generic over it so the switch/evict/pin logic is unit-testable with
 * a fake payload and no GPU.
 *
 * Concurrency contract:
 *   - `acquire(id)` returns a Handle that PINS the slot (refcount) — the slot
 *     cannot be evicted while any Handle for it lives. This is the in-flight-
 *     request guard: a running decode always holds a Handle, so its device
 *     buffers are never freed under it.
 *   - Only ONE materialization runs at a time (serialized switch). A miss
 *     while a switch is in progress waits and retries (and usually finds the
 *     model freshly resident). Eviction of a pinned victim waits until its
 *     last Handle is released (an implicit drain). The slow factory call runs
 *     WITHOUT the pool lock held, so releases/hits proceed meanwhile.
 *
 * Not exception-safe against a throwing Payload destructor (don't). The
 * factory may throw / return null to signal a materialization failure.
 */
template <class Payload>
class AttachedModelPool {
public:
    /// Materialize the model with id `modelId`. Returns the payload, or null /
    /// throws on failure. Runs without the pool lock held (it is slow).
    using Factory = std::function<std::unique_ptr<Payload>(const std::string& modelId)>;

    /**
     * RAII pin on a resident slot. While it lives, the slot is not evictable.
     * Move-only. `->` / `*` / `get()` reach the payload.
     */
    class Handle {
    public:
        Handle() = default;
        Handle(AttachedModelPool* pool, std::string id, Payload* payload)
            : _pool{pool}, _id{std::move(id)}, _payload{payload} {}

        ~Handle() { reset(); }

        Handle(Handle&& o) noexcept
            : _pool{o._pool}, _id{std::move(o._id)}, _payload{o._payload} {
            o._pool = nullptr;
            o._payload = nullptr;
        }
        Handle& operator=(Handle&& o) noexcept {
            if (this != &o) {
                reset();
                _pool = o._pool;
                _id = std::move(o._id);
                _payload = o._payload;
                o._pool = nullptr;
                o._payload = nullptr;
            }
            return *this;
        }
        Handle(const Handle&)            = delete;
        Handle& operator=(const Handle&) = delete;

        [[nodiscard]] Payload* operator->() const noexcept { return _payload; }
        [[nodiscard]] Payload& operator*()  const noexcept { return *_payload; }
        [[nodiscard]] Payload* get()        const noexcept { return _payload; }
        [[nodiscard]] const std::string& modelId() const noexcept { return _id; }
        explicit operator bool() const noexcept { return _payload != nullptr; }

        /// Release the pin early (idempotent).
        void reset() noexcept {
            if (_pool != nullptr) {
                _pool->release(_id);
                _pool = nullptr;
                _payload = nullptr;
            }
        }

    private:
        AttachedModelPool* _pool{nullptr};
        std::string        _id{};
        Payload*           _payload{nullptr};
    };

    AttachedModelPool(std::size_t              capacity,
                      std::vector<std::string> knownModelIds,
                      Factory                  factory)
        : _capacity{capacity == 0 ? std::size_t{1} : capacity},
          _factory{std::move(factory)} {
        for (auto& id : knownModelIds) {
            _known.insert(std::move(id));
        }
    }

    AttachedModelPool(const AttachedModelPool&)            = delete;
    AttachedModelPool& operator=(const AttachedModelPool&) = delete;
    AttachedModelPool(AttachedModelPool&&)                 = delete;
    AttachedModelPool& operator=(AttachedModelPool&&)      = delete;

    /**
     * Acquire a ready payload for `modelId`, materializing (and evicting the
     * LRU unpinned slot if full) on a miss. Blocks during a switch. Throws
     * std::runtime_error on an unknown model id or a factory failure.
     */
    [[nodiscard]] Handle acquire(const std::string& modelId) {
        std::unique_lock<std::mutex> lk{_mx};
        if (!_known.contains(modelId)) {
            throw std::runtime_error{
                "AttachedModelPool: unknown model '" + modelId + "'"};
        }
        for (;;) {
            if (auto it = _resident.find(modelId); it != _resident.end()) {
                it->second.refs++;
                it->second.lastUsed = ++_clock;
                return Handle{this, modelId, it->second.payload.get()};
            }
            // Miss. Serialize switches: only one materialization at a time.
            if (_switching) {
                _cv.wait(lk);
                continue;
            }
            _switching = true;

            // Make room: evict the LRU unpinned slot; if all are pinned, wait
            // for a release (implicit drain) — holding the switch token.
            while (_resident.size() >= _capacity) {
                std::string   victim;
                std::uint64_t oldest = UINT64_MAX;
                for (const auto& [id, slot] : _resident) {
                    if (slot.refs == 0 && slot.lastUsed < oldest) {
                        oldest = slot.lastUsed;
                        victim = id;
                    }
                }
                if (victim.empty()) {
                    _cv.wait(lk);   // all pinned; wait for an in-flight release
                    continue;
                }
                _resident.erase(victim);  // destroy payload -> frees the model
            }

            // Materialize OUTSIDE the lock (slow); other acquires see
            // _switching and wait, releases still proceed.
            lk.unlock();
            std::unique_ptr<Payload> payload;
            std::string              err;
            try {
                payload = _factory(modelId);
            } catch (const std::exception& x) {
                err = x.what();
            } catch (...) {
                err = "unknown factory exception";
            }
            lk.lock();
            _switching = false;
            _cv.notify_all();

            if (!payload) {
                throw std::runtime_error{
                    "AttachedModelPool: materialize '" + modelId +
                    "' failed" + (err.empty() ? "" : (": " + err))};
            }
            Slot slot;
            slot.payload  = std::move(payload);
            slot.refs     = 1;
            slot.lastUsed = ++_clock;
            auto [it2, inserted] = _resident.emplace(modelId, std::move(slot));
            (void)inserted;
            return Handle{this, modelId, it2->second.payload.get()};
        }
    }

    [[nodiscard]] std::size_t capacity() const noexcept { return _capacity; }

    [[nodiscard]] std::size_t residentCount() const {
        std::lock_guard<std::mutex> lk{_mx};
        return _resident.size();
    }

    [[nodiscard]] bool isResident(const std::string& modelId) const {
        std::lock_guard<std::mutex> lk{_mx};
        return _resident.contains(modelId);
    }

    [[nodiscard]] bool knows(const std::string& modelId) const {
        std::lock_guard<std::mutex> lk{_mx};
        return _known.contains(modelId);
    }

    /// Read-only snapshot of the resident slots, for status/telemetry. Invokes
    /// `fn(const std::string& id, const Payload&)` once per currently-
    /// materialized slot under the pool lock, WITHOUT touching LRU order or
    /// refcounts (so it never perturbs eviction) and without taking any per-
    /// engine generate lock. `fn` must be cheap (a telemetry read) — it runs
    /// inside the short pool critical section. It never contends with the slow
    /// factory materialization, which runs with the pool lock released.
    template <class Fn>
    void snapshotResident(Fn&& fn) const {
        std::lock_guard<std::mutex> lk{_mx};
        for (const auto& [id, slot] : _resident) {
            fn(id, *slot.payload);
        }
    }

private:
    struct Slot {
        std::unique_ptr<Payload> payload;
        int                      refs{0};
        std::uint64_t            lastUsed{0};
    };

    void release(const std::string& modelId) noexcept {
        std::lock_guard<std::mutex> lk{_mx};
        auto it = _resident.find(modelId);
        if (it != _resident.end() && it->second.refs > 0) {
            if (--it->second.refs == 0) {
                _cv.notify_all();  // a pinned victim may now be evictable
            }
        }
    }

    mutable std::mutex               _mx;
    std::condition_variable          _cv;
    std::unordered_set<std::string>  _known;
    std::unordered_map<std::string, Slot> _resident;
    std::size_t                      _capacity;
    Factory                          _factory;
    bool                             _switching{false};
    std::uint64_t                    _clock{0};
};

} // namespace mimirmind::runtime::serving
