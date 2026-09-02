// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "runtime/InferenceEngine.hpp"
#include "runtime/serving/AttachedModelPool.hpp"
#include "runtime/serving/ContinuousBatcher.hpp"
#include "runtime/spec/SpeculativeDecoder.hpp"
#include "server/ModelProvider.hpp"

#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mimirmind::server {

/**
 * The materialized bundle the pool holds per resident model (M-Munin.3). Owns
 * the worker-side InferenceEngine (device BF16) plus the IPC keep-alives whose
 * mappings back the attached weights. Destroying it (on eviction) frees the
 * device memory and detaches from Munin — so at most `capacity` models are
 * materialized at once. `mutex` serialises requests to this engine (the
 * per-model request lock the dispatcher would otherwise take on defaultMutex).
 */
struct PooledEngine {
    std::unique_ptr<runtime::InferenceEngine> engine;
    std::shared_ptr<void>                     keepAliveImporter;  // outlives client
    std::shared_ptr<void>                     keepAliveClient;
    std::string                               title;
    std::mutex                                mutex;
    /// M-Munin.3 (full): this slot's own continuous batcher (null if the
    /// model is below the serving-class capacity threshold or batching is
    /// disabled) and spec-dec decoder (null if speculative.enabled=false or
    /// this model isn't the speculative.target). Built by ServeMode's
    /// factory alongside `engine`, torn down automatically on eviction.
    /// Declared LAST on purpose: C++ destroys members in reverse
    /// declaration order, so `spec`/`batcher` (which hold a reference into
    /// `engine`, e.g. a worker thread that calls back into it) are torn
    /// down BEFORE `engine` itself.
    std::unique_ptr<runtime::serving::ContinuousBatcher> batcher;
    std::unique_ptr<runtime::SpeculativeDecoder>          spec;
};

/**
 * Concrete pool-backed ModelProvider (M-Munin.3 Brick 2b). Wraps an
 * AttachedModelPool<PooledEngine> + a caller-supplied materialize factory
 * (built by ServeMode, capturing the config / backend / Munin socket). Each
 * `acquire()` returns a ready engine pinned by a `pin` whose destruction
 * releases the pool slot — so an in-flight request (including an async stream
 * holding the pin) blocks eviction of the model it is using.
 *
 * K=1 MVP: one materialized model at a time; a request for a not-resident model
 * blocks on the ~materialization (evict LRU + attach + dequant). Session-affine
 * / low-switch-frequency use only (see the ADR).
 */
class AttachedModelProvider : public ModelProvider {
public:
    using Pool    = runtime::serving::AttachedModelPool<PooledEngine>;
    using Factory = Pool::Factory;   // std::function<unique_ptr<PooledEngine>(const std::string&)>

    AttachedModelProvider(std::size_t                 capacity,
                          std::vector<ProvidedModel>  models,
                          std::string                 defaultModelId,
                          Factory                     factory)
        : _pool{capacity, idsOf(models), std::move(factory)},
          _models{std::move(models)},
          _defaultId{std::move(defaultModelId)} {
        for (const auto& m : _models) {
            _known.insert(m.id);
        }
    }

    [[nodiscard]] std::string defaultModelId() const override { return _defaultId; }

    [[nodiscard]] bool knows(const std::string& modelId) const override {
        return _known.contains(modelId);
    }

    [[nodiscard]] std::vector<ProvidedModel> listModels() const override {
        return _models;
    }

    [[nodiscard]] std::optional<AcquiredModel> acquire(const std::string& modelId) override {
        if (!knows(modelId)) {
            return std::nullopt;
        }
        // May throw on a materialization failure — that propagates to the
        // dispatcher (503), per the ModelProvider contract.
        typename Pool::Handle handle = _pool.acquire(modelId);
        PooledEngine* payload = handle.get();
        // The pin IS the handle: while any copy lives the slot stays pinned;
        // its destruction calls pool.release(id).
        std::shared_ptr<void> pin =
            std::make_shared<typename Pool::Handle>(std::move(handle));
        return AcquiredModel{payload->engine.get(), &payload->mutex,
                             std::move(pin), modelId, payload->title,
                             payload->batcher.get(), payload->spec.get()};
    }

    [[nodiscard]] std::vector<ResidentModelMemory> residentModels() const override {
        std::vector<ResidentModelMemory> out;
        // Read-only pool snapshot: does not disturb LRU or take a generate
        // lock. memoryTelemetry() is a fast counter read, safe under the pool
        // lock.
        _pool.snapshotResident([&](const std::string& id, const PooledEngine& pe) {
            if (pe.engine == nullptr) {
                return;
            }
            const auto mt = pe.engine->memoryTelemetry();
            out.push_back(ResidentModelMemory{
                id, pe.title, id == _defaultId,
                mt.weightBytes, mt.servingActive, mt.kvResidentBytes, mt.kvNumBlocks});
        });
        return out;
    }

    [[nodiscard]] std::size_t poolCapacity() const override { return _pool.capacity(); }

private:
    [[nodiscard]] static std::vector<std::string>
    idsOf(const std::vector<ProvidedModel>& models) {
        std::vector<std::string> ids;
        ids.reserve(models.size());
        for (const auto& m : models) {
            ids.push_back(m.id);
        }
        return ids;
    }

    Pool                            _pool;
    std::vector<ProvidedModel>      _models;
    std::string                     _defaultId;
    std::unordered_set<std::string> _known;
};

} // namespace mimirmind::server
