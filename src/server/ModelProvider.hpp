// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace mimirmind::runtime {
class InferenceEngine;
class SpeculativeDecoder;
namespace serving {
class ContinuousBatcher;
}
}

namespace mimirmind::server {

/**
 * A model made ready for ONE request by a ModelProvider (M-Munin.3 seam).
 *
 * `pin` is an opaque ownership token that keeps the backing slot alive: while
 * any copy of it lives, the provider's worker-side materialize/evict cache
 * cannot evict this model, so `engine`/`mutex` stay valid — including across an
 * async stream that outlives the handler frame. The request MUST keep `pin`
 * alive for its whole lifetime. For the eager (non-pooled) path there is no
 * provider and this type is never produced; the engines are process-lifetime
 * resident and `pin` stays empty.
 */
struct AcquiredModel {
    runtime::InferenceEngine*            engine{nullptr};
    std::mutex*                          mutex{nullptr};
    std::shared_ptr<void>                pin{};    // release-on-destroy pin on the slot
    std::string                          id{};
    std::string                          title{};
    /// M-Munin.3 (full): this slot's own continuous batcher / spec-dec
    /// decoder, when the provider builds them per-slot (see
    /// AttachedModelProvider). Null when the model has no serving-class
    /// batcher (below the capacity threshold) or spec-dec is off/not
    /// applicable for it. The eager (non-pooled) path never produces an
    /// AcquiredModel at all, so these fields only matter in pool mode.
    runtime::serving::ContinuousBatcher* batcher{nullptr};
    runtime::SpeculativeDecoder*         spec{nullptr};
};

/// Minimal model descriptor for /v1/models listing.
struct ProvidedModel {
    std::string id{};
    std::string title{};
};

/**
 * Seam for M-Munin.3 per-request model switching. When a RequestDispatcher is
 * given a provider, it resolves each request's `model` through acquire()
 * (which may materialize a worker-side slot, evicting the LRU one) instead of
 * a fixed engine table. Absent a provider (default), the dispatcher keeps its
 * eager, all-engines-resident behavior unchanged — no regression for the
 * standalone / co-resident path.
 *
 * The concrete pool-backed implementation (owning an AttachedModelPool + the
 * Munin materialize factory) is wired by ServeMode in attached mode; this
 * header only defines the interface the dispatcher depends on.
 */
class ModelProvider {
public:
    virtual ~ModelProvider() = default;

    /// Fallback model id for requests that omit `model` (or send "").
    [[nodiscard]] virtual std::string defaultModelId() const = 0;

    /// True if `modelId` is a known (servable) model.
    [[nodiscard]] virtual bool knows(const std::string& modelId) const = 0;

    /// All known model ids, for /v1/models (default first by convention).
    [[nodiscard]] virtual std::vector<ProvidedModel> listModels() const = 0;

    /**
     * Acquire a ready engine for `modelId`, materializing on a miss (and
     * evicting the LRU unpinned slot when the pool is full). The returned
     * AcquiredModel's `pin` MUST outlive the whole request, including any
     * async stream. Returns std::nullopt when `modelId` is not known; throws
     * std::runtime_error on a materialization failure.
     */
    [[nodiscard]] virtual std::optional<AcquiredModel> acquire(const std::string& modelId) = 0;
};

} // namespace mimirmind::server
