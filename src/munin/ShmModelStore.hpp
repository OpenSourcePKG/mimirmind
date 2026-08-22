// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/config/Config.hpp"
#include "munin/ShmChunkAllocator.hpp"
#include "munin/ShmLoadedModel.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace mimirmind::munin {

/**
 * Owns every model a CUDA/GB10 Munin holds resident in POSIX-shm — the shm
 * analogue of ModelStore (ADR 2026-08-14, step 4). Config-driven: loads each
 * `cfg.models[loadOnStart:true]` entry into a ShmLoadedModel (GgufReader +
 * memfd-backed ShmChunkAllocator, raw-byte MVP). Immutable after
 * construction for MVP scope — dynamic add/evict is M-Munin.3.
 *
 * Unlike ModelStore this needs no L0Context / UsmAllocator — the chunks are
 * plain memfd host RAM — so it builds in mimirmind_core_common with no GPU
 * SDK. Dropping the store releases every model (munmap + close each memfd),
 * invalidating every attached worker's mapping (the shutdown semantics
 * Munin promises).
 *
 * Attaches look up by `modelId`. Missing id -> `find` returns nullptr and
 * ShmAttachSession returns an error to the worker.
 */
class ShmModelStore {
public:
    /**
     * Load every `loadOnStart:true` entry in `cfg.models`. Throws
     * std::runtime_error on any GGUF-load failure, a duplicate id, or when
     * no model is loadable — Munin refuses to come up half-loaded.
     * `chunkBytes` sizes the per-model ShmChunkAllocator.
     */
    explicit ShmModelStore(
        const ::mimirmind::core::config::Config& cfg,
        std::size_t chunkBytes = ShmChunkAllocator::kDefaultChunkBytes);

    ~ShmModelStore();

    ShmModelStore(const ShmModelStore&)            = delete;
    ShmModelStore& operator=(const ShmModelStore&) = delete;
    ShmModelStore(ShmModelStore&&)                 = delete;
    ShmModelStore& operator=(ShmModelStore&&)      = delete;

    /// O(1) lookup by model id. nullptr when not loaded; stable for the
    /// store's lifetime.
    [[nodiscard]] const ShmLoadedModel* find(std::string_view modelId) const noexcept;

    struct ModelSummary {
        std::string   id;
        std::string   fingerprint;
        std::uint64_t totalBytes{0};
        std::uint32_t tensorCount{0};
    };

    [[nodiscard]] std::vector<ModelSummary> summaries() const;

    [[nodiscard]] std::size_t size() const noexcept { return _byId.size(); }

private:
    std::unordered_map<std::string, std::unique_ptr<ShmLoadedModel>> _byId;
};

} // namespace mimirmind::munin
