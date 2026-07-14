// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "core/config/Config.hpp"
#include "core/gguf/GgufReader.hpp"
#include "core/gguf/WeightsMap.hpp"
#include "core/ipc/TensorManifest.hpp"
#include "core/l0/L0Context.hpp"
#include "core/l0/UsmAllocator.hpp"
#include "munin/ChunkAllocator.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace mimirmind::munin {

/**
 * One model loaded into USM(host), ready to be exported to an attached
 * worker. The GgufReader owns the mmap and the per-tensor USM allocations;
 * dropping the LoadedModel releases everything (which is what Munin
 * shutdown does — and, by design, invalidates every worker's IPC pointers).
 *
 * Not thread-safe on its own — the ModelStore serialises look-ups behind
 * a mutex. Once a LoadedModel is handed out via a const reference, the
 * per-tensor `usmPtr` values are stable for the process lifetime, so
 * concurrent readers of the immutable payload are fine.
 */
struct LoadedModel {
    std::string                                              id{};
    std::string                                              path{};
    std::string                                              fingerprint{};
    std::uint64_t                                            totalBytes{0};
    // Chunk-owned USM backing every tensor's `usmPtr`. Kept alive for
    // the entire lifetime of the LoadedModel — dropping it releases
    // all model memory and invalidates every attached worker's IPC
    // pointers. Ordered before `reader` so it outlives the reader's
    // views (declaration order == destruction order in reverse).
    std::unique_ptr<::mimirmind::munin::ChunkAllocator>      chunks{};
    std::unique_ptr<::mimirmind::core::gguf::GgufReader>     reader{};
    std::unique_ptr<::mimirmind::core::gguf::WeightsMap>     weights{};

    /**
     * Populate a `TensorManifest` (wire v2) from this model. Fills the
     * `chunks` list from the ChunkAllocator's used-byte counts and, per
     * tensor, the `chunkIndex + chunkOffset` recorded by
     * `GgufReader::loadTensorsIntoChunks`. One IPC HANDLE is emitted
     * per chunk, not per tensor — see M-Munin.1a ADR.
     */
    [[nodiscard]] ::mimirmind::core::ipc::TensorManifest buildManifest() const;
};

/**
 * Owns every model Munin holds in USM. Constructed once at daemon startup
 * from `Config.models[loadOnStart]`, immutable afterwards for MVP scope —
 * dynamic add/evict is Tier-3 (`munin-ctl preload/evict`, see M-Munin ADR).
 *
 * Attaches look up by `modelId` (matches `ModelEntry.id`). Missing id →
 * `find` returns `nullptr` and the AttachSession returns an error to the
 * worker rather than exporting silently-wrong tensors.
 *
 * The allocator is passed by reference and must outlive the store. It is
 * shared across all models — a single per-process USM pool keeps the
 * probe-based allocation-ceiling accounting honest (the pool sees the
 * sum of all model footprints, not just one).
 */
class ModelStore {
public:
    /**
     * Load every entry in `cfg.models` with `loadOnStart:true`. Throws on
     * any GGUF-load failure — Munin refuses to come up in a partially-loaded
     * state because a worker attaching to model X while model Y is missing
     * would silently succeed for X and then fail an hour later when a
     * request routes to Y.
     *
     * Each loaded model gets its own `ChunkAllocator` (M-Munin.1a ADR):
     * tensors are packed into ~1 GiB chunks so the IPC-Export path emits
     * one handle per chunk instead of one per tensor. `l0Ctx` is needed
     * to construct those chunk allocators via `zeMemAllocHost`. The
     * `UsmAllocator` reference is retained for possible future non-chunk
     * allocations (scratch buffers, non-tensor state); for MVP scope it
     * is unused by the constructor itself.
     */
    ModelStore(const ::mimirmind::core::config::Config& cfg,
               ::mimirmind::core::l0::L0Context&        l0Ctx,
               ::mimirmind::core::l0::UsmAllocator&    allocator);

    ~ModelStore();

    ModelStore(const ModelStore&)            = delete;
    ModelStore& operator=(const ModelStore&) = delete;
    ModelStore(ModelStore&&)                 = delete;
    ModelStore& operator=(ModelStore&&)      = delete;

    /**
     * O(1) lookup by model id. Returns nullptr when not loaded.
     * The returned pointer is stable for the store's lifetime.
     */
    [[nodiscard]] const LoadedModel* find(std::string_view modelId) const noexcept;

    /**
     * Snapshot of {id, fingerprint, totalBytes} for the healthz response.
     * Const-correct: the vector is a copy, the returned strings are
     * duplicated from the store.
     */
    struct ModelSummary {
        std::string   id;
        std::string   fingerprint;
        std::uint64_t totalBytes{0};
        std::uint32_t tensorCount{0};
    };

    [[nodiscard]] std::vector<ModelSummary> summaries() const;

    [[nodiscard]] std::size_t size() const noexcept { return _byId.size(); }

private:
    std::unordered_map<std::string, std::unique_ptr<LoadedModel>> _byId;
};

} // namespace mimirmind::munin