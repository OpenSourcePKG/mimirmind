// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include "runtime/KvCache.hpp"
#include "runtime/serving/ServingKvSubstrate.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace mimirmind::compute { class ComputeOps; }

namespace mimirmind::runtime::serving {

/**
 * Non-paged per-slot KV substrate for the L0 / Xe-LPG serving path
 * (M9.1 / ADR 2026-08-18, decision R1: NON-PAGED).
 *
 * Owns `maxBatch` independent `KvCache` slabs — one contiguous cache per
 * concurrent slot, each `[contextCap, kvDim(layer)]` per layer. A slot's
 * `KvCache` is exactly what `GemmaBaseBackend::runBlockBatched` writes
 * into (per-sequence `KvCache* caches[i]`), so the batched-decode kernel
 * needs no new KV path — the slab pool IS its contract.
 *
 * Per-layer `kvDim` is honoured verbatim, so the Gemma 4 SWA/full-attn
 * head-dim split (8 × 256 sliding vs 2 × 512 full) and the E4B shared-KV
 * aliasing (`kvSourceLayer`) both come through the underlying `KvCache`
 * constructor — no geometry assumptions live here.
 *
 * Deliberately NOT paged: on the single-user / low-concurrency Xe-LPG
 * target, block-paging only adds block-table indirection with no
 * cross-sequence sharing benefit. One physical slab per slot; the slot
 * count (== maxBatch) bounds the resident KV footprint.
 *
 * Lifecycle: one pool per serving engine, allocated once at serving
 * start. Slots are reused across requests via `resetSlot`.
 */
class KvCacheSlabPool final : public ServingKvSubstrate {
public:
    /**
     * Allocate `maxBatch` slabs, each a `KvCache(contextCap, kvDimPerLayer,
     * kvSourceLayer, dtype)`. All of `maxBatch` / `contextCap` must be > 0
     * and `kvDimPerLayer` non-empty; the ctor throws `std::invalid_argument`
     * otherwise. `kvSourceLayer` follows the `KvCache` contract (empty =
     * identity / every layer owns its KV).
     */
    KvCacheSlabPool(compute::ComputeOps&      ops,
                    std::size_t               maxBatch,
                    std::size_t               contextCap,
                    std::vector<std::size_t>  kvDimPerLayer,
                    std::vector<std::size_t>  kvSourceLayer = {},
                    KvDtype                   dtype = KvDtype::F32);

    KvCacheSlabPool(const KvCacheSlabPool&)            = delete;
    KvCacheSlabPool& operator=(const KvCacheSlabPool&) = delete;
    KvCacheSlabPool(KvCacheSlabPool&&)                 = delete;
    KvCacheSlabPool& operator=(KvCacheSlabPool&&)      = delete;

    // --- ServingKvSubstrate ---------------------------------------------
    [[nodiscard]] std::size_t capacity()   const noexcept override { return _slots.size(); }
    [[nodiscard]] std::size_t contextCap() const noexcept override { return _contextCap; }
    [[nodiscard]] std::size_t numLayers()  const noexcept override { return _numLayers; }

    [[nodiscard]] KvCache&       slot(std::size_t i)       override;
    [[nodiscard]] const KvCache& slot(std::size_t i) const override;

    void resetSlot(std::size_t i) noexcept override;

    // --- Batched-decode convenience -------------------------------------
    /**
     * Fill `out` with the `KvCache*` of the active prefix `[0, nSeq)` so
     * the caller can pass `std::span<KvCache* const>{out}` straight to
     * `runBlockBatched`. `out` is cleared and resized (the batcher keeps a
     * reusable vector so the hot path stays allocation-free after warmup).
     * Throws `std::out_of_range` if `nSeq > capacity()`.
     */
    void activeSlotCaches(std::size_t nSeq, std::vector<KvCache*>& out);

    /// Per-layer KV width (elements) — mirrors the slab's `KvCache`.
    [[nodiscard]] std::size_t kvDim(std::size_t layer) const noexcept;
    [[nodiscard]] KvDtype     dtype() const noexcept { return _dtype; }

private:
    std::size_t _contextCap;
    std::size_t _numLayers;
    KvDtype     _dtype;
    std::vector<std::unique_ptr<KvCache>> _slots;
};

} // namespace mimirmind::runtime::serving