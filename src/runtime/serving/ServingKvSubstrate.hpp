// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#pragma once

#include <cstddef>

namespace mimirmind::runtime { class KvCache; }

namespace mimirmind::runtime::serving {

/**
 * Backend-neutral per-slot KV substrate for serving-class batched decode
 * (M9.1 / ADR 2026-08-18 "Xe-LPG Serving-KV-Substrat").
 *
 * This is the seam the ComputeOps-neutral batched-decode path programs
 * against: `GemmaBaseBackend::runBlockBatched` consumes a
 * `std::span<KvCache* const>` of the active sequences' per-slot caches,
 * so a substrate only has to hand out one `KvCache` per concurrent slot.
 *
 * Two substrates satisfy this serving model:
 *   - `KvCacheSlabPool` (this contract) — the NON-PAGED L0/Xe-LPG store:
 *     one contiguous per-slot `KvCache` slab, sized to the serving
 *     context cap. Fits the single-user / low-concurrency profile where
 *     block-paging buys no sharing benefit, and matches the per-sequence
 *     `KvCache` contract `runBlockBatched` already writes into.
 *   - `PagedKvPool` — the paged, block-shared store the qwen35moe CUDA /
 *     Bragi path uses via a DIFFERENT contract (block/slot scatter +
 *     `paged_attention_v1`). It is deliberately NOT made to implement
 *     this interface: its access model (one shared physical pool indexed
 *     by per-sequence block tables) is not a per-slot `KvCache` provider.
 *     The serving loop selects the substrate per backend; the paged path
 *     stays untouched (ADR Option C).
 *
 * F32 baseline today (matches the KvCache F32-only serving path); the
 * dtype is a property of the concrete substrate, not this interface.
 */
class ServingKvSubstrate {
public:
    virtual ~ServingKvSubstrate() = default;

    /// Number of concurrent slots this substrate can serve (== maxBatch).
    [[nodiscard]] virtual std::size_t capacity() const noexcept = 0;

    /// Per-slot logical context cap in tokens (each slot's `KvCache`
    /// `maxSeq`). A request bound to a slot may decode up to this length.
    [[nodiscard]] virtual std::size_t contextCap() const noexcept = 0;

    /// Number of KV-holding layers each slot covers.
    [[nodiscard]] virtual std::size_t numLayers() const noexcept = 0;

    /// The per-slot `KvCache` for slot `i` (`i < capacity()`).
    /// Implementations throw `std::out_of_range` on an invalid slot.
    [[nodiscard]] virtual KvCache&       slot(std::size_t i)       = 0;
    [[nodiscard]] virtual const KvCache& slot(std::size_t i) const = 0;

    /// Reset slot `i` to length 0 — called when a new request is bound
    /// to the slot so it re-prefills from an empty cache.
    virtual void resetSlot(std::size_t i) noexcept = 0;
};

} // namespace mimirmind::runtime::serving