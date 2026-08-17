// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/serving/KvCacheSlabPool.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace mimirmind::runtime::serving {

KvCacheSlabPool::KvCacheSlabPool(compute::ComputeOps&     ops,
                                 std::size_t              maxBatch,
                                 std::size_t              contextCap,
                                 std::vector<std::size_t> kvDimPerLayer,
                                 std::vector<std::size_t> kvSourceLayer,
                                 KvDtype                  dtype)
    : _contextCap(contextCap),
      _numLayers(kvDimPerLayer.size()),
      _dtype(dtype)
{
    if (maxBatch == 0) {
        throw std::invalid_argument("KvCacheSlabPool: maxBatch must be > 0");
    }
    if (contextCap == 0) {
        throw std::invalid_argument("KvCacheSlabPool: contextCap must be > 0");
    }
    if (kvDimPerLayer.empty()) {
        throw std::invalid_argument(
            "KvCacheSlabPool: kvDimPerLayer must not be empty");
    }

    _slots.reserve(maxBatch);
    for (std::size_t s = 0; s < maxBatch; ++s) {
        // One contiguous per-slot cache. kvDimPerLayer / kvSourceLayer are
        // copied per slot so each slab owns its own device storage (or the
        // KvCache-internal aliasing for shared-KV layers).
        _slots.push_back(std::make_unique<KvCache>(
            ops, contextCap, kvDimPerLayer, kvSourceLayer, dtype));
    }
}

KvCache& KvCacheSlabPool::slot(std::size_t i) {
    if (i >= _slots.size()) {
        throw std::out_of_range(
            "KvCacheSlabPool::slot: index " + std::to_string(i) +
            " >= capacity " + std::to_string(_slots.size()));
    }
    return *_slots[i];
}

const KvCache& KvCacheSlabPool::slot(std::size_t i) const {
    if (i >= _slots.size()) {
        throw std::out_of_range(
            "KvCacheSlabPool::slot: index " + std::to_string(i) +
            " >= capacity " + std::to_string(_slots.size()));
    }
    return *_slots[i];
}

void KvCacheSlabPool::resetSlot(std::size_t i) noexcept {
    // noexcept per the interface; a bad index is a caller bug — clamp
    // silently rather than throw across the serving loop boundary.
    if (i < _slots.size()) {
        _slots[i]->reset();
    }
}

void KvCacheSlabPool::activeSlotCaches(std::size_t             nSeq,
                                       std::vector<KvCache*>&  out) {
    if (nSeq > _slots.size()) {
        throw std::out_of_range(
            "KvCacheSlabPool::activeSlotCaches: nSeq " + std::to_string(nSeq) +
            " > capacity " + std::to_string(_slots.size()));
    }
    out.clear();
    out.reserve(nSeq);
    for (std::size_t s = 0; s < nSeq; ++s) {
        out.push_back(_slots[s].get());
    }
}

std::size_t KvCacheSlabPool::kvDim(std::size_t layer) const noexcept {
    // All slabs share the same per-layer geometry; slot 0 is representative.
    return _slots.empty() ? 0 : _slots.front()->kvDim(layer);
}

} // namespace mimirmind::runtime::serving