// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Stefan Werfling

#include "runtime/KvCache.hpp"

#include "core/log/Log.hpp"
#include "core/gpu/l0/UsmAllocator.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace mimirmind::runtime {

KvCache::KvCache(UsmAllocator&            alloc,
                 std::size_t              maxSeq,
                 std::vector<std::size_t> kvDimPerLayer,
                 std::vector<std::size_t> kvSourceLayer,
                 KvDtype                  dtype)
    : _alloc{alloc},
      _maxSeq{maxSeq},
      _kvDim{std::move(kvDimPerLayer)},
      _dtype{dtype},
      _kvSource{std::move(kvSourceLayer)} {
    allocateLayers();
}

KvCache::KvCache(UsmAllocator& alloc,
                 std::size_t   nLayers,
                 std::size_t   maxSeq,
                 std::size_t   nKvHeads,
                 std::size_t   headDim,
                 KvDtype       dtype)
    : _alloc{alloc},
      _maxSeq{maxSeq},
      _kvDim(nLayers, nKvHeads * headDim),
      _dtype{dtype} {
    allocateLayers();
}

namespace {

const char* dtypeName(KvDtype d) noexcept {
    switch (d) {
        case KvDtype::F32:  return "f32";
        case KvDtype::FP16: return "fp16";
        case KvDtype::Q8_0: return "q8_0";
    }
    return "?";
}

} // namespace

void KvCache::allocateLayers() {
    const std::size_t nLayers = _kvDim.size();
    _layerBytes.assign(nLayers, 0);
    _kBuf.assign(nLayers, nullptr);
    _vBuf.assign(nLayers, nullptr);

    // Identity fallback so downstream code can always read _kvSource[L].
    if (_kvSource.empty()) {
        _kvSource.resize(nLayers);
        for (std::size_t i = 0; i < nLayers; ++i) _kvSource[i] = i;
    }

    // M10.2 Phase 1a — Q8_0 stores in 32-element blocks. Every layer's
    // kvDim must be a multiple of 32 so no block spans a fraction of a
    // token's row. Enforced at construction so misconfigured models fail
    // loudly instead of silently corrupting adjacent rows.
    if (_dtype == KvDtype::Q8_0) {
        const std::size_t blkE = kvBlockElements(_dtype);
        for (std::size_t i = 0; i < nLayers; ++i) {
            if (_kvDim[i] % blkE != 0) {
                throw std::runtime_error(
                    "KvCache: layer " + std::to_string(i) +
                    " kvDim=" + std::to_string(_kvDim[i]) +
                    " is not a multiple of " + std::to_string(blkE) +
                    " — Q8_0 storage requires block-aligned rows. "
                    "Use dtype=fp16 or f32 for this model, or verify "
                    "the backend's kvDimPerLayer() computation.");
            }
        }
    }

    std::size_t ownedBytes  = 0;
    std::size_t savedBytes  = 0;
    std::size_t ownCount    = 0;
    std::size_t aliasCount  = 0;
    std::size_t minDim = _kvDim.empty() ? 0 : _kvDim[0];
    std::size_t maxDim = minDim;
    for (std::size_t i = 0; i < nLayers; ++i) {
        const std::size_t src = _kvSource[i];
        // Row-major footprint per token, dtype-aware. For F32/FP16 the
        // formula reduces to `_kvDim[i] * elementBytes`; for Q8_0 it is
        // `(_kvDim[i] / 32) * 34`. `_maxSeq` rows fit end to end in the
        // per-layer allocation.
        const std::size_t bytesPerRow =
            kvBytesForElements(_dtype, _kvDim[i]);
        const std::size_t bytes = _maxSeq * bytesPerRow;
        minDim = std::min(minDim, _kvDim[i]);
        maxDim = std::max(maxDim, _kvDim[i]);
        if (src == i) {
            _layerBytes[i] = bytes;
            _kBuf[i] = _alloc.allocate(bytes);
            _vBuf[i] = _alloc.allocate(bytes);
            ownedBytes += 2 * bytes;
            ++ownCount;
        } else {
            // Source must have been visited already (src < i) — the
            // backend guarantees this via the shared-KV offset math.
            _layerBytes[i] = 0;                // marker: don't dealloc
            _kBuf[i] = _kBuf[src];
            _vBuf[i] = _vBuf[src];
            savedBytes += 2 * bytes;
            ++aliasCount;
        }
    }
    const double totalMiB =
        static_cast<double>(ownedBytes) / (1024.0 * 1024.0);
    const double savedMiB =
        static_cast<double>(savedBytes) / (1024.0 * 1024.0);
    MM_LOG_INFO("kvcache",
                "allocated nLayers={} maxSeq={} kvDim={}{} dtype={} "
                "(block {}B × {}elem) own={} aliased={} total {:.2f} MiB "
                "(saved {:.2f} MiB via alias)",
                nLayers, _maxSeq,
                minDim,
                (minDim == maxDim
                     ? std::string{}
                     : std::string{".."} + std::to_string(maxDim)),
                dtypeName(_dtype),
                kvBlockBytes(_dtype), kvBlockElements(_dtype),
                ownCount, aliasCount, totalMiB, savedMiB);
}

KvCache::~KvCache() {
    for (std::size_t i = 0; i < _kBuf.size(); ++i) {
        if (_layerBytes[i] == 0) continue;   // aliased — source owns
        _alloc.deallocate(_kBuf[i], _layerBytes[i]);
        _alloc.deallocate(_vBuf[i], _layerBytes[i]);
    }
}

void* KvCache::writeSlotK(std::size_t layer) noexcept {
    // Dtype-aware byte stride per row. For F32/FP16 this is
    // kvDim*elementBytes; for Q8_0 it is (kvDim/32)*34. Same call
    // for all three dtypes keeps the slot pointer correct without
    // per-dtype branches at every call site.
    const std::size_t byteOffset =
        _length * kvBytesForElements(_dtype, _kvDim[layer]);
    return static_cast<std::byte*>(_kBuf[layer]) + byteOffset;
}

void* KvCache::writeSlotV(std::size_t layer) noexcept {
    const std::size_t byteOffset =
        _length * kvBytesForElements(_dtype, _kvDim[layer]);
    return static_cast<std::byte*>(_vBuf[layer]) + byteOffset;
}

const void* KvCache::baseK(std::size_t layer) const noexcept {
    return _kBuf[layer];
}

const void* KvCache::baseV(std::size_t layer) const noexcept {
    return _vBuf[layer];
}

} // namespace mimirmind::runtime