#include "runtime/KvCache.hpp"

#include "runtime/Log.hpp"
#include "runtime/UsmAllocator.hpp"

#include <algorithm>
#include <utility>

namespace mimirmind::runtime {

KvCache::KvCache(UsmAllocator&            alloc,
                 std::size_t              maxSeq,
                 std::vector<std::size_t> kvDimPerLayer,
                 std::vector<std::size_t> kvSourceLayer)
    : _alloc{alloc},
      _maxSeq{maxSeq},
      _kvDim{std::move(kvDimPerLayer)},
      _kvSource{std::move(kvSourceLayer)} {
    allocateLayers();
}

KvCache::KvCache(UsmAllocator& alloc,
                 std::size_t   nLayers,
                 std::size_t   maxSeq,
                 std::size_t   nKvHeads,
                 std::size_t   headDim)
    : _alloc{alloc},
      _maxSeq{maxSeq},
      _kvDim(nLayers, nKvHeads * headDim) {
    allocateLayers();
}

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

    std::size_t ownedBytes  = 0;
    std::size_t savedBytes  = 0;
    std::size_t ownCount    = 0;
    std::size_t aliasCount  = 0;
    std::size_t minDim = _kvDim.empty() ? 0 : _kvDim[0];
    std::size_t maxDim = minDim;
    for (std::size_t i = 0; i < nLayers; ++i) {
        const std::size_t src = _kvSource[i];
        const std::size_t bytes = _maxSeq * _kvDim[i] * sizeof(float);
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
                "allocated nLayers={} maxSeq={} kvDim={}{} own={} "
                "aliased={} total {:.2f} MiB (saved {:.2f} MiB via alias)",
                nLayers, _maxSeq,
                minDim,
                (minDim == maxDim
                     ? std::string{}
                     : std::string{".."} + std::to_string(maxDim)),
                ownCount, aliasCount, totalMiB, savedMiB);
}

KvCache::~KvCache() {
    for (std::size_t i = 0; i < _kBuf.size(); ++i) {
        if (_layerBytes[i] == 0) continue;   // aliased — source owns
        _alloc.deallocate(_kBuf[i], _layerBytes[i]);
        _alloc.deallocate(_vBuf[i], _layerBytes[i]);
    }
}

float* KvCache::writeSlotK(std::size_t layer) noexcept {
    return static_cast<float*>(_kBuf[layer]) + _length * _kvDim[layer];
}

float* KvCache::writeSlotV(std::size_t layer) noexcept {
    return static_cast<float*>(_vBuf[layer]) + _length * _kvDim[layer];
}

const float* KvCache::baseK(std::size_t layer) const noexcept {
    return static_cast<const float*>(_kBuf[layer]);
}

const float* KvCache::baseV(std::size_t layer) const noexcept {
    return static_cast<const float*>(_vBuf[layer]);
}

} // namespace mimirmind::runtime