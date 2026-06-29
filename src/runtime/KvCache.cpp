#include "runtime/KvCache.hpp"

#include "runtime/Log.hpp"
#include "runtime/UsmAllocator.hpp"

#include <algorithm>
#include <utility>

namespace mimirmind::runtime {

KvCache::KvCache(UsmAllocator&            alloc,
                 std::size_t              maxSeq,
                 std::vector<std::size_t> kvDimPerLayer)
    : _alloc{alloc},
      _maxSeq{maxSeq},
      _kvDim{std::move(kvDimPerLayer)} {
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
    _layerBytes.resize(nLayers);
    _kBuf.reserve(nLayers);
    _vBuf.reserve(nLayers);
    std::size_t totalBytes = 0;
    std::size_t minDim = _kvDim.empty() ? 0 : _kvDim[0];
    std::size_t maxDim = minDim;
    for (std::size_t i = 0; i < nLayers; ++i) {
        _layerBytes[i] = _maxSeq * _kvDim[i] * sizeof(float);
        _kBuf.push_back(_alloc.allocate(_layerBytes[i]));
        _vBuf.push_back(_alloc.allocate(_layerBytes[i]));
        totalBytes += 2 * _layerBytes[i];
        minDim = std::min(minDim, _kvDim[i]);
        maxDim = std::max(maxDim, _kvDim[i]);
    }
    const double totalMiB =
        static_cast<double>(totalBytes) / (1024.0 * 1024.0);
    MM_LOG_INFO("kvcache",
                "allocated nLayers={} maxSeq={} kvDim={}{} total {:.2f} MiB",
                nLayers, _maxSeq,
                minDim,
                (minDim == maxDim
                     ? std::string{}
                     : std::string{".."} + std::to_string(maxDim)),
                totalMiB);
}

KvCache::~KvCache() {
    for (std::size_t i = 0; i < _kBuf.size(); ++i) {
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