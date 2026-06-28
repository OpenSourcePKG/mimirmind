#include "runtime/KvCache.hpp"

#include "runtime/Log.hpp"
#include "runtime/UsmAllocator.hpp"

namespace mimirmind::runtime {

KvCache::KvCache(UsmAllocator& alloc,
                 std::size_t   nLayers,
                 std::size_t   maxSeq,
                 std::size_t   nKvHeads,
                 std::size_t   headDim)
    : _alloc{alloc},
      _nLayers{nLayers},
      _maxSeq{maxSeq},
      _nKvHeads{nKvHeads},
      _headDim{headDim}
{
    _kvDim      = _nKvHeads * _headDim;
    _layerBytes = _maxSeq * _kvDim * sizeof(float);

    _kBuf.reserve(_nLayers);
    _vBuf.reserve(_nLayers);
    for (std::size_t i = 0; i < _nLayers; ++i) {
        _kBuf.push_back(_alloc.allocate(_layerBytes));
        _vBuf.push_back(_alloc.allocate(_layerBytes));
    }

    const double totalMiB =
        static_cast<double>(2 * _nLayers * _layerBytes) / (1024.0 * 1024.0);
    MM_LOG_INFO("kvcache",
                "allocated nLayers={} maxSeq={} nKvHeads={} headDim={} "
                "-> {} bytes/layer per side, total {:.2f} MiB",
                _nLayers, _maxSeq, _nKvHeads, _headDim,
                _layerBytes, totalMiB);
}

KvCache::~KvCache() {
    for (std::size_t i = 0; i < _nLayers; ++i) {
        _alloc.deallocate(_kBuf[i], _layerBytes);
        _alloc.deallocate(_vBuf[i], _layerBytes);
    }
}

float* KvCache::writeSlotK(std::size_t layer) noexcept {
    return static_cast<float*>(_kBuf[layer]) + _length * _kvDim;
}

float* KvCache::writeSlotV(std::size_t layer) noexcept {
    return static_cast<float*>(_vBuf[layer]) + _length * _kvDim;
}

const float* KvCache::baseK(std::size_t layer) const noexcept {
    return static_cast<const float*>(_kBuf[layer]);
}

const float* KvCache::baseV(std::size_t layer) const noexcept {
    return static_cast<const float*>(_vBuf[layer]);
}

} // namespace mimirmind::runtime